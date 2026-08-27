"""
Optional 3D Pose Gait Extraction Step for ARGUS AI Pipeline.

Converts tracking box crops -> 2D keypoints -> Temporal 3D pose lifting -> Normalized 3D joint sequence -> 256-D 3D Gait Embedding.
Disabled by default (`enabled=False`). Falls back gracefully to 2D GEI gait matching when disabled or unavailable.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from models.architectures.pose_gait_3d import (
    PoseGait3DNet,
    PoseLifter3D,
    TemporalPoseBuffer,
)


class Gait3DStep:
    """
    Independent 3D Pose Gait Analysis Step.
    """

    def __init__(
        self,
        enabled: bool = False,
        pose_model_path: str = "models/weights/yolov8n-pose.pt",
        weights_path: str | None = "runs/exp_006_3d/best_model.pth",
        sequence_length: int = 30,
        conf_threshold: float = 0.30,
        device: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.sequence_length = sequence_length
        self.conf_threshold = conf_threshold
        from automation.device_manager import DeviceManager

        self.device = DeviceManager.get_instance().resolve_component_device(device)

        self.pose_estimator = None
        self.pose_lifter = None
        self.gait_net = None
        self.pose_buffer = None

        if self.enabled:
            self._initialize_models(pose_model_path, weights_path)

    def _initialize_models(self, pose_model_path: str, weights_path: str | None) -> None:
        """Loads YOLOv8-pose, PoseLifter3D, and PoseGait3DNet onto configured device."""
        try:
            from ultralytics import YOLO

            self.pose_estimator = YOLO(pose_model_path)
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            print(f"[WARN] 3D Gait: Failed to load pose estimator ({e}). Disabling 3D Gait.")
            self.enabled = False
            return

        self.pose_lifter = PoseLifter3D().to(self.device).eval()
        self.gait_net = PoseGait3DNet(embedding_dim=256).to(self.device).eval()
        self.pose_buffer = TemporalPoseBuffer(max_length=self.sequence_length, conf_threshold=self.conf_threshold)

        if weights_path:
            w_path = Path(weights_path)
            if not w_path.exists():
                print(f"[WARN] 3D Gait: Checkpoint path does not exist ({weights_path}). Disabling 3D Gait.")
                self.enabled = False
                return

            try:
                ckpt = torch.load(w_path, map_location=self.device)
                if isinstance(ckpt, dict):
                    if "embedding_dim" in ckpt and ckpt["embedding_dim"] != 256:
                        raise ValueError(f"Checkpoint embedding dimension mismatch: {ckpt['embedding_dim']} != 256")

                    if "lifter" in ckpt:
                        self.pose_lifter.load_state_dict(ckpt["lifter"])
                    if "gait_net" in ckpt:
                        self.gait_net.load_state_dict(ckpt["gait_net"])
            except (RuntimeError, ValueError, OSError, EOFError) as e:
                print(f"[WARN] 3D Gait: Failed to load checkpoint weights ({e}). Disabling 3D Gait.")
                self.enabled = False

    def process_frame_crop(self, track_id: str, crop: np.ndarray) -> np.ndarray | None:
        """
        Extracts 2D pose from frame crop, updates track buffer, and returns 3D gait embedding if buffer full.

        Args:
            track_id: Unique tracking identity ID string.
            crop: BGR image crop of detected person.

        Returns:
            256-D L2-normalized 3D gait embedding numpy array, or None if disabled/buffering.
        """
        if not self.enabled or self.pose_estimator is None or crop is None or crop.size == 0:
            return None

        if crop.ndim == 2 or crop.shape[2] == 1:
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        else:
            crop_bgr = crop

        results = self.pose_estimator(crop_bgr, conf=0.05, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return None

        xy = results[0].keypoints.xy[0].cpu().numpy()
        conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)

        h, w = crop.shape[:2]
        xy_norm = xy / np.array([max(w, 1), max(h, 1)])
        kpts_2d = np.concatenate([xy_norm, conf[:, None]], axis=-1)

        self.pose_buffer.add_keypoints(track_id, kpts_2d)
        seq_2d = self.pose_buffer.get_sequence(track_id)

        if seq_2d is None:
            return None

        with torch.no_grad():
            tensor_2d = torch.from_numpy(seq_2d).float().unsqueeze(0).to(self.device)
            joints_3d = self.pose_lifter(tensor_2d)
            emb = self.gait_net(joints_3d).squeeze(0).cpu().numpy()

        return emb.astype(np.float32)

    def prune_stale_tracks(self, active_track_ids: list[str]) -> None:
        """Removes track buffers for track IDs no longer active in tracker."""
        if self.pose_buffer is not None:
            active_set = set(active_track_ids)
            stale_keys = [k for k in self.pose_buffer.buffers if k not in active_set]
            for k in stale_keys:
                self.pose_buffer.clear(k)

    def reset_track(self, track_id: str | None = None) -> None:
        if self.pose_buffer is not None:
            self.pose_buffer.clear(track_id)
