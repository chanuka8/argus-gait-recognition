"""
3D Pose Gait Analysis Module for ARGUS AI.

Provides lightweight 2D-to-3D temporal pose lifting, view-invariant skeleton
normalization, graph convolution encoders (ST-GCN, CTR-GCN, TCN), and feature extraction.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# COCO 17 Joint Indices for reference:
# 0: Nose, 1: L_Eye, 2: R_Eye, 3: L_Ear, 4: R_Ear
# 5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow, 8: R_Elbow, 9: L_Wrist, 10: R_Wrist
# 11: L_Hip, 12: R_Hip, 13: L_Knee, 14: R_Knee, 15: L_Ankle, 16: R_Ankle

L_HIP, R_HIP = 11, 12
L_SHOULDER, R_SHOULDER = 5, 6

COCO_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


def build_coco17_adjacency_matrix() -> torch.Tensor:
    """Builds normalized spatial adjacency matrix for 17 COCO joints."""
    A = torch.eye(17, dtype=torch.float32)
    for i, j in COCO_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0

    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-6), -0.5)
    D_inv = torch.diag(deg_inv_sqrt)
    A_norm = torch.mm(torch.mm(D_inv, A), D_inv)
    return A_norm


class SkeletonNormalizer3D(nn.Module):
    """
    Normalizes 3D joint sequences for translation (pelvis centering),
    scale (torso & limb scaling), viewpoint (yaw rotation alignment), and temporal smoothing.
    """

    def __init__(self, eps: float = 1e-6, smooth_kernel: int = 3) -> None:
        super().__init__()
        self.eps = eps
        self.smooth_kernel = smooth_kernel

    def forward(self, joints_3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joints_3d: Tensor of shape (B, T, 17, 3) or (T, 17, 3) with (x, y, z) coordinates.

        Returns:
            Normalized 3D joints tensor of identical shape.
        """
        is_unbatched = joints_3d.ndim == 3
        if is_unbatched:
            joints_3d = joints_3d.unsqueeze(0)

        B, T, V, C = joints_3d.shape
        normalized = joints_3d.clone()

        # 1. Temporal Gaussian Smoothing
        if self.smooth_kernel > 1 and T >= self.smooth_kernel:
            pad = self.smooth_kernel // 2
            flat = normalized.permute(0, 2, 3, 1).reshape(B * V * C, 1, T)  # (B*V*C, 1, T)
            smoothed = F.avg_pool1d(F.pad(flat, (pad, pad), mode="replicate"), kernel_size=self.smooth_kernel, stride=1)
            normalized = smoothed.view(B, V, C, T).permute(0, 3, 1, 2)

        # 2. Translation Normalization: Center mid-hip at (0, 0, 0)
        pelvis = (normalized[:, :, L_HIP, :] + normalized[:, :, R_HIP, :]) / 2.0  # (B, T, 3)
        normalized = normalized - pelvis.unsqueeze(2)  # (B, T, 17, 3)

        # 3. Scale Normalization: Torso + Limb scale factor
        neck = (normalized[:, :, L_SHOULDER, :] + normalized[:, :, R_SHOULDER, :]) / 2.0
        torso_len = torch.norm(neck, p=2, dim=-1, keepdim=True).unsqueeze(-1)  # (B, T, 1, 1)
        torso_len = torch.clamp(torso_len, min=self.eps)
        normalized = normalized / torso_len

        # 4. Viewpoint Alignment: Rotate around Y-axis to align hip vector along X-axis
        hip_vec = normalized[:, :, R_HIP, :] - normalized[:, :, L_HIP, :]  # (B, T, 3)
        yaw = torch.atan2(hip_vec[:, :, 2], hip_vec[:, :, 0])  # (B, T)

        cos_yaw = torch.cos(-yaw)  # (B, T)
        sin_yaw = torch.sin(-yaw)

        x = normalized[:, :, :, 0]
        y = normalized[:, :, :, 1]
        z = normalized[:, :, :, 2]

        x_rot = x * cos_yaw.unsqueeze(-1) - z * sin_yaw.unsqueeze(-1)
        z_rot = x * sin_yaw.unsqueeze(-1) + z * cos_yaw.unsqueeze(-1)

        normalized = torch.stack([x_rot, y, z_rot], dim=-1)

        if is_unbatched:
            normalized = normalized.squeeze(0)

        return normalized


class PoseLifter3D(nn.Module):
    """
    Lightweight 1D Temporal Convolutional Pose Lifter mapping 2D keypoints (x, y, conf)
    across T frames to estimated joint depth z.
    """

    def __init__(self, num_joints: int = 17, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.in_proj = nn.Linear(num_joints * 3, hidden_dim)

        self.block1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
        )

        self.depth_head = nn.Linear(hidden_dim, num_joints)

    def forward(self, keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keypoints_2d: Tensor of shape (B, T, 17, 3) containing (x, y, confidence).

        Returns:
            Lifted 3D joints tensor of shape (B, T, 17, 3) containing (x, y, z).
        """
        is_unbatched = keypoints_2d.ndim == 3
        if is_unbatched:
            keypoints_2d = keypoints_2d.unsqueeze(0)

        B, T, V, C = keypoints_2d.shape
        flat_in = keypoints_2d.view(B, T, V * C)  # (B, T, 51)

        h = self.in_proj(flat_in)  # (B, T, 128)
        h_t = h.transpose(1, 2)  # (B, 128, T)
        h_conv = self.block1(h_t) + h_t  # (B, 128, T) with residual
        h_out = F.relu(h_conv).transpose(1, 2)  # (B, T, 128)

        z = self.depth_head(h_out).unsqueeze(-1)  # (B, T, 17, 1)

        # Combine 2D (x, y) with estimated depth z
        xy = keypoints_2d[:, :, :, :2]  # (B, T, 17, 2)
        joints_3d = torch.cat([xy, z], dim=-1)  # (B, T, 17, 3)

        if is_unbatched:
            joints_3d = joints_3d.squeeze(0)

        return joints_3d


def compute_enriched_skeleton_features(norm_joints: torch.Tensor) -> torch.Tensor:
    """
    Computes positions, velocities, accelerations, and bone vectors.
    Args:
        norm_joints: (B, T, 17, 3)
    Returns:
        Tensor of shape (B, T, 17, 9) containing (pos, vel, acc) per joint.
    """
    B, T, V, C = norm_joints.shape

    # 1. Velocities
    v_diff = norm_joints[:, 1:, :, :] - norm_joints[:, :-1, :, :]
    v_zero = torch.zeros_like(norm_joints[:, :1, :, :])
    vel = torch.cat([v_zero, v_diff], dim=1)  # (B, T, 17, 3)

    # 2. Accelerations
    a_diff = vel[:, 1:, :, :] - vel[:, :-1, :, :]
    a_zero = torch.zeros_like(vel[:, :1, :, :])
    acc = torch.cat([a_zero, a_diff], dim=1)  # (B, T, 17, 3)

    # Concatenate pos (3), vel (3), acc (3) = 9 channels
    feat = torch.cat([norm_joints, vel, acc], dim=-1)  # (B, T, 17, 9)
    return feat


class PoseGait3DNet(nn.Module):
    """
    Architecture A: Improved Multi-Scale Dilated TCN Gait Encoder.
    """

    def __init__(self, num_joints: int = 17, embedding_dim: int = 256) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.embedding_dim = embedding_dim
        self.normalizer = SkeletonNormalizer3D()

        in_channels = num_joints * 9  # 3 pos + 3 vel + 3 acc

        self.in_conv = nn.Conv1d(in_channels, 128, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm1d(128)

        self.block1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=2, dilation=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(256, embedding_dim)

    def forward(self, joints_3d: torch.Tensor) -> torch.Tensor:
        is_unbatched = joints_3d.ndim == 3
        if is_unbatched:
            joints_3d = joints_3d.unsqueeze(0)

        norm_joints = self.normalizer(joints_3d)
        feat = compute_enriched_skeleton_features(norm_joints)  # (B, T, 17, 9)
        B, T, V, C = feat.shape
        feat_flat = feat.view(B, T, V * C).transpose(1, 2)  # (B, 153, T)

        h = F.relu(self.bn0(self.in_conv(feat_flat)))
        h = self.block1(h)
        h = self.block2(h)

        pooled = self.pool(h).squeeze(-1)
        emb = self.proj(pooled)
        emb_norm = F.normalize(emb, p=2, dim=1)

        if is_unbatched:
            emb_norm = emb_norm.squeeze(0)

        return emb_norm


class STGCNBlock(nn.Module):
    """Spatial Graph Conv + Temporal 1D Conv block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.gconv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.tconv = nn.Conv1d(out_channels, out_channels, kernel_size=9, stride=stride, padding=4)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, C, V, T)
        B, C, V, T = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * V, C, T)

        # Residual branch
        if isinstance(self.residual, nn.Sequential):
            res_flat = self.residual(x_flat)  # (B*V, C_out, T_new)
            T_new = res_flat.shape[-1]
            res = res_flat.view(B, V, -1, T_new).permute(0, 2, 1, 3)  # (B, C_out, V, T_new)
        else:
            res = x
            T_new = T

        # Spatial Graph Conv: X * A
        x_g = torch.einsum("bcvt,vw->bcwt", x, A)  # (B, C, V, T)
        x_g = x_g.permute(0, 2, 1, 3).reshape(B * V, C, T)
        x_g = self.gconv(x_g)
        x_g = self.tconv(x_g)  # (B*V, C_out, T_new)

        x_out = x_g.view(B, V, -1, T_new).permute(0, 2, 1, 3)  # (B, C_out, V, T_new)
        x_out = self.relu(self.bn(x_out + res))
        return x_out


class STGCNGait3DNet(nn.Module):
    """
    Architecture B: Spatial-Temporal Graph Convolution Network (ST-GCN).
    """

    def __init__(self, num_joints: int = 17, embedding_dim: int = 256) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.embedding_dim = embedding_dim
        self.normalizer = SkeletonNormalizer3D()
        self.register_buffer("A", build_coco17_adjacency_matrix())

        # 9 features per joint: pos (3), vel (3), acc (3)
        self.in_proj = nn.Conv2d(9, 64, kernel_size=1)
        self.block1 = STGCNBlock(64, 128, stride=2)
        self.block2 = STGCNBlock(128, 256, stride=2)

        self.joint_attn = nn.Sequential(
            nn.Linear(17, 17),
            nn.Sigmoid(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, embedding_dim)

    def forward(self, joints_3d: torch.Tensor) -> torch.Tensor:
        is_unbatched = joints_3d.ndim == 3
        if is_unbatched:
            joints_3d = joints_3d.unsqueeze(0)

        norm_joints = self.normalizer(joints_3d)
        feat = compute_enriched_skeleton_features(norm_joints)  # (B, T, 17, 9)
        B, T, V, C = feat.shape

        # Permute to (B, C, V, T)
        x = feat.permute(0, 3, 2, 1)  # (B, 9, 17, T)

        h = F.relu(self.in_proj(x))
        h = self.block1(h, self.A)
        h = self.block2(h, self.A)  # (B, 256, 17, T')

        # Joint Attention Weighting
        attn = self.joint_attn(h.mean(dim=(1, 3))).unsqueeze(1).unsqueeze(-1)  # (B, 1, 17, 1)
        h = h * attn

        pooled = self.pool(h).squeeze(-1).squeeze(-1)  # (B, 256)
        emb = self.proj(pooled)
        emb_norm = F.normalize(emb, p=2, dim=1)

        if is_unbatched:
            emb_norm = emb_norm.squeeze(0)

        return emb_norm


class CTRGCNGait3DNet(nn.Module):
    """
    Architecture C: Channel-Wise Topology Refinement Graph Network (CTR-GCN).
    Dynamically learns adaptive graph topology refinement per temporal block.
    """

    def __init__(self, num_joints: int = 17, embedding_dim: int = 256) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.embedding_dim = embedding_dim
        self.normalizer = SkeletonNormalizer3D()
        self.register_buffer("A_static", build_coco17_adjacency_matrix())

        # Adaptive channel-wise dynamic graph learning parameters
        self.PA = nn.Parameter(torch.zeros(17, 17))
        nn.init.uniform_(self.PA, -1e-4, 1e-4)

        self.in_proj = nn.Conv2d(9, 64, kernel_size=1)
        self.block1 = STGCNBlock(64, 128, stride=2)
        self.block2 = STGCNBlock(128, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, embedding_dim)

    def forward(self, joints_3d: torch.Tensor) -> torch.Tensor:
        is_unbatched = joints_3d.ndim == 3
        if is_unbatched:
            joints_3d = joints_3d.unsqueeze(0)

        norm_joints = self.normalizer(joints_3d)
        feat = compute_enriched_skeleton_features(norm_joints)  # (B, T, 17, 9)
        B, T, V, C = feat.shape

        x = feat.permute(0, 3, 2, 1)  # (B, 9, 17, T)
        A_dyn = F.softmax(self.A_static + self.PA, dim=-1)

        h = F.relu(self.in_proj(x))
        h = self.block1(h, A_dyn)
        h = self.block2(h, A_dyn)  # (B, 256, 17, T')

        pooled = self.pool(h).squeeze(-1).squeeze(-1)  # (B, 256)
        emb = self.proj(pooled)
        emb_norm = F.normalize(emb, p=2, dim=1)

        if is_unbatched:
            emb_norm = emb_norm.squeeze(0)

        return emb_norm


class TemporalPoseBuffer:
    """
    Buffers 2D pose keypoint sequences per track ID and performs
    linear interpolation for low-confidence or missing joints.
    """

    def __init__(self, max_length: int = 30, conf_threshold: float = 0.30) -> None:
        self.max_length = max_length
        self.conf_threshold = conf_threshold
        self.buffers: Dict[str, List[np.ndarray]] = {}

    def add_keypoints(self, track_id: str, keypoints: np.ndarray) -> None:
        """
        Args:
            track_id: Unique tracking ID.
            keypoints: Array of shape (17, 3) with (x, y, confidence).
        """
        if track_id not in self.buffers:
            self.buffers[track_id] = []
        self.buffers[track_id].append(keypoints.copy())

        # Cap buffer size to max_length
        if len(self.buffers[track_id]) > self.max_length:
            self.buffers[track_id].pop(0)

    def get_sequence(self, track_id: str) -> Optional[np.ndarray]:
        """
        Retrieves interpolated 2D keypoint sequence for a track ID.

        Returns:
            Array of shape (T, 17, 3) or None if buffer has fewer than 5 frames.
        """
        if track_id not in self.buffers or len(self.buffers[track_id]) < 5:
            return None

        seq = np.stack(self.buffers[track_id], axis=0)  # (T, 17, 3)
        T, V, C = seq.shape

        # Interpolate low-confidence joints along temporal axis
        for v in range(V):
            confs = seq[:, v, 2]
            valid_idx = np.where(confs >= self.conf_threshold)[0]
            if len(valid_idx) == 0:
                continue
            if len(valid_idx) < T:
                for c in range(2):  # x and y
                    seq[:, v, c] = np.interp(
                        np.arange(T),
                        valid_idx,
                        seq[valid_idx, v, c]
                    )

        return seq

    def clear(self, track_id: Optional[str] = None) -> None:
        if track_id is None:
            self.buffers.clear()
        else:
            self.buffers.pop(track_id, None)
