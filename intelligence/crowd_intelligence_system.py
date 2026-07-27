"""
Unified Crowd Intelligence System.

Integrates:
  1. Stage 1: Crowd-Aware Occlusion Handling (CrowdOcclusionAnalyzer)
  2. Stage 2: Recognition Deferral & Evidence Accumulation (RecognitionDeferralEngine)
  3. Stage 3: Multi-Camera Evidence Fusion (MultiCameraEvidenceFusion)
  4. Stage 4: Automatic Camera Topology Learning (CameraTopologyLearner)

Disabled by default (`enabled: false`) for 100% backward compatibility.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import yaml

from intelligence.camera_topology_learner import CameraTopologyLearner
from intelligence.crowd_occlusion_analyzer import CrowdOcclusionAnalyzer, FrameCrowdAnalysis
from intelligence.multi_camera_evidence_fusion import MultiCameraEvidenceFusion
from intelligence.recognition_deferral_engine import RecognitionDeferralEngine
from monitoring.logging_config import get_logger


@dataclass
class CrowdIntelligenceEvaluation:
    crowd_density_level: str
    crowd_density_score: float
    track_occlusion_score: float
    clean_frame_ratio: float
    recognition_state: str
    recognition_deferred: bool
    defer_reason: str
    accumulated_evidence_count: int
    fused_identity: str
    fused_score: float
    fusion_state: str
    contributing_cameras: List[str]
    topology_observation_accepted: bool
    should_alert: bool


class CrowdIntelligenceSystem:
    """
    Unified Orchestrator for Crowd-Intelligence Features across single-camera and multi-camera pipelines.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.logger = get_logger("crowd_intelligence")
        cfg = config or self._load_default_config()

        self.enabled = bool(cfg.get("enabled", False))

        # Component configurations
        occ_cfg = cfg.get("occlusion", {})
        def_cfg = cfg.get("recognition_deferral", {})
        fus_cfg = cfg.get("multi_camera_fusion", {})
        top_cfg = cfg.get("topology_learning", {})

        # Sub-engines
        self.occlusion_analyzer = CrowdOcclusionAnalyzer(occ_cfg)
        self.deferral_engine = RecognitionDeferralEngine(def_cfg)
        self.fusion_engine = MultiCameraEvidenceFusion(fus_cfg)
        self.topology_learner = CameraTopologyLearner(top_cfg)

    def is_enabled(self) -> bool:
        return self.enabled

    @staticmethod
    def _load_default_config() -> Dict[str, Any]:
        config_path = Path("configs/inference.yaml")
        defaults = {
            "enabled": False,
            "occlusion": {
                "enabled": False,
                "smoothing_window": 5,
                "moderate_threshold": 0.35,
                "high_threshold": 0.60,
                "severe_threshold": 0.80,
                "minimum_clean_frames": 18,
                "minimum_clean_ratio": 0.70,
            },
            "recognition_deferral": {
                "enabled": False,
                "evidence_window": 5,
                "minimum_confirmations": 3,
                "minimum_reliability": 0.70,
                "evidence_ttl_seconds": 10.0,
                "identity_ttl_seconds": 5.0,
            },
            "multi_camera_fusion": {
                "enabled": False,
                "evidence_ttl_seconds": 15.0,
                "minimum_cameras": 2,
                "minimum_fused_score": 0.85,
                "weights": {
                    "gait": 0.30,
                    "appearance": 0.15,
                    "open_set": 0.15,
                    "temporal": 0.15,
                    "reliability": 0.15,
                    "transition": 0.10,
                },
            },
            "topology_learning": {
                "enabled": False,
                "shadow_mode": True,
                "minimum_samples": 20,
                "maximum_travel_seconds": 600.0,
                "export_path": "outputs/learned_camera_topology.yaml",
            },
        }

        if not config_path.exists():
            return defaults

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                ci_section = data.get("crowd_intelligence", {})
                if isinstance(ci_section, dict):
                    # Merge defaults recursively
                    for key, val in defaults.items():
                        if key not in ci_section:
                            ci_section[key] = val
                    return ci_section
        except Exception:
            pass

        return defaults

    def process_frame(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int] = (1080, 1920),
        camera_id: str = "cam_00",
        timestamp: Optional[float] = None,
    ) -> FrameCrowdAnalysis:
        """Stage 1: Process frame crowd density and per-track occlusion analysis."""
        return self.occlusion_analyzer.analyze_frame(
            detections=detections,
            frame_shape=frame_shape,
            camera_id=camera_id,
            timestamp=timestamp,
        )

    def evaluate_track_recognition(
        self,
        camera_id: str,
        track_id: Any,
        identity_candidate: str,
        similarity: float,
        quality: float,
        open_set_state: str,
        temporal_decision: str,
        reliability: float,
        occlusion_score: float = 0.0,
        clean_frame_count: int = 20,
        clean_frame_ratio: float = 1.0,
        global_track_id: Optional[str] = None,
        source_camera: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> CrowdIntelligenceEvaluation:
        """
        Evaluate Stage 2 (Deferral), Stage 3 (Fusion), and Stage 4 (Topology Learning) for a track.
        """
        now = timestamp if timestamp is not None else time.monotonic()
        entity_key = global_track_id or f"{camera_id}_{track_id}"

        # Baseline mode if overall crowd_intelligence is disabled
        if not self.enabled:
            is_confirmed = (open_set_state == "KNOWN" and similarity >= 0.85 and identity_candidate != "UNKNOWN")
            return CrowdIntelligenceEvaluation(
                crowd_density_level="LOW",
                crowd_density_score=0.0,
                track_occlusion_score=occlusion_score,
                clean_frame_ratio=clean_frame_ratio,
                recognition_state="CONFIRMED" if is_confirmed else open_set_state,
                recognition_deferred=False,
                defer_reason="",
                accumulated_evidence_count=1,
                fused_identity=identity_candidate if is_confirmed else "UNKNOWN",
                fused_score=similarity if is_confirmed else 0.0,
                fusion_state="CONFIRMED" if is_confirmed else "UNKNOWN",
                contributing_cameras=[camera_id],
                topology_observation_accepted=False,
                should_alert=is_confirmed,
            )

        # Stage 2: Deferral & Accumulation
        def_res = self.deferral_engine.evaluate_and_accumulate(
            camera_id=camera_id,
            track_id=track_id,
            identity_candidate=identity_candidate,
            similarity=similarity,
            quality=quality,
            open_set_state=open_set_state,
            temporal_decision=temporal_decision,
            reliability=reliability,
            occlusion=occlusion_score,
            clean_frame_count=clean_frame_count,
            clean_frame_ratio=clean_frame_ratio,
            timestamp=now,
        )

        # Stage 3: Multi-Camera Evidence Fusion
        if self.fusion_engine.is_enabled():
            self.fusion_engine.add_observation(
                camera_id=camera_id,
                local_track_id=track_id,
                global_track_id=global_track_id,
                identity_candidate=def_res.identity,
                gait_similarity=similarity,
                quality_score=quality,
                occlusion_score=occlusion_score,
                track_reliability=reliability,
                timestamp=now,
            )
            fusion_res = self.fusion_engine.fuse_evidence(
                entity_key=entity_key,
                fallback_identity=def_res.identity,
                fallback_score=def_res.confidence,
                current_time=now,
            )
        else:
            is_confirmed = def_res.recognition_state == "CONFIRMED"
            fusion_res = type("FusionRes", (), {
                "fused_identity": def_res.identity,
                "fused_score": def_res.confidence,
                "fusion_state": "CONFIRMED" if is_confirmed else "UNKNOWN",
                "contributing_cameras": [camera_id],
            })()

        # Stage 4: Topology Learning
        topology_accepted = False
        if self.topology_learner.is_enabled():
            self.topology_learner.record_camera_exit(
                camera_id=camera_id,
                identity=def_res.identity,
                reliability=reliability,
                occlusion=occlusion_score,
                timestamp=now,
            )
            if source_camera and source_camera != camera_id:
                topology_accepted = self.topology_learner.observe_transition(
                    source_camera=source_camera,
                    destination_camera=camera_id,
                    identity=def_res.identity,
                    reliability=reliability,
                    occlusion=occlusion_score,
                    is_known_identity=(open_set_state == "KNOWN"),
                    is_temporally_confirmed=(temporal_decision in ("MAJORITY_VOTE", "CONFIRMED")),
                    timestamp=now,
                )

        return CrowdIntelligenceEvaluation(
            crowd_density_level="MODERATE" if occlusion_score > 0.35 else "LOW",
            crowd_density_score=round(occlusion_score, 4),
            track_occlusion_score=round(occlusion_score, 4),
            clean_frame_ratio=round(clean_frame_ratio, 4),
            recognition_state=def_res.recognition_state.value if hasattr(def_res.recognition_state, "value") else str(def_res.recognition_state),
            recognition_deferred=def_res.recognition_deferred,
            defer_reason=def_res.defer_reason,
            accumulated_evidence_count=def_res.accumulated_evidence_count,
            fused_identity=fusion_res.fused_identity,
            fused_score=fusion_res.fused_score,
            fusion_state=fusion_res.fusion_state.value if hasattr(fusion_res.fusion_state, "value") else str(fusion_res.fusion_state),
            contributing_cameras=fusion_res.contributing_cameras,
            topology_observation_accepted=topology_accepted,
            should_alert=def_res.should_alert,
        )

    def cleanup_inactive(self, max_idle_seconds: float = 15.0, current_time: Optional[float] = None) -> None:
        """Periodic cleanup across all sub-engines."""
        now = current_time if current_time is not None else time.monotonic()
        self.occlusion_analyzer.cleanup_inactive(max_idle_seconds, now)
        self.deferral_engine.cleanup_inactive(max_idle_seconds, now)
        self.fusion_engine.cleanup_inactive(max_idle_seconds, now)
        self.topology_learner.cleanup_inactive(max_idle_seconds, now)
