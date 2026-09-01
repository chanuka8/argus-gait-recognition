from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from intelligence.concurrent_track_manager import (
    PersonTrackContext,
    TrackLifecycleState,
)
from monitoring.logging_config import get_logger


class AdaptivePersonLoadTier(str, enum.Enum):
    FULL_QUALITY = "FULL_QUALITY"
    REDUCED_PROCESSING_FPS = "REDUCED_PROCESSING_FPS"
    AGGRESSIVE_FRAME_SKIPPING = "AGGRESSIVE_FRAME_SKIPPING"
    MICRO_BATCHING = "MICRO_BATCHING"
    QUEUE_BACKPRESSURE = "QUEUE_BACKPRESSURE"
    DEGRADED_MODE = "DEGRADED_MODE"
    AUTOMATIC_RECOVERY = "AUTOMATIC_RECOVERY"


@dataclass
class SchedulerPolicyParameters:
    tier: AdaptivePersonLoadTier = AdaptivePersonLoadTier.FULL_QUALITY
    reid_update_interval: int = 8
    max_batch_size: int = 16
    skip_confirmed_reid: bool = False
    gait_extraction_interval: int = 10
    drop_stale_frames: bool = False
    max_tracks_per_batch: int = 32
    target_fps_scale: float = 1.0


class AdaptivePersonProcessingPolicy:
    def __init__(
        self,
        cpu_high_watermark: float = 85.0,
        cpu_recovery_watermark: float = 70.0,
        vram_high_watermark: float = 88.0,
        vram_recovery_watermark: float = 75.0,
        ram_high_watermark: float = 85.0,
        latency_p95_high_ms: float = 50.0,
    ) -> None:
        self.cpu_high_watermark = cpu_high_watermark
        self.cpu_recovery_watermark = cpu_recovery_watermark
        self.vram_high_watermark = vram_high_watermark
        self.vram_recovery_watermark = vram_recovery_watermark
        self.ram_high_watermark = ram_high_watermark
        self.latency_p95_high_ms = latency_p95_high_ms

        self.current_tier = AdaptivePersonLoadTier.FULL_QUALITY
        self._last_state_change = time.monotonic()
        self._logger = get_logger("adaptive_person_policy")

    def evaluate_policy(
        self,
        cpu_percent: float = 0.0,
        ram_percent: float = 0.0,
        vram_percent: float = 0.0,
        gpu_utilization: float = 0.0,
        p95_latency_ms: float = 0.0,
        queue_depth: int = 0,
        active_tracks_count: int = 0,
    ) -> SchedulerPolicyParameters:
        now = time.monotonic()
        prev_tier = self.current_tier


        severe_pressure = (
            cpu_percent >= self.cpu_high_watermark + 5.0
            or ram_percent >= self.ram_high_watermark + 5.0
            or vram_percent >= self.vram_high_watermark + 5.0
            or (queue_depth >= 15 and p95_latency_ms >= self.latency_p95_high_ms * 2.0)
        )
        high_pressure = (
            cpu_percent >= self.cpu_high_watermark
            or ram_percent >= self.ram_high_watermark
            or vram_percent >= self.vram_high_watermark
            or p95_latency_ms >= self.latency_p95_high_ms
            or queue_depth >= 8
        )
        moderate_pressure = (
            cpu_percent >= (self.cpu_recovery_watermark + 5.0)
            or active_tracks_count >= 50
            or queue_depth >= 4
        )
        healthy = (
            cpu_percent <= self.cpu_recovery_watermark
            and ram_percent <= (self.ram_high_watermark - 10.0)
            and vram_percent <= self.vram_recovery_watermark
            and queue_depth <= 2
        )

        if severe_pressure:
            self.current_tier = AdaptivePersonLoadTier.DEGRADED_MODE
        elif high_pressure:
            if queue_depth >= 10:
                self.current_tier = AdaptivePersonLoadTier.QUEUE_BACKPRESSURE
            else:
                self.current_tier = AdaptivePersonLoadTier.AGGRESSIVE_FRAME_SKIPPING
        elif moderate_pressure:
            if active_tracks_count >= 30:
                self.current_tier = AdaptivePersonLoadTier.MICRO_BATCHING
            else:
                self.current_tier = AdaptivePersonLoadTier.REDUCED_PROCESSING_FPS
        elif healthy and self.current_tier != AdaptivePersonLoadTier.FULL_QUALITY and (now - self._last_state_change) >= 1.5:

            self.current_tier = AdaptivePersonLoadTier.AUTOMATIC_RECOVERY
            if (now - self._last_state_change) >= 3.0:
                self.current_tier = AdaptivePersonLoadTier.FULL_QUALITY

        if self.current_tier != prev_tier:
            self._last_state_change = now
            self._logger.info(
                f"[ADAPTIVE_TIER_TRANSITION] Policy transitioned: '{prev_tier}' -> '{self.current_tier}' "
                f"(CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}%, VRAM: {vram_percent:.1f}%, "
                f"Queue: {queue_depth}, Active Tracks: {active_tracks_count})"
            )


        if self.current_tier == AdaptivePersonLoadTier.FULL_QUALITY:
            return SchedulerPolicyParameters(
                tier=AdaptivePersonLoadTier.FULL_QUALITY,
                reid_update_interval=6,
                max_batch_size=32,
                skip_confirmed_reid=False,
                gait_extraction_interval=10,
                drop_stale_frames=False,
                max_tracks_per_batch=32,
                target_fps_scale=1.0,
            )
        if self.current_tier == AdaptivePersonLoadTier.REDUCED_PROCESSING_FPS:
            return SchedulerPolicyParameters(
                tier=AdaptivePersonLoadTier.REDUCED_PROCESSING_FPS,
                reid_update_interval=12,
                max_batch_size=24,
                skip_confirmed_reid=False,
                gait_extraction_interval=12,
                drop_stale_frames=False,
                max_tracks_per_batch=24,
                target_fps_scale=0.85,
            )
        if self.current_tier == AdaptivePersonLoadTier.MICRO_BATCHING:
            return SchedulerPolicyParameters(
                tier=AdaptivePersonLoadTier.MICRO_BATCHING,
                reid_update_interval=16,
                max_batch_size=16,
                skip_confirmed_reid=True,
                gait_extraction_interval=15,
                drop_stale_frames=False,
                max_tracks_per_batch=16,
                target_fps_scale=0.75,
            )
        if self.current_tier == AdaptivePersonLoadTier.AGGRESSIVE_FRAME_SKIPPING:
            return SchedulerPolicyParameters(
                tier=AdaptivePersonLoadTier.AGGRESSIVE_FRAME_SKIPPING,
                reid_update_interval=24,
                max_batch_size=8,
                skip_confirmed_reid=True,
                gait_extraction_interval=20,
                drop_stale_frames=True,
                max_tracks_per_batch=12,
                target_fps_scale=0.50,
            )
        if self.current_tier == AdaptivePersonLoadTier.QUEUE_BACKPRESSURE:
            return SchedulerPolicyParameters(
                tier=AdaptivePersonLoadTier.QUEUE_BACKPRESSURE,
                reid_update_interval=32,
                max_batch_size=4,
                skip_confirmed_reid=True,
                gait_extraction_interval=25,
                drop_stale_frames=True,
                max_tracks_per_batch=8,
                target_fps_scale=0.35,
            )
        if self.current_tier == AdaptivePersonLoadTier.DEGRADED_MODE:
            return SchedulerPolicyParameters(
                tier=AdaptivePersonLoadTier.DEGRADED_MODE,
                reid_update_interval=48,
                max_batch_size=4,
                skip_confirmed_reid=True,
                gait_extraction_interval=30,
                drop_stale_frames=True,
                max_tracks_per_batch=4,
                target_fps_scale=0.25,
            )


        return SchedulerPolicyParameters(
            tier=AdaptivePersonLoadTier.AUTOMATIC_RECOVERY,
            reid_update_interval=10,
            max_batch_size=20,
            skip_confirmed_reid=False,
            gait_extraction_interval=12,
            drop_stale_frames=False,
            max_tracks_per_batch=20,
            target_fps_scale=0.90,
        )


@dataclass
class BatchCandidateItem:
    camera_id: str
    track_id: int
    crop: np.ndarray
    bbox: list[int]
    context: PersonTrackContext
    priority: float = 0.0


class PersonTrackScheduler:
    def __init__(self, adaptive_policy: AdaptivePersonProcessingPolicy | None = None) -> None:
        self.policy_engine = adaptive_policy or AdaptivePersonProcessingPolicy()
        self._lock = threading.Lock()
        self._logger = get_logger("person_scheduler")


        self._camera_credits: dict[str, float] = {}
        self._camera_quantum = 10.0


        self._total_batches_dispatched = 0
        self._total_persons_batched = 0
        self._total_skipped_confirmed = 0

    def select_reid_candidates(
        self,
        candidate_items: list[BatchCandidateItem],
        policy_params: SchedulerPolicyParameters,
        frame_index: int = 0,
    ) -> list[BatchCandidateItem]:
        if not candidate_items:
            return []

        eligible: list[BatchCandidateItem] = []

        with self._lock:
            for item in candidate_items:
                ctx = item.context


                if policy_params.skip_confirmed_reid and ctx.state == TrackLifecycleState.IDENTIFIED:
                    self._total_skipped_confirmed += 1
                    continue


                frames_since_update = frame_index - ctx.appearance_last_frame
                if ctx.appearance_embedding is not None and frames_since_update < policy_params.reid_update_interval:
                    continue





                unconfirmed_boost = 2.0 if ctx.fused_identity == "UNKNOWN_PERSON" else 1.0
                priority_score = (frames_since_update * unconfirmed_boost * max(0.5, ctx.track_confidence))


                cam_credit = self._camera_credits.get(item.camera_id, self._camera_quantum)
                item.priority = priority_score * (1.0 + cam_credit / 100.0)

                eligible.append(item)


            eligible.sort(key=lambda x: x.priority, reverse=True)


            batch = eligible[: policy_params.max_batch_size]


            for item in batch:
                self._camera_credits[item.camera_id] = max(
                    0.0, self._camera_credits.get(item.camera_id, self._camera_quantum) - 1.0
                )


            for cid in list(self._camera_credits.keys()):
                self._camera_credits[cid] = min(50.0, self._camera_credits[cid] + 0.5)

            self._total_batches_dispatched += 1
            self._total_persons_batched += len(batch)
            return batch

    def get_scheduler_telemetry(self) -> dict[str, Any]:
        with self._lock:
            return {
                "current_policy_tier": self.policy_engine.current_tier.value,
                "total_batches_dispatched": self._total_batches_dispatched,
                "total_persons_batched": self._total_persons_batched,
                "total_skipped_confirmed": self._total_skipped_confirmed,
                "camera_credits": dict(self._camera_credits),
            }
