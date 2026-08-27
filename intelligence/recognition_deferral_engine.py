"""
Stage 2: Recognition Deferral and Evidence Accumulation Engine.

Manages deterministic recognition deferral, evidence accumulation buffers,
TTL expiration, and watchlist suppression for deferred/uncertain tracks.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RecognitionState(str, Enum):
    CONFIRMED = "CONFIRMED"
    DEFERRED_INSUFFICIENT_EVIDENCE = "DEFERRED_INSUFFICIENT_EVIDENCE"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN"


@dataclass
class EvidenceRecord:
    identity_candidate: str
    similarity: float
    quality: float
    open_set_state: str
    temporal_decision: str
    reliability: float
    occlusion: float
    timestamp: float


@dataclass
class DeferralResult:
    recognition_state: RecognitionState
    recognition_deferred: bool
    identity: str
    confidence: float
    defer_reason: str
    accumulated_evidence_count: int
    should_alert: bool = False
    evidence_history: list[EvidenceRecord] = field(default_factory=list)


class RecognitionDeferralEngine:
    """
    Evaluates recognition candidates and defers identity decisions until
    accumulated evidence meets confidence, quality, reliability, and clean-frame criteria.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.evidence_window = int(cfg.get("evidence_window", 5))
        self.minimum_confirmations = int(cfg.get("minimum_confirmations", 3))
        self.minimum_reliability = float(cfg.get("minimum_reliability", 0.70))
        self.evidence_ttl_seconds = float(cfg.get("evidence_ttl_seconds", 10.0))
        self.identity_ttl_seconds = float(cfg.get("identity_ttl_seconds", 5.0))

        self.evidence_buffers: dict[tuple[str, Any], list[EvidenceRecord]] = {}
        self.retained_identities: dict[tuple[str, Any], tuple[str, float]] = {}

    def is_enabled(self) -> bool:
        return self.enabled

    def evaluate_and_accumulate(
        self,
        camera_id: str,
        track_id: Any,
        identity_candidate: str,
        similarity: float,
        quality: float,
        open_set_state: str,
        temporal_decision: str,
        reliability: float,
        occlusion: float,
        clean_frame_count: int = 20,
        clean_frame_ratio: float = 1.0,
        minimum_clean_frames: int = 18,
        minimum_clean_ratio: float = 0.70,
        quality_threshold: float = 0.60,
        moderate_occlusion_threshold: float = 0.35,
        timestamp: float | None = None,
    ) -> DeferralResult:
        """
        Evaluate candidate match, accumulate evidence into bounded buffer, and determine recognition state.
        """
        now = timestamp if timestamp is not None else time.monotonic()
        key = (camera_id, track_id)

        if not self.enabled:
            is_confirmed = open_set_state == "KNOWN" and similarity >= 0.85 and identity_candidate != "UNKNOWN"
            state = (
                RecognitionState.CONFIRMED
                if is_confirmed
                else (RecognitionState.UNCERTAIN if open_set_state == "UNCERTAIN" else RecognitionState.UNKNOWN)
            )
            return DeferralResult(
                recognition_state=state,
                recognition_deferred=False,
                identity=identity_candidate if is_confirmed else "UNKNOWN",
                confidence=similarity if is_confirmed else 0.0,
                defer_reason="",
                accumulated_evidence_count=1,
                should_alert=is_confirmed,
            )

        new_record = EvidenceRecord(
            identity_candidate=identity_candidate,
            similarity=similarity,
            quality=quality,
            open_set_state=open_set_state,
            temporal_decision=temporal_decision,
            reliability=reliability,
            occlusion=occlusion,
            timestamp=now,
        )

        if key not in self.evidence_buffers:
            self.evidence_buffers[key] = []

        buf = self.evidence_buffers[key]
        buf.append(new_record)

        buf[:] = [r for r in buf if (now - r.timestamp) <= self.evidence_ttl_seconds]
        if len(buf) > self.evidence_window * 2:
            buf[:] = buf[-self.evidence_window * 2 :]

        accumulated_count = len(buf)

        defer_reasons = []

        if identity_candidate == "UNKNOWN" or open_set_state == "UNKNOWN":
            return DeferralResult(
                recognition_state=RecognitionState.UNKNOWN,
                recognition_deferred=False,
                identity="UNKNOWN",
                confidence=0.0,
                defer_reason="Identity labeled UNKNOWN",
                accumulated_evidence_count=accumulated_count,
                should_alert=False,
                evidence_history=buf,
            )

        if occlusion > moderate_occlusion_threshold:
            defer_reasons.append(f"Occlusion ({occlusion:.2f}) > threshold ({moderate_occlusion_threshold})")

        if clean_frame_count < minimum_clean_frames:
            defer_reasons.append(f"Clean frame count ({clean_frame_count}) < min ({minimum_clean_frames})")

        if clean_frame_ratio < minimum_clean_ratio:
            defer_reasons.append(f"Clean frame ratio ({clean_frame_ratio:.2f}) < min ({minimum_clean_ratio:.2f})")

        if quality < quality_threshold:
            defer_reasons.append(f"Quality ({quality:.2f}) < threshold ({quality_threshold:.2f})")

        if open_set_state == "UNCERTAIN":
            defer_reasons.append("Open-set state is UNCERTAIN")

        if temporal_decision not in ("MAJORITY_VOTE", "CONFIRMED"):
            defer_reasons.append(f"Temporal decision '{temporal_decision}' not confirmed")

        if reliability < self.minimum_reliability:
            defer_reasons.append(f"Reliability ({reliability:.2f}) < min ({self.minimum_reliability:.2f})")

        matching_confirmations = [
            r
            for r in buf
            if r.identity_candidate == identity_candidate
            and r.reliability >= self.minimum_reliability
            and r.open_set_state == "KNOWN"
        ]

        if len(matching_confirmations) < self.minimum_confirmations:
            defer_reasons.append(
                f"Confirmations count ({len(matching_confirmations)}) < required ({self.minimum_confirmations})"
            )

        if defer_reasons:
            retained_id = "UNKNOWN"
            if key in self.retained_identities:
                prev_id, prev_ts = self.retained_identities[key]
                if (now - prev_ts) <= self.identity_ttl_seconds and prev_id == identity_candidate and quality >= 0.50:
                    retained_id = prev_id

            return DeferralResult(
                recognition_state=RecognitionState.DEFERRED_INSUFFICIENT_EVIDENCE,
                recognition_deferred=True,
                identity=retained_id if retained_id != "UNKNOWN" else identity_candidate,
                confidence=similarity,
                defer_reason="; ".join(defer_reasons),
                accumulated_evidence_count=accumulated_count,
                should_alert=False,
                evidence_history=buf,
            )

        self.retained_identities[key] = (identity_candidate, now)

        return DeferralResult(
            recognition_state=RecognitionState.CONFIRMED,
            recognition_deferred=False,
            identity=identity_candidate,
            confidence=similarity,
            defer_reason="",
            accumulated_evidence_count=accumulated_count,
            should_alert=True,
            evidence_history=buf,
        )

    def cleanup_inactive(self, max_idle_seconds: float = 15.0, current_time: float | None = None) -> None:
        """Clean expired evidence buffers and retained identities."""
        now = current_time if current_time is not None else time.monotonic()

        for key, buf in list(self.evidence_buffers.items()):
            self.evidence_buffers[key] = [r for r in buf if (now - r.timestamp) <= max_idle_seconds]
            if not self.evidence_buffers[key]:
                del self.evidence_buffers[key]

        for key, (pid, ts) in list(self.retained_identities.items()):
            if (now - ts) > max_idle_seconds:
                del self.retained_identities[key]
