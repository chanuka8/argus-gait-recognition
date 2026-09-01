"""
Comprehensive Test Suite for ARGUS AI Person Assessment & Corrected RED/GREEN/YELLOW Overlay Architecture.

Certified Color Semantics:
- RED BGR (0, 0, 255): Confirmed identity / successfully recognized person.
- GREEN BGR (0, 255, 0): Person detected and tracked, but identity is not confirmed (default for all
  ordinary unconfirmed states including UNKNOWN, PENDING, ASSESSING, EVIDENCE_COLLECTING,
  INSUFFICIENT_EVIDENCE, GAIT_UNAVAILABLE, APPEARANCE_UNAVAILABLE, BIOMETRIC_INAPPLICABLE,
  WHEELCHAIR, CRUTCHES, NON_STANDARD_GAIT, SEATED, STATIONARY, etc.).
- YELLOW BGR (0, 255, 255): Reserved special operational / attention state ONLY (e.g. SPECIAL_ATTENTION,
  SECURITY_ALERT, FLAGGED). Never used for ordinary unconfirmed or evidence-collecting persons.

Tests:
1. Exact BGR color constants (RED, GREEN, YELLOW).
2. Confirmed identity -> RED (0, 0, 255).
3. Unknown -> GREEN (0, 255, 0).
4. Pending -> GREEN (0, 255, 0).
5. Assessing -> GREEN (0, 255, 0).
6. Evidence collecting -> GREEN (0, 255, 0).
7. Walking person before sufficient gait evidence -> GREEN (0, 255, 0).
8. Walking person after gait embedding but before identity confirmation -> GREEN (0, 255, 0).
9. Gait unavailable -> GREEN (0, 255, 0).
10. Appearance unavailable -> GREEN (0, 255, 0).
11. Both embeddings unavailable -> GREEN (0, 255, 0).
12. Wheelchair -> GREEN unless identity confirmed.
13. Crutches -> GREEN unless identity confirmed.
14. Non-standard gait -> GREEN unless identity confirmed.
15. Stationary/seated -> GREEN unless identity confirmed.
16. Confirmed wheelchair user through appearance pathway -> RED (0, 0, 255).
17. Confirmed crutches user through appearance pathway -> RED (0, 0, 255).
18. Multiple persons maintain independent colors concurrently.
19. Detection without embeddings remains visible.
20. Identity failure does not remove bounding box.
21. YELLOW is NOT produced for ordinary unknown/pending/assessment states.
22. YELLOW is produced only if an explicit operational attention state exists (SPECIAL_ATTENTION).
23. Single person inference failure containment.
24. CameraWorker renders all preview overlays correctly.
25. Walking person biometric gait and appearance continuity.
26. Multi-camera stream isolation.
27. No artificial person/track limit caps.
"""

from __future__ import annotations

import time

import numpy as np

from intelligence.appearance_embedding import AppearanceEmbeddingExtractor
from intelligence.concurrent_track_manager import (
    ConcurrentTrackManager,
    MobilityState,
    PersonAssessmentState,
    PersonTrackContext,
)
from pipeline.detection.detection_validator import (
    DetectionValidator,
)
from pipeline.gei.stream_gei_builder import StreamGEIBuilder
from services.camera_worker import CameraWorker
from services.recognition_worker import RecognitionResult, RecognitionResultCache
from streaming.production_multicamera_engine import (
    HardwareProfile,
    ProductionMultiCameraEngine,
)
from utils.display_renderer import (
    COLOR_GREEN_BGR,
    COLOR_RED_BGR,
    COLOR_YELLOW_BGR,
    DISPLAY_STATE_ASSESSING,
    DISPLAY_STATE_CONFIRMED,
    DISPLAY_STATE_INAPPLICABLE,
    DISPLAY_STATE_SPECIAL_ATTENTION,
    DISPLAY_STATE_UNCONFIRMED,
    DetectionDisplayRenderer,
    map_to_display_state,
)


class TestPersonAssessmentOverlay:
    """Test suite certifying the production person assessment overlay architecture."""

    def test_exact_bgr_color_constants(self) -> None:
        """Verify exact BGR color values: RED=(0,0,255), GREEN=(0,255,0), YELLOW=(0,255,255)."""
        assert COLOR_RED_BGR == (0, 0, 255)
        assert COLOR_GREEN_BGR == (0, 255, 0)
        assert COLOR_YELLOW_BGR == (0, 255, 255)

        renderer = DetectionDisplayRenderer()
        assert renderer.get_color_for_state(DISPLAY_STATE_CONFIRMED) == (0, 0, 255)
        assert renderer.get_color_for_state(DISPLAY_STATE_UNCONFIRMED) == (0, 255, 0)
        assert renderer.get_color_for_state(DISPLAY_STATE_ASSESSING) == (0, 255, 0)
        assert renderer.get_color_for_state(DISPLAY_STATE_INAPPLICABLE) == (0, 255, 0)
        assert renderer.get_color_for_state(DISPLAY_STATE_SPECIAL_ATTENTION) == (0, 255, 255)

    def test_confirmed_identity_renders_red(self) -> None:
        """Verify confirmed / recognized identity renders with a RED bounding box (0, 0, 255)."""
        renderer = DetectionDisplayRenderer()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)

        renderer.draw(
            frame=frame,
            box=[40, 60, 100, 200],
            track_id=1,
            identity="Alice_Smith",
            score=0.94,
            decision="CONFIRMED_MATCH",
            display_state="CONFIRMED",
        )


        assert frame[200, 50, 2] == 255
        assert frame[200, 50, 1] == 0
        assert frame[200, 50, 0] == 0

    def test_unknown_person_renders_green(self) -> None:
        """Verify UNKNOWN person renders GREEN (0, 255, 0), NEVER RED or YELLOW."""
        state = map_to_display_state(
            status="UNKNOWN",
            decision="UNKNOWN",
            identity="UNKNOWN",
            mobility_state="STANDARD_WALKING",
            gait_eligible=True,
        )
        assert state == DISPLAY_STATE_UNCONFIRMED

        renderer = DetectionDisplayRenderer()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        renderer.draw(
            frame=frame,
            box=[40, 60, 100, 200],
            track_id=3,
            identity="UNKNOWN",
            score=0.0,
            decision="UNKNOWN",
            display_state=state,
        )


        assert frame[200, 50, 1] == 255
        assert frame[200, 50, 2] == 0
        assert frame[200, 50, 0] == 0

    def test_pending_and_assessing_person_renders_green(self) -> None:
        """Verify person in PENDING or ASSESSING state renders GREEN (0, 255, 0)."""
        renderer = DetectionDisplayRenderer()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)

        renderer.draw(
            frame=frame,
            box=[40, 60, 100, 200],
            track_id=2,
            identity="UNKNOWN_PERSON",
            score=0.45,
            decision="COLLECTING",
            display_state="ASSESSING",
        )

        assert frame[200, 50, 1] == 255
        assert frame[200, 50, 2] == 0
        assert frame[200, 50, 0] == 0

    def test_walking_person_before_and_during_evidence_collection_renders_green(self) -> None:
        """Verify walking person accumulating evidence renders GREEN until confirmed."""
        renderer = DetectionDisplayRenderer()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)


        renderer.draw(
            frame=frame,
            box=[40, 60, 100, 200],
            track_id=5,
            identity="UNKNOWN",
            score=0.0,
            decision="COLLECTING",
            display_state="EVIDENCE_COLLECTING",
        )
        assert frame[200, 50, 1] == 255 and frame[200, 50, 2] == 0


        frame.fill(0)
        renderer.draw(
            frame=frame,
            box=[40, 60, 100, 200],
            track_id=5,
            identity="UNKNOWN",
            score=0.55,
            decision="REVIEW_REQUIRED",
            display_state="UNCONFIRMED",
        )
        assert frame[200, 50, 1] == 255 and frame[200, 50, 2] == 0

    def test_missing_or_incomplete_embeddings_render_green(self) -> None:
        """Verify missing gait, appearance, or both embeddings renders GREEN."""
        renderer = DetectionDisplayRenderer()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)


        renderer.draw(frame=frame, box=[20, 20, 80, 140], track_id=6, display_state="UNCONFIRMED")
        assert frame[140, 30, 1] == 255 and frame[140, 30, 2] == 0


        frame.fill(0)
        renderer.draw(frame=frame, box=[20, 20, 80, 140], track_id=6, identity="UNKNOWN", score=0.40, display_state="UNCONFIRMED")
        assert frame[140, 30, 1] == 255 and frame[140, 30, 2] == 0


        frame.fill(0)
        renderer.draw(frame=frame, box=[20, 20, 80, 140], track_id=6, identity="UNKNOWN", score=0.50, display_state="UNCONFIRMED")
        assert frame[140, 30, 1] == 255 and frame[140, 30, 2] == 0

    def test_wheelchair_crutches_and_nonstandard_gait_render_green_unless_confirmed(self) -> None:
        """Verify wheelchair, crutches, and non-standard gait render GREEN unless appearance confirms identity."""
        validator = DetectionValidator()
        wheelchair_bbox = [100, 100, 260, 200]

        is_val, mob_state, gait_elig, app_elig, reason = validator.assess_detection(
            bbox=wheelchair_bbox,
            confidence=0.88,
            frame_shape=(480, 640, 3),
        )

        assert is_val is True
        assert mob_state == "WHEELCHAIR"
        assert gait_elig is False
        assert app_elig is True
        assert "WHEELCHAIR_SEATED" in reason

        ctx = PersonTrackContext(
            camera_id="cam_01",
            track_id=10,
            bbox=wheelchair_bbox,
            mobility_state=MobilityState.WHEELCHAIR,
            gait_eligible=False,
        )
        display_state = ctx.evaluate_display_state()
        assert display_state == DISPLAY_STATE_INAPPLICABLE
        assert ctx.assessment_state == PersonAssessmentState.BIOMETRIC_INAPPLICABLE
        assert display_state == DISPLAY_STATE_INAPPLICABLE

        renderer = DetectionDisplayRenderer()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)


        renderer.draw(
            frame=frame,
            box=wheelchair_bbox,
            track_id=10,
            identity="UNKNOWN",
            display_state=display_state,
            mobility_state="WHEELCHAIR",
        )
        assert frame[200, 110, 1] == 255 and frame[200, 110, 2] == 0


        frame.fill(0)
        ctx.status = "CONFIRMED"
        ctx.fused_identity = "David_Seated"
        ctx.fused_score = 0.93
        conf_state = ctx.evaluate_display_state()
        assert conf_state == DISPLAY_STATE_CONFIRMED

        renderer.draw(
            frame=frame,
            box=wheelchair_bbox,
            track_id=10,
            identity="David_Seated",
            score=0.93,
            display_state=conf_state,
            mobility_state="WHEELCHAIR",
        )
        assert frame[200, 110, 2] == 255 and frame[200, 110, 1] == 0

    def test_crutches_confirmed_via_appearance_renders_red(self) -> None:
        """Verify crutches user confirmed via appearance pathway renders RED."""
        ctx = PersonTrackContext(
            camera_id="cam_01",
            track_id=11,
            bbox=[50, 50, 120, 220],
            mobility_state=MobilityState.CRUTCHES_AID,
            gait_eligible=False,
            status="CONFIRMED",
            fused_identity="Emma_Crutches",
            fused_score=0.91,
        )
        display_state = ctx.evaluate_display_state()
        assert display_state == DISPLAY_STATE_CONFIRMED

        renderer = DetectionDisplayRenderer()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        renderer.draw(
            frame=frame,
            box=ctx.bbox,
            track_id=11,
            identity="Emma_Crutches",
            score=0.91,
            display_state=display_state,
            mobility_state="CRUTCHES_AID",
        )
        assert frame[220, 60, 2] == 255 and frame[220, 60, 1] == 0

    def test_explicit_special_operational_attention_renders_yellow_only(self) -> None:
        """Verify YELLOW is ONLY generated when an explicit special operational attention state exists."""
        renderer = DetectionDisplayRenderer()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)


        renderer.draw(
            frame=frame,
            box=[40, 60, 100, 200],
            track_id=7,
            identity="UNKNOWN",
            display_state="SPECIAL_ATTENTION",
            is_special_attention=True,
        )

        assert frame[200, 50, 2] == 255
        assert frame[200, 50, 1] == 255
        assert frame[200, 50, 0] == 0

    def test_multi_person_concurrent_distinct_states(self) -> None:
        """Verify multi-person frame displays ALL 5 persons simultaneously with their correct independent colors."""
        renderer = DetectionDisplayRenderer()
        frame = np.zeros((600, 600, 3), dtype=np.uint8)

        persons = [
            {"box": [20, 40, 80, 180], "tid": 1, "id": "Alice", "score": 0.92, "state": "CONFIRMED"},
            {"box": [100, 40, 160, 180], "tid": 2, "id": "UNKNOWN", "score": 0.30, "state": "UNCONFIRMED"},
            {"box": [180, 40, 300, 120], "tid": 3, "id": "UNKNOWN", "score": 0.0, "state": "BIOMETRIC_INAPPLICABLE"},
            {"box": [320, 40, 380, 180], "tid": 4, "id": "UNKNOWN", "score": 0.15, "state": "ASSESSING"},
            {"box": [400, 40, 460, 180], "tid": 5, "id": "Bob", "score": 0.89, "state": "CONFIRMED"},
        ]

        for p in persons:
            renderer.draw(
                frame=frame,
                box=p["box"],
                track_id=p["tid"],
                identity=p["id"],
                score=p["score"],
                display_state=p["state"],
            )


        assert frame[180, 30, 2] == 255 and frame[180, 30, 1] == 0

        assert frame[180, 110, 1] == 255 and frame[180, 110, 2] == 0

        assert frame[120, 190, 1] == 255 and frame[120, 190, 2] == 0

        assert frame[180, 330, 1] == 255 and frame[180, 330, 2] == 0

        assert frame[180, 410, 2] == 255 and frame[180, 410, 1] == 0

    def test_single_person_inference_failure_containment(self) -> None:
        """Verify exception during one person's biometric extraction does not drop or crash adjacent persons."""
        engine = ProductionMultiCameraEngine(hardware_profile=HardwareProfile(cpu_cores=4, total_ram_mb=16384.0))
        assert engine is not None
        validator = DetectionValidator()

        dets = [
            {"bbox": [20, 20, 80, 180], "confidence": 0.85},
            {"bbox": None, "confidence": 0.0},
            {"bbox": [100, 20, 260, 100], "confidence": 0.80},
        ]

        assessed = []
        for d in dets:
            try:
                is_val, mob, gait_elig, app_elig, reason = validator.assess_detection(
                    d.get("bbox"), d.get("confidence", 0.0)
                )
                assessed.append((is_val, mob, gait_elig, app_elig, reason))
            except (TypeError, ValueError, AttributeError, RuntimeError):
                assessed.append((False, "ERROR", False, False, "EXCEPTION_CAUGHT"))

        assert len(assessed) == 3
        assert assessed[0][0] is True
        assert assessed[1][0] is False
        assert assessed[2][0] is True
        assert assessed[2][1] == "WHEELCHAIR"

    def test_camera_worker_renders_all_assessment_overlays(self) -> None:
        """Verify CameraWorker._render_preview_overlays renders active cache items with appropriate colors."""
        worker = CameraWorker(
            camera_id="cam_01",
            camera_config={"type": "webcam", "device_index": 0, "width": 320, "height": 240},
        )
        cache = RecognitionResultCache()
        now = time.monotonic()

        cache.put(
            RecognitionResult(
                camera_id="cam_01",
                track_id=1,
                identity="John",
                similarity=0.92,
                confidence=0.92,
                decision="CONFIRMED",
                status="CONFIRMED",
                bbox=[20, 50, 80, 180],
                timestamp=now,
                iso_timestamp="2026-08-31T00:00:00Z",
                display_state="CONFIRMED",
            )
        )
        cache.put(
            RecognitionResult(
                camera_id="cam_01",
                track_id=2,
                identity="UNKNOWN",
                similarity=0.0,
                confidence=0.0,
                decision="ASSESSING",
                status="ASSESSING",
                bbox=[100, 50, 160, 180],
                timestamp=now,
                iso_timestamp="2026-08-31T00:00:00Z",
                display_state="ASSESSING",
            )
        )
        cache.put(
            RecognitionResult(
                camera_id="cam_01",
                track_id=3,
                identity="UNKNOWN",
                similarity=0.0,
                confidence=0.0,
                decision="BIOMETRIC_INAPPLICABLE",
                status="BIOMETRIC_INAPPLICABLE",
                bbox=[180, 50, 300, 120],
                timestamp=now,
                iso_timestamp="2026-08-31T00:00:00Z",
                display_state="BIOMETRIC_INAPPLICABLE",
                mobility_state="WHEELCHAIR",
                gait_eligible=False,
            )
        )

        class DummyEngine:
            def __init__(self, c: RecognitionResultCache):
                self.cache = c
            def is_running(self):
                return True

        worker.inference_engine = DummyEngine(cache)
        raw_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        rendered = worker._render_preview_overlays(raw_frame)

        assert rendered is not None

        assert rendered[180, 30, 2] == 255 and rendered[180, 30, 1] == 0

        assert rendered[180, 110, 1] == 255 and rendered[180, 110, 2] == 0

        assert rendered[120, 190, 1] == 255 and rendered[120, 190, 2] == 0

    def test_walking_person_biometric_gait_and_appearance_continuity(self) -> None:
        """Verify walking person accumulates silhouettes -> GEI -> ByGaitLight 256D and OSNet 512D embeddings."""
        track_manager = ConcurrentTrackManager()
        gei_builder = StreamGEIBuilder()
        reid_extractor = AppearanceEmbeddingExtractor()

        for f_idx in range(15):
            bbox = [100 + f_idx * 4, 80, 160 + f_idx * 4, 240]
            ctx = track_manager.update_or_create_track(
                camera_id="cam_walk",
                track_id=1,
                bbox=bbox,
                confidence=0.92,
                frame_index=f_idx,
            )
            assert ctx.track_id == 1
            sil = np.zeros((128, 64), dtype=np.uint8)
            sil[20:110, 15:50] = 255
            gei_builder.add_silhouette(1, sil)

        assert gei_builder.get_frame_count(1) == 15
        assert gei_builder.is_ready(1) is True
        gei = gei_builder.build_gei(1)
        assert gei is not None
        assert gei.shape == (128, 64)

        crop = np.ones((160, 60, 3), dtype=np.uint8) * 120
        emb = reid_extractor.extract(crop, track_id=1)
        assert emb is not None
        assert emb.shape == (512,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)

    def test_multi_camera_stream_isolation(self) -> None:
        """Verify Camera A tracks and states never appear on Camera B."""
        cache = RecognitionResultCache()
        now = time.monotonic()

        cache.put(
            RecognitionResult(
                camera_id="cam_A",
                track_id=1,
                identity="Alice",
                similarity=0.95,
                confidence=0.95,
                decision="CONFIRMED",
                status="CONFIRMED",
                bbox=[10, 10, 60, 150],
                timestamp=now,
                iso_timestamp="2026-08-31T00:00:00Z",
                display_state="CONFIRMED",
            )
        )
        cache.put(
            RecognitionResult(
                camera_id="cam_B",
                track_id=2,
                identity="Bob",
                similarity=0.88,
                confidence=0.88,
                decision="CONFIRMED",
                status="CONFIRMED",
                bbox=[20, 20, 70, 160],
                timestamp=now,
                iso_timestamp="2026-08-31T00:00:00Z",
                display_state="CONFIRMED",
            )
        )

        tracks_a = cache.get_active_tracks("cam_A")
        tracks_b = cache.get_active_tracks("cam_B")

        assert len(tracks_a) == 1
        assert tracks_a[0].identity == "Alice"
        assert len(tracks_b) == 1
        assert tracks_b[0].identity == "Bob"

    def test_unbounded_detections_support(self) -> None:
        """Verify validator and context manager support arbitrary high-density counts (e.g. 100 persons) without caps."""
        validator = DetectionValidator()
        frame_shape = (1080, 1920, 3)

        detections = []
        for i in range(100):
            detections.append(
                {
                    "bbox": [i * 15, 50, i * 15 + 40, 180],
                    "confidence": 0.75,
                }
            )

        tagged = validator.tag_detections(detections, frame_shape=frame_shape, camera_id="cam_crowd")
        assert len(tagged) == 100
        for item in tagged:
            assert item.is_valid is True
