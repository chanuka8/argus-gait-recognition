"""Regression tests for the shared live-inference concurrency gate.

Background: identify_image, analyze_video, and the synchronous enroll
fallback used to call synchronous, CPU-bound GaitService methods directly
inside async FastAPI route handlers - with no asyncio.to_thread offload and
no concurrency bound. That blocked the single event loop for the full
duration of every inference call (health/auth froze during inference), and
GaitService.enroll_images mutates shared, unlocked instance state
(self.gallery_features/self.gallery_labels) before saving - safe only
because direct synchronous calls happened to serialize on the event loop.

The fix wraps these calls in `asyncio.to_thread` under one shared
`asyncio.Semaphore(1)` (app.core.inference_gate) - freeing the event loop
for unrelated requests while still guaranteeing at most one live inference
call in flight at a time, matching the previous de-facto behavior without
exposing the shared-state race that true unbounded parallelism would.

These tests use a lightweight fake GaitService (no real model loading) so
they stay fast and safe to run under low memory headroom.
"""

import io
import threading
import time

from fastapi.testclient import TestClient

from app.api.server import app
from app.core.inference_gate import MAX_CONCURRENT_LIVE_INFERENCE, get_inference_gate
from app.security_layer.auth import get_session_store


class _ConcurrencyTrackingFakeService:
    """Stands in for GaitService.process_image_bytes: sleeps briefly and
    records the peak number of concurrent invocations, without touching any
    real model."""

    def __init__(self, sleep_seconds: float = 0.3):
        self.sleep_seconds = sleep_seconds
        self._active = 0
        self._peak_active = 0
        self._lock = threading.Lock()
        self.call_count = 0

    def process_image_bytes(self, image_bytes: bytes, camera_id: str = "upload-image") -> dict:
        with self._lock:
            self._active += 1
            self._peak_active = max(self._peak_active, self._active)
            self.call_count += 1
        try:
            time.sleep(self.sleep_seconds)
            return {"event_id": "fake", "camera_id": camera_id, "status": "processed"}
        finally:
            with self._lock:
                self._active -= 1

    @property
    def peak_active(self) -> int:
        with self._lock:
            return self._peak_active


def setup_function():
    get_session_store().clear()


def _auth_headers():
    session = get_session_store().create_session(
        operator_id="gate_test_op",
        username="gate_test_op",
        role="investigator",
    )
    return {"Authorization": f"Bearer {session.token}"}


def test_gate_default_limit_is_one():
    assert MAX_CONCURRENT_LIVE_INFERENCE == 1
    assert get_inference_gate()._value == 1


def test_identify_image_concurrent_requests_never_overlap():
    """Two real, concurrent HTTP requests to /identify/image must never
    execute the underlying inference call at the same time."""
    fake_service = _ConcurrencyTrackingFakeService(sleep_seconds=0.3)
    headers = _auth_headers()

    try:
        with TestClient(app) as client:
            # Startup already ran and assigned a real GaitService to
            # app.state.gait_service - override it now, after startup.
            app.state.gait_service = fake_service

            def do_request():
                client.post(
                    "/api/v1/identify/image",
                    files={"file": ("test.jpg", io.BytesIO(b"\xff\xd8\xff\xe0fake"), "image/jpeg")},
                    data={"camera_id": "cam1"},
                    headers=headers,
                )

            threads = [threading.Thread(target=do_request) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert fake_service.call_count == 3
        assert fake_service.peak_active == 1, (
            f"Expected inference calls to be serialized (peak_active=1), "
            f"got peak_active={fake_service.peak_active} - the gate did not serialize concurrent requests"
        )
    finally:
        app.state.gait_service = None


def test_health_endpoint_responsive_during_inference():
    """The event loop must not be blocked by an in-flight inference call:
    a concurrent /health request must complete quickly, not wait for
    inference to finish."""
    fake_service = _ConcurrencyTrackingFakeService(sleep_seconds=1.0)
    headers = _auth_headers()

    try:
        with TestClient(app) as client:
            app.state.gait_service = fake_service
            health_latencies = []

            def do_inference():
                client.post(
                    "/api/v1/identify/image",
                    files={"file": ("test.jpg", io.BytesIO(b"\xff\xd8\xff\xe0fake"), "image/jpeg")},
                    data={"camera_id": "cam1"},
                    headers=headers,
                )

            inference_thread = threading.Thread(target=do_inference)
            inference_thread.start()
            time.sleep(0.2)  # let inference start and take the gate

            t0 = time.perf_counter()
            resp = client.get("/health")
            elapsed = time.perf_counter() - t0
            health_latencies.append(elapsed)

            inference_thread.join(timeout=10)

        assert resp.status_code == 200
        assert health_latencies[0] < 0.5, (
            f"/health took {health_latencies[0]:.2f}s while inference was running "
            f"(sleep_seconds=1.0) - the event loop appears to still be blocked"
        )
    finally:
        app.state.gait_service = None
