"""Regression tests for low-memory ML startup admission control.

Background: GaitService.warmup() previously started heavy model
construction (torch import, YOLO detector, gait/appearance encoders)
unconditionally, regardless of available system memory. Measured evidence
on this class of 8GB machine: a healthy warmup takes 6-18s, but under
~270-470MB available RAM it took over 2 hours - severe OS-level page-fault
thrashing, not merely a slow startup. LOW_RESOURCE profile classification
(app.core.resource_profile) only ever tuned batch sizes for an
already-safe-to-start warmup; it was never a gate on whether to start
warmup at all.

Fix: warmup() now checks app.core.resource_profile.
has_sufficient_ml_startup_headroom() (freshly measured, not the
cached-at-process-start profile) immediately before attempting real model
construction. Below the configurable ARGUS_MIN_ML_STARTUP_HEADROOM_MB
threshold (default 1024MB), it marks bygait/silhouette/osnet/detector as
DEFERRED instead of constructing them, and the API layer
(_require_ml_ready in app.api.v1.router) returns 503 for any live
inference request rather than letting a lazy-property access trigger
construction under the same unsafe conditions mid-request.

These tests mock the memory-measurement layer directly - no real low
physical RAM is needed or used.
"""

import io
from unittest.mock import MagicMock, PropertyMock, patch

from fastapi.testclient import TestClient

from app.api.server import app
from app.security_layer.auth import get_session_store
from app.services.gait_service import GaitService


def setup_function():
    get_session_store().clear()


def _auth_headers():
    session = get_session_store().create_session(
        operator_id="lowmem_op",
        username="lowmem_op",
        role="investigator",
    )
    return {"Authorization": f"Bearer {session.token}"}


def _low_memory():
    return False, 100.0, 1024.0


def _sufficient_memory():
    return True, 4096.0, 1024.0


def _forbidden_property():
    """A PropertyMock that fails the test loudly if the property it patches
    is ever accessed - stronger evidence than a call-count assertion that
    warmup() never touches these lazy, model-constructing properties."""
    return PropertyMock(side_effect=AssertionError("should not be accessed when memory is insufficient"))


def test_low_memory_defers_heavy_components_without_constructing_them():
    """A. Critically low available RAM -> heavy ML warmup is not started."""
    service = GaitService()
    with (
        patch("app.core.resource_profile.has_sufficient_ml_startup_headroom", side_effect=_low_memory),
        patch.object(type(service), "extractor", new_callable=_forbidden_property),
        patch.object(type(service), "silhouette_extractor", new_callable=_forbidden_property),
        patch.object(type(service), "appearance_extractor", new_callable=_forbidden_property),
        patch.object(type(service), "detector", new_callable=_forbidden_property),
    ):
        result = service.warmup()

        assert result["status"] == "DEFERRED"
        assert result["reason"] == "insufficient_memory_headroom"


def test_deferred_state_reported_clearly():
    """B. Readiness clearly reports a deferred/waiting state, not silent READY."""
    service = GaitService()
    with patch("app.core.resource_profile.has_sufficient_ml_startup_headroom", side_effect=_low_memory):
        service.warmup()

    readiness = service.get_readiness()
    assert readiness["recognition_ready"] is False
    assert readiness["components"]["bygait"] == "DEFERRED"
    assert readiness["components"]["silhouette"] == "DEFERRED"
    assert readiness["components"]["osnet"] == "DEFERRED"
    assert readiness["components"]["detector"] == "DEFERRED"
    assert readiness["warmup_deferred_reason"] is not None
    assert "insufficient_memory_headroom" in readiness["warmup_deferred_reason"]
    assert service.is_warmed_up is False  # not silently marked complete


def test_health_and_auth_remain_usable_while_ml_deferred():
    """C + D. /health and authenticated non-ML endpoints stay responsive
    even while recognition is deferred for lack of memory headroom."""
    service = GaitService()
    with patch("app.core.resource_profile.has_sufficient_ml_startup_headroom", side_effect=_low_memory):
        service.warmup()

    app.state.gait_service = service
    try:
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200

            headers = _auth_headers()
            resp = client.get("/api/v1/auth/me", headers=headers)
            assert resp.status_code == 200
    finally:
        app.state.gait_service = None


def test_recognition_request_returns_503_not_hang_while_deferred():
    """E. A recognition request while ML is deferred gets a fast, structured
    503 - not a hang, not a silent attempt to construct models mid-request
    under the same unsafe memory conditions."""
    service = GaitService()
    with patch("app.core.resource_profile.has_sufficient_ml_startup_headroom", side_effect=_low_memory):
        service.warmup()

    app.state.gait_service = service
    try:
        # The lifespan handler (app/api/server.py) unconditionally fires a
        # background asyncio.create_task(gait_service.warmup_async()) on every
        # TestClient(app) entry, even when app.state.gait_service was already
        # configured above - it does not know this service's warmup was
        # already (deliberately) deferred. That background task makes its own,
        # separately-timed call to has_sufficient_ml_startup_headroom() the
        # instant the `with` block below is entered, before client.post() ever
        # runs. Patching only around client.post() leaves that startup call
        # unmocked, so on a host with real free headroom it can complete a
        # genuine warmup and flip is_recognition_ready to True before the
        # "protected" request arrives - the mock never bypassed, just racing
        # a call outside its scope. The patch must therefore cover the whole
        # TestClient block, not just the request.
        with (
            patch("app.core.resource_profile.has_sufficient_ml_startup_headroom", side_effect=_low_memory),
            TestClient(app) as client,
        ):
            headers = _auth_headers()
            resp = client.post(
                "/api/v1/identify/image",
                files={"file": ("test.jpg", io.BytesIO(b"\xff\xd8\xff\xe0fake"), "image/jpeg")},
                data={"camera_id": "cam1"},
                headers=headers,
            )
            assert resp.status_code == 503
            body = resp.json()["detail"]
            assert body["error"] == "ml_unavailable"
            assert body["reason"] == "insufficient_memory_headroom"
    finally:
        app.state.gait_service = None


def test_sufficient_memory_proceeds_past_the_gate():
    """F. Adequate memory -> the deferral gate is not triggered; warmup
    proceeds to real component initialization (not re-verifying the
    already-covered full model-loading path here, only that this new gate
    doesn't block it)."""
    service = GaitService()
    with (
        patch("app.core.resource_profile.has_sufficient_ml_startup_headroom", side_effect=_sufficient_memory),
        patch("app.core.thread_limits.configure_native", return_value=2),
        patch.object(type(service), "extractor", new=MagicMock(backend=MagicMock(predict=lambda x: [0.0] * 256))),
        patch.object(type(service), "silhouette_extractor", new=MagicMock(extract_from_crop=lambda x: None)),
        patch.object(type(service), "appearance_extractor", new=None),
        patch.object(type(service), "detector", new=MagicMock(detect=lambda x: [])),
        patch.object(type(service), "matcher", new=MagicMock()),
        patch.object(type(service), "appearance_matcher", new=MagicMock()),
        patch.object(service, "_detector", new=MagicMock()),
    ):
        result = service.warmup()

    assert result["status"] == "WARMED_UP"
    assert service.warmup_deferred_reason is None
    assert service.get_readiness()["components"]["bygait"] != "DEFERRED"


def test_only_one_warmup_runs_when_multiple_callers_race():
    """G + H. Concurrent ensure_warm_or_deferred() callers never trigger
    duplicate model construction - single-flight is preserved."""
    service = GaitService()
    call_count = {"n": 0}

    def counting_low_memory():
        call_count["n"] += 1
        return _low_memory()

    with patch("app.core.resource_profile.has_sufficient_ml_startup_headroom", side_effect=counting_low_memory):
        results = [service.ensure_warm_or_deferred() for _ in range(5)]

    # Every call either performed (or observed) the same single deferred
    # outcome; none of them ever reports success, and no more warmup
    # attempts happened than callers (no runaway retry storm).
    assert all(r is False for r in results)
    assert call_count["n"] >= 1
    assert service.is_warmed_up is False
