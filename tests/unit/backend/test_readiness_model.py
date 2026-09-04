from fastapi.testclient import TestClient

from api.server import app
from services.gait_service import GaitService


def test_gait_service_initial_readiness():
    """Verify that GaitService initializes with API_READY and GALLERY_READY without waiting for model warmup."""
    service = GaitService()
    readiness = service.get_readiness()

    assert readiness["api_ready"] is True
    assert readiness["states"]["API_READY"] is True
    assert readiness["states"]["GALLERY_READY"] is True
    assert readiness["states"]["RECOGNITION_READY"] is False
    assert readiness["components"]["api"] == "READY"
    assert readiness["components"]["gallery"] == "READY"
    assert service.is_recognition_ready is False


def test_health_and_readiness_endpoints():
    """Verify that /api/v1/health and /api/v1/readiness return immediately with granular readiness state."""
    with TestClient(app) as client:
        # Test /api/v1/health
        health_resp = client.get("/api/v1/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert "readiness" in health_data
        assert health_data["readiness"]["API_READY"] is True

        # Test /api/v1/readiness
        ready_resp = client.get("/api/v1/readiness")
        assert ready_resp.status_code == 200
        ready_data = ready_resp.json()
        assert ready_data["api_ready"] is True
        assert "states" in ready_data
        assert "components" in ready_data
        assert ready_data["states"]["API_READY"] is True


def test_warmup_transitions_recognition_ready():
    """Verify that completing warmup transitions all components to READY and RECOGNITION_READY to True."""
    service = GaitService()
    warmup_res = service.warmup()

    assert warmup_res["status"] == "WARMED_UP"
    assert service.is_warmed_up is True
    assert service.is_recognition_ready is True

    readiness = service.get_readiness()
    assert readiness["recognition_ready"] is True
    assert readiness["states"]["BYGAIT_READY"] is True
    assert readiness["states"]["OSNET_READY"] is True
    assert readiness["states"]["SILHOUETTE_READY"] is True
    assert readiness["states"]["RECOGNITION_READY"] is True
    assert readiness["components"]["bygait"] == "READY"
    assert readiness["components"]["osnet"] == "READY"
    assert readiness["components"]["silhouette"] == "READY"
