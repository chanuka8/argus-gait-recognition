import pytest
from fastapi.testclient import TestClient

from app.api.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_normal_frontend_static_asset_served(client):
    """Normal frontend assets in frontend/dist are served with HTTP 200."""
    resp = client.get("/logo.png")
    assert resp.status_code == 200
    assert "image/png" in resp.headers.get("content-type", "")
    assert len(resp.content) > 0


def test_root_route_serves_frontend_index(client):
    """Root route / serves index.html."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_spa_client_routes_fallback_to_index(client):
    """SPA client-side navigation routes fall back to index.html with HTTP 200."""
    for route in ["/dashboard", "/cctv", "/cctv/network", "/missing-persons", "/cases"]:
        resp = client.get(route)
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


def test_encoded_path_traversal_requirements_txt_rejected_with_404(client):
    """Encoded traversal /..%2f..%2frequirements.txt must return 404 without leaking requirements.txt."""
    resp = client.get("/..%2f..%2frequirements.txt")
    assert resp.status_code == 404
    data = resp.json()
    assert data.get("detail") == "Not Found"
    assert "fastapi" not in resp.text.lower()
    assert "torch" not in resp.text.lower()


def test_encoded_path_traversal_env_example_rejected_with_404(client):
    """Encoded traversal /..%2f..%2f.env.example must return 404 without leaking .env.example."""
    resp = client.get("/..%2f..%2f.env.example")
    assert resp.status_code == 404
    data = resp.json()
    assert data.get("detail") == "Not Found"
    assert "ARGUS" not in resp.text
    assert "FIREBASE" not in resp.text


def test_deep_path_traversal_rejected_with_404(client):
    """Arbitrary deep path traversals outside frontend/dist are rejected with 404."""
    traversal_paths = [
        "/..%2f..%2f..%2f..%2fWindows%2fwin.ini",
        "/..%2f..%2f..%2f..%2fetc%2fpasswd",
        "/..%2f..%2fapi%2fserver.py",
        "/assets/..%2f..%2f..%2frequirements.txt",
        "/..%5c..%5crequirements.txt",
        "/nested/..%2f..%2f..%2frequirements.txt",
    ]
    for path in traversal_paths:
        resp = client.get(path)
        assert resp.status_code == 404
        assert resp.json().get("detail") == "Not Found"


def test_encoded_backslash_path_traversal_rejected_with_404(client):
    """Encoded backslash traversal /..%5c..%5crequirements.txt must return 404 without falling back to index.html."""
    for path in ["/..%5c..%5crequirements.txt", "/..%5c..%5c..%5cetc%5cpasswd", "/..%5c..%5c.env.example"]:
        resp = client.get(path)
        assert resp.status_code == 404
        data = resp.json()
        assert data.get("detail") == "Not Found"
        assert "<!doctype html>" not in resp.text.lower()


def test_mixed_slash_backslash_path_traversal_rejected_with_404(client):
    """Mixed slash and backslash traversals must return 404 without falling back to index.html."""
    for path in ["/..%2f..%5crequirements.txt", "/..%5c..%2frequirements.txt", "/nested/..%5c..%2f..%2frequirements.txt"]:
        resp = client.get(path)
        assert resp.status_code == 404
        data = resp.json()
        assert data.get("detail") == "Not Found"
        assert "<!doctype html>" not in resp.text.lower()


def test_reserved_prefixes_not_routed_to_spa(client):
    """API, docs, and health endpoints are not intercepted by the SPA handler."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "healthy"

    resp = client.get("/health")
    assert resp.status_code == 200

    resp = client.get("/docs")
    assert resp.status_code == 200
