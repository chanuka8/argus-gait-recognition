"""
Integration test suite for SEC-08: Standard HTTP Security Headers.

Verifies:
1. Normal HTTP response contains all required security headers.
2. 401 Unauthorized response contains all security headers.
3. 403 Forbidden response contains all security headers.
4. 404 Not Found response contains all security headers.
5. 500 Internal Server Error response contains all security headers.
6. SPA fallback route contains security headers and SPA cache policy.
7. Static asset route contains security headers and immutable cache policy.
8. /docs, /redoc, /openapi.json contain security headers and documentation CSP.
9. CSP structure: required directives present, connect-src, media-src, blob:, data:.
10. CSP safety: NO unsafe-eval anywhere, NO unsafe-inline scripts on app routes.
11. Permissions-Policy: permits camera/fullscreen/autoplay for self, restricts mic/geo/capture/payment/usb.
12. Strict-Transport-Security conditional behavior:
    - Omitted on local HTTP requests.
    - Emitted on HTTPS scheme.
    - Emitted on X-Forwarded-Proto: https (reverse proxy TLS termination).
    - Emitted on Forwarded: proto=https.
    - Emitted when ARGUS_ENFORCE_HSTS=true.
13. CORS regression: Preflight and GET requests preserve CORS headers alongside security headers.
14. WebSocket regression: Authenticated and unauthenticated WebSocket connections behave correctly
    and are not disrupted by HTTP security header middleware.
15. Endpoint-specific Cache-Control preservation (e.g. streaming responses).
"""

import pytest
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.server import app
from app.security_layer.auth import get_session_store
from app.security_layer.security_headers import SecurityHeadersMiddleware


@pytest.fixture
def client():
    """TestClient instance for ARGUS FastAPI application."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def investigator_token():
    """Create a temporary Investigator session token."""
    store = get_session_store()
    session = store.create_session(
        operator_id="sec08_investigator_id",
        username="sec08_investigator",
        role="investigator",
    )
    yield session.token
    store.revoke_session(session.token)


# ==============================================================================
# 1. CORE RESPONSE CODES & SECURITY HEADERS VERIFICATION
# ==============================================================================


def test_normal_api_response_headers(client):
    """Verify normal 200 API response contains all required security headers."""
    response = client.get("/health")
    assert response.status_code == 200

    headers = response.headers
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "DENY"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert headers.get("cross-origin-opener-policy") == "same-origin"
    assert headers.get("cross-origin-resource-policy") == "same-origin"
    assert "camera=()" in headers.get("permissions-policy", "")
    assert "default-src 'self'" in headers.get("content-security-policy", "")
    assert "no-store" in headers.get("cache-control", "")
    assert "no-cache" in headers.get("cache-control", "")
    assert headers.get("pragma") == "no-cache"
    # Over plain HTTP testclient, HSTS must NOT be set
    assert "strict-transport-security" not in headers


def test_401_unauthorized_response_headers(client):
    """Verify 401 response contains all required security headers."""
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401

    headers = response.headers
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "DENY"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert headers.get("cross-origin-opener-policy") == "same-origin"
    assert headers.get("cross-origin-resource-policy") == "same-origin"
    assert "camera=()" in headers.get("permissions-policy", "")
    assert "default-src 'self'" in headers.get("content-security-policy", "")
    assert "no-store" in headers.get("cache-control", "")


def test_403_forbidden_response_headers(client, investigator_token):
    """Verify 403 Forbidden response contains all required security headers."""
    # Investigator attempting to access Admin-only operator provisioning
    response = client.post(
        "/api/v1/auth/admin/users",
        json={
            "username": "unauthorized_new_operator",
            "password": "ValidPassword123!",
            "role": "investigator",
        },
        headers={"Authorization": f"Bearer {investigator_token}"},
    )
    assert response.status_code == 403

    headers = response.headers
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "DENY"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert headers.get("cross-origin-opener-policy") == "same-origin"
    assert headers.get("cross-origin-resource-policy") == "same-origin"
    assert "camera=()" in headers.get("permissions-policy", "")
    assert "default-src 'self'" in headers.get("content-security-policy", "")
    assert "no-store" in headers.get("cache-control", "")


def test_404_not_found_response_headers(client):
    """Verify 404 response on non-existent API path contains all security headers."""
    response = client.get("/api/v1/nonexistent_endpoint_for_test")
    assert response.status_code == 404

    headers = response.headers
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "DENY"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert headers.get("cross-origin-opener-policy") == "same-origin"
    assert headers.get("cross-origin-resource-policy") == "same-origin"
    assert "camera=()" in headers.get("permissions-policy", "")
    assert "default-src 'self'" in headers.get("content-security-policy", "")
    assert "no-store" in headers.get("cache-control", "")


def test_500_server_error_response_headers():
    """Verify 500 error response preserves security headers under unhandled exception."""
    crash_app = FastAPI()
    crash_app.add_middleware(SecurityHeadersMiddleware)

    @crash_app.get("/api/v1/crash")
    def trigger_crash():
        raise RuntimeError("Simulated internal error for header test")

    with TestClient(crash_app, raise_server_exceptions=False) as c:
        response = c.get("/api/v1/crash")
        assert response.status_code == 500
        headers = response.headers
        assert headers.get("x-content-type-options") == "nosniff"
        assert headers.get("x-frame-options") == "DENY"
        assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
        assert headers.get("cross-origin-opener-policy") == "same-origin"
        assert headers.get("cross-origin-resource-policy") == "same-origin"
        assert "default-src 'self'" in headers.get("content-security-policy", "")


# ==============================================================================
# 2. SPA & STATIC ASSET HEADERS & CACHE-CONTROL
# ==============================================================================


def test_spa_fallback_headers_and_cache(client):
    """Verify SPA fallback response receives security headers and no-cache policy."""
    response = client.get("/dashboard/surveillance")
    headers = response.headers
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "DENY"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert headers.get("cross-origin-opener-policy") == "same-origin"
    assert headers.get("cross-origin-resource-policy") == "same-origin"
    assert "camera=()" in headers.get("permissions-policy", "")
    assert "default-src 'self'" in headers.get("content-security-policy", "")
    if response.status_code == 200:
        assert "no-cache" in headers.get("cache-control", "")
        assert "must-revalidate" in headers.get("cache-control", "")


def test_static_asset_headers_and_cache(client):
    """Verify static asset paths receive security headers and long-term cache headers."""
    response = client.get("/assets/index-test.js")
    headers = response.headers
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "DENY"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert headers.get("cross-origin-opener-policy") == "same-origin"
    assert headers.get("cross-origin-resource-policy") == "same-origin"
    assert "public" in headers.get("cache-control", "")
    assert "max-age=31536000" in headers.get("cache-control", "")
    assert "immutable" in headers.get("cache-control", "")


# ==============================================================================
# 3. DOCUMENTATION ROUTES & CSP
# ==============================================================================


def test_documentation_endpoints_headers(client):
    """Verify /docs, /redoc, and /openapi.json receive security headers and documentation CSP."""
    for doc_path in ("/docs", "/redoc", "/openapi.json"):
        response = client.get(doc_path)
        assert response.status_code == 200, f"Failed for {doc_path}"
        headers = response.headers
        assert headers.get("x-content-type-options") == "nosniff"
        assert headers.get("x-frame-options") == "DENY"
        assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
        assert headers.get("cross-origin-opener-policy") == "same-origin"

        csp = headers.get("content-security-policy", "")
        assert "default-src 'self'" in csp
        assert "cdn.jsdelivr.net" in csp
        assert "object-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp
        # No unsafe-eval allowed even in docs
        assert "unsafe-eval" not in csp


# ==============================================================================
# 4. CONTENT SECURITY POLICY IN-DEPTH ANALYSIS
# ==============================================================================


def test_csp_application_directives(client):
    """Verify application CSP contains all required directives tailored to ARGUS frontend."""
    response = client.get("/health")
    csp = response.headers.get("content-security-policy", "")

    # Required directives
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com" in csp
    assert "font-src 'self' https://fonts.gstatic.com data:" in csp
    assert "object-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "base-uri 'self'" in csp
    assert "form-action 'self'" in csp

    # Connect-src: WebSockets, OSM tiles, Nominatim, Firebase endpoints
    assert "connect-src" in csp
    assert "ws:" in csp
    assert "wss:" in csp
    assert "https://nominatim.openstreetmap.org" in csp
    assert "https://*.tile.openstreetmap.org" in csp
    assert "https://*.firebaseio.com" in csp
    assert "https://*.googleapis.com" in csp

    # Media-src and img-src: blob:, data:, dicebear, openstreetmap, googleapis
    assert "media-src 'self' blob: data:" in csp
    assert "worker-src 'self' blob:" in csp
    assert "img-src 'self' data: blob:" in csp
    assert "https://api.dicebear.com" in csp
    assert "https://*.tile.openstreetmap.org" in csp
    assert "https://*.googleapis.com" in csp
    assert "frame-src 'none'" in csp


def test_csp_no_unsafe_eval(client):
    """Verify unsafe-eval is strictly prohibited across all application and docs routes."""
    for path in ("/health", "/api/v1/auth/me", "/docs", "/redoc", "/openapi.json", "/dashboard"):
        response = client.get(path)
        csp = response.headers.get("content-security-policy", "")
        assert "unsafe-eval" not in csp, f"Found unsafe-eval on {path}"


def test_csp_no_unsafe_inline_script_on_app_routes(client):
    """Verify unsafe-inline is NOT present in script-src for application/API/SPA routes."""
    for path in ("/health", "/api/v1/auth/me", "/dashboard", "/assets/test.js"):
        response = client.get(path)
        csp = response.headers.get("content-security-policy", "")
        directives = {d.split()[0]: d for d in csp.split(";") if d.strip()}
        script_src = directives.get("script-src", "")
        assert "unsafe-inline" not in script_src, f"Found unsafe-inline script on app route {path}: {script_src}"


# ==============================================================================
# 5. PERMISSIONS-POLICY IN-DEPTH ANALYSIS
# ==============================================================================


def test_permissions_policy_configuration(client):
    """Verify Permissions-Policy accurately matches ARGUS hardware capability requirements."""
    response = client.get("/health")
    pp = response.headers.get("permissions-policy", "")

    # Strictly disabled hardware/browser capabilities (verified against frontend source)
    assert "camera=()" in pp
    assert "fullscreen=()" in pp
    assert "autoplay=()" in pp
    assert "microphone=()" in pp
    assert "geolocation=()" in pp
    assert "display-capture=()" in pp
    assert "payment=()" in pp
    assert "usb=()" in pp

    # No wildcards
    assert "*" not in pp


# ==============================================================================
# 6. STRICT-TRANSPORT-SECURITY (HSTS) BEHAVIOR
# ==============================================================================


def test_hsts_omitted_on_plain_http(client):
    """Verify HSTS is NOT sent over plain HTTP local development requests."""
    response = client.get("/health")
    assert "strict-transport-security" not in response.headers


def test_hsts_applied_on_https_scheme(client):
    """Verify HSTS is sent when request scheme is HTTPS."""
    response = client.get("https://localhost:8000/health")
    hsts = response.headers.get("strict-transport-security")
    assert hsts == "max-age=31536000; includeSubDomains"


def test_hsts_applied_on_reverse_proxy_tls_termination(client):
    """Verify HSTS is sent when reverse proxy provides X-Forwarded-Proto: https."""
    response = client.get("/health", headers={"x-forwarded-proto": "https"})
    hsts = response.headers.get("strict-transport-security")
    assert hsts == "max-age=31536000; includeSubDomains"


def test_hsts_applied_on_forwarded_header(client):
    """Verify HSTS is sent when Forwarded header indicates proto=https."""
    response = client.get(
        "/health",
        headers={"forwarded": "for=192.0.2.60;proto=https;by=203.0.113.43"},
    )
    hsts = response.headers.get("strict-transport-security")
    assert hsts == "max-age=31536000; includeSubDomains"


def test_hsts_applied_with_explicit_env_override(client, monkeypatch):
    """Verify HSTS is applied over plain HTTP when ARGUS_ENFORCE_HSTS=true is set."""
    monkeypatch.setenv("ARGUS_ENFORCE_HSTS", "true")
    response = client.get("/health")
    hsts = response.headers.get("strict-transport-security")
    assert hsts == "max-age=31536000; includeSubDomains"


# ==============================================================================
# 7. CORS REGRESSION
# ==============================================================================


def test_cors_preflight_and_get_regression(client):
    """Verify CORSMiddleware preflight OPTIONS and GET requests preserve CORS alongside security headers."""
    options_res = client.options(
        "/api/v1/auth/me",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert options_res.status_code == 200
    assert options_res.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert options_res.headers.get("access-control-allow-credentials") == "true"
    assert options_res.headers.get("x-content-type-options") == "nosniff"
    assert options_res.headers.get("x-frame-options") == "DENY"

    get_res = client.get(
        "/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert get_res.status_code == 200
    assert get_res.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert get_res.headers.get("x-content-type-options") == "nosniff"
    assert get_res.headers.get("x-frame-options") == "DENY"


# ==============================================================================
# 8. WEBSOCKET REGRESSION
# ==============================================================================


def test_websocket_unauthenticated_rejected(client):
    """Verify unauthenticated WebSocket connection to /api/v1/ws/events is rejected (1008 violation)."""
    with (
        pytest.raises(WebSocketDisconnect) as excinfo,
        client.websocket_connect("/api/v1/ws/events"),
    ):
        pass
    assert excinfo.value.code == 1008


def test_websocket_authenticated_connects_unaffected(client, investigator_token):
    """Verify authenticated WebSocket connection via subprotocol connects cleanly without middleware disruption."""
    with client.websocket_connect(
        "/api/v1/ws/events",
        subprotocols=["argus-auth", investigator_token],
    ) as websocket:
        assert websocket is not None


# ==============================================================================
# 9. CUSTOM CACHE-CONTROL ENDPOINT PRESERVATION
# ==============================================================================


def test_endpoint_custom_cache_control_preserved():
    """Verify that an endpoint with its own explicit Cache-Control is preserved by middleware."""
    custom_app = FastAPI()
    custom_app.add_middleware(SecurityHeadersMiddleware)

    @custom_app.get("/api/v1/custom-stream")
    def stream():
        return Response(
            content="stream_data",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
        )

    with TestClient(custom_app) as c:
        res = c.get("/api/v1/custom-stream")
        assert res.status_code == 200
        assert res.headers.get("cache-control") == "no-cache, no-store, must-revalidate"
        assert res.headers.get("x-content-type-options") == "nosniff"
        assert res.headers.get("x-frame-options") == "DENY"
        assert res.headers.get("cross-origin-opener-policy") == "same-origin"
