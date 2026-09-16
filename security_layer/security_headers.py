"""
Standard HTTP Security Headers Middleware for ARGUS AI (SEC-08).

Provides centralized, deterministic HTTP security headers across all API,
error, static, SPA, and documentation endpoints, preserving non-HTTP (WebSocket)
traffic untouched.
"""

import os

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Core security headers
HEADER_X_CONTENT_TYPE_OPTIONS = "nosniff"
HEADER_X_FRAME_OPTIONS = "DENY"
HEADER_REFERRER_POLICY = "strict-origin-when-cross-origin"
HEADER_COOP = "same-origin"
HEADER_CORP = "same-origin"

# ARGUS Permissions-Policy:
# Verified against frontend source: ARGUS does not use browser camera getUserMedia,
# fullscreen API, video autoplay, microphone, geolocation, screen capture, payment, or usb.
# All capabilities are strictly disabled for maximum defense-in-depth.
HEADER_PERMISSIONS_POLICY = (
    "camera=(), fullscreen=(), autoplay=(), microphone=(), geolocation=(), display-capture=(), payment=(), usb=()"
)

# HSTS Policy for HTTPS deployments (365 days + subdomains)
HEADER_HSTS = "max-age=31536000; includeSubDomains"

# Application Content Security Policy (React SPA, API, and Static Files)
# - script-src 'self' (NO unsafe-eval, NO unsafe-inline)
# - style-src 'self' 'unsafe-inline' https://fonts.googleapis.com (required for React inline styles, Leaflet, Google Fonts)
# - img-src 'self' data: blob: https://api.dicebear.com https://*.tile.openstreetmap.org https://*.googleapis.com
# - font-src 'self' https://fonts.gstatic.com data:
# - connect-src 'self' ws: wss: https://nominatim.openstreetmap.org https://*.tile.openstreetmap.org https://*.firebaseio.com wss://*.firebaseio.com https://*.googleapis.com
# - media-src 'self' blob: data:
# - worker-src 'self' blob:
# - frame-src 'none'
# - object-src 'none'
# - frame-ancestors 'none'
# - base-uri 'self'
# - form-action 'self'
CSP_APPLICATION = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "img-src 'self' data: blob: https://api.dicebear.com https://*.tile.openstreetmap.org https://*.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "connect-src 'self' ws: wss: https://nominatim.openstreetmap.org https://*.tile.openstreetmap.org https://*.firebaseio.com wss://*.firebaseio.com https://*.googleapis.com; "
    "media-src 'self' blob: data:; "
    "worker-src 'self' blob:; "
    "frame-src 'none'; "
    "object-src 'none'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

# Documentation CSP (Swagger UI / ReDoc on /docs, /redoc, /openapi.json)
# Requires CDN script/style sources and inline script for Swagger UI bundle initialization.
CSP_DOCS = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.redoc.ly; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
    "img-src 'self' data: blob: https://fastapi.tiangolo.com https://cdn.jsdelivr.net; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "connect-src 'self'; "
    "frame-src 'none'; "
    "object-src 'none'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

# Cache-Control policies
CACHE_CONTROL_SENSITIVE = "no-store, no-cache, must-revalidate, private"
CACHE_CONTROL_STATIC = "public, max-age=31536000, immutable"
CACHE_CONTROL_SPA = "no-cache, must-revalidate"


def is_https_request(scope: Scope, req_headers: Headers) -> bool:
    """
    Determines if request was served over HTTPS or terminated at a TLS-enabled reverse proxy,
    or if HSTS is explicitly forced via ARGUS_ENFORCE_HSTS.
    """
    if os.getenv("ARGUS_ENFORCE_HSTS", "").strip().lower() in ("true", "1", "yes"):
        return True

    scheme = scope.get("scheme", "")
    if scheme == "https":
        return True

    forwarded_proto = req_headers.get("x-forwarded-proto", "")
    if forwarded_proto.lower() == "https":
        return True

    forwarded = req_headers.get("forwarded", "")
    return "proto=https" in forwarded.lower()


def get_csp_for_path(path: str) -> str:
    """Returns tailored CSP for documentation versus application endpoints."""
    clean_path = path.rstrip("/")
    if clean_path in ("/docs", "/redoc", "/openapi.json"):
        return CSP_DOCS
    return CSP_APPLICATION


def apply_cache_control(headers: MutableHeaders, path: str) -> None:
    """
    Applies appropriate Cache-Control headers if not already set by endpoint.
    Sensitive API responses receive 'no-store, no-cache, must-revalidate, private'.
    Static assets receive 'public, max-age=31536000, immutable'.
    SPA root/HTML receives 'no-cache, must-revalidate'.
    """
    if "cache-control" in headers:
        return

    if path.startswith("/assets/"):
        headers["cache-control"] = CACHE_CONTROL_STATIC
    elif (
        path in ("/", "")
        or path.endswith(".html")
        or not (path.startswith("/api/") or path in ("/health", "/status", "/metrics"))
    ):
        headers["cache-control"] = CACHE_CONTROL_SPA
    else:
        # Sensitive API endpoints, surveillance routes, metrics, etc.
        headers["cache-control"] = CACHE_CONTROL_SENSITIVE
        headers["pragma"] = "no-cache"


def inject_security_headers(headers: MutableHeaders, path: str, is_https: bool) -> None:
    """Injects all standard security headers into response headers."""
    # 1. MIME-type sniffing protection
    headers["x-content-type-options"] = HEADER_X_CONTENT_TYPE_OPTIONS

    # 2. Clickjacking protection
    headers["x-frame-options"] = HEADER_X_FRAME_OPTIONS

    # 3. Referrer Policy
    headers["referrer-policy"] = HEADER_REFERRER_POLICY

    # 4. Cross-Origin policies
    headers["cross-origin-opener-policy"] = HEADER_COOP
    headers["cross-origin-resource-policy"] = HEADER_CORP

    # 5. Permissions-Policy
    headers["permissions-policy"] = HEADER_PERMISSIONS_POLICY

    # 6. Content-Security-Policy
    headers["content-security-policy"] = get_csp_for_path(path)

    # 7. Strict-Transport-Security (conditional on HTTPS / reverse proxy / explicit env)
    if is_https:
        headers["strict-transport-security"] = HEADER_HSTS

    # 8. Cache-Control (response-aware)
    apply_cache_control(headers, path)


class SecurityHeadersMiddleware:
    """
    Pure ASGI middleware injecting HTTP security headers.
    Completely ignores non-HTTP scopes (such as WebSockets).
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        req_headers = Headers(scope=scope)
        path = scope.get("path", "")
        is_https = is_https_request(scope, req_headers)

        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = MutableHeaders(scope=message)
                inject_security_headers(headers, path, is_https)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            if not response_started:
                resp = PlainTextResponse("Internal Server Error", status_code=500)
                mutable = MutableHeaders(raw=resp.raw_headers)
                inject_security_headers(mutable, path, is_https)
                await resp(scope, receive, send)
            else:
                raise
