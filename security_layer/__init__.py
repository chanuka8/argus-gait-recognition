"""
ARGUS Security Layer

Provides security event evaluation, logging, and secure RTSP credential resolution
for surveillance and gait recognition workflows.
"""

from security_layer.credentials import (
    CredentialManager,
    resolve_camera_config,
    sanitize_rtsp_url,
)

__all__ = [
    "CredentialManager",
    "resolve_camera_config",
    "sanitize_rtsp_url",
]
