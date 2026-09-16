"""SEC-09: Centralized API error and exception sanitization helper for ARGUS AI.

Ensures that internal exception details (such as filesystem paths, database
connection strings, tokens, credentials, stack traces, and internal Python
exception types) are never disclosed to external API clients, while preserving
legitimate client input validation messages and maintaining rich server-side
diagnostic logging.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("ARGUS.ErrorSanitizer")

# Patterns that indicate an internal, OS, or sensitive exception string
_SENSITIVE_PATTERNS = (
    r"[a-zA-Z]:\\",  # Windows absolute drive paths (e.g. C:\)
    r"/[a-zA-Z0-9_\-\.]+(?:/[a-zA-Z0-9_\-\.]+)+",  # Unix paths (e.g. /data/...)
    r"\.py\b",  # Python source files
    r"\.pem\b|\.key\b|\.enc\b|\.env\b",  # Keys and secret files
    r"\[Errno\s+\d+\]",  # OS error numbers
    r"Traceback\s*\(most\s+recent",  # Python tracebacks
    r"File\s+[\"'].*?[\"'],\s+line",  # Traceback line pointers
    r"postgresql://|mysql://|mongodb://|sqlite://|http://|https://",  # Connection URIs
    r"password|secret|token|bearer|private_key",  # Credentials
    r"\b(?:RuntimeError|ValueError|TypeError|KeyError|IndexError|AttributeError|OSError|FileNotFoundError|PermissionError):",  # Exception type markers
    r"cv2\.error",  # OpenCV internal C++ exceptions
)

_SENSITIVE_REGEX = re.compile("|".join(_SENSITIVE_PATTERNS), re.IGNORECASE)


def is_sensitive_exception_text(text: str) -> bool:
    """Return True if text appears to contain internal file paths, tokens,
    OS error indicators, or raw Python exception representations."""
    if not text:
        return False
    return bool(_SENSITIVE_REGEX.search(text))


def sanitize_validation_error(
    error: Any,
    fallback_message: str,
    allowed_known_messages: tuple[str, ...] = (),
    allowed_prefixes: tuple[str, ...] = (),
) -> str:
    """Validate whether an error string is an intentional user-facing validation
    message that is safe to return to the client.

    If the error text matches an allowed known message or starts with an allowed
    prefix AND does not contain sensitive leakage patterns, it is returned.
    Otherwise, fallback_message is returned.
    """
    raw_text = str(error).strip() if error is not None else ""
    if not raw_text:
        return fallback_message

    # Never return text containing sensitive indicators
    if is_sensitive_exception_text(raw_text):
        return fallback_message

    # Check exact allowed messages
    if raw_text in allowed_known_messages:
        return raw_text

    # Check allowed prefixes
    for prefix in allowed_prefixes:
        if raw_text.startswith(prefix):
            return raw_text

    return fallback_message
