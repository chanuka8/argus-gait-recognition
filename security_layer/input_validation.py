"""ARGUS AI — Centralized Input Validation for Filesystem-Safe Identifiers.

Security Guarantees:
  - Defence-in-depth: both input format validation AND canonical path containment.
  - Rejects path traversal sequences (../, ..\\, encoded variants).
  - Rejects absolute paths, UNC paths, drive-qualified paths.
  - Rejects control characters, null bytes.
  - Validates that the resolved filesystem path remains inside the intended storage root.
  - Windows path semantics are handled via Path.resolve() + is_relative_to().

Usage:
  validate_person_id(person_id)           → raises HTTPException(400) on invalid input
  validate_path_containment(base, target) → raises HTTPException(400) on escape
  sanitize_person_id_for_filename(pid)    → returns filesystem-safe string
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

from fastapi import HTTPException, status

# Conservative identifier pattern: letters, digits, spaces, hyphens, underscores, periods.
# This preserves compatibility with existing ARGUS person names while blocking path separators.
_VALID_PERSON_ID_RE = re.compile(r"^[a-zA-Z0-9 _\-\.]+$")

# Characters safe for filesystem filenames (superset used when sanitizing for filenames)
_FILENAME_SAFE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- ")

# Video ingestion and upload memory safety bounds (SEC-07)
MAX_REFERENCE_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MiB (realistic high-bitrate CCTV/reference videos)
MAX_ANALYZE_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MiB (for fast synchronous analysis)
MAX_UPLOAD_CHUNK_SIZE = 10 * 1024 * 1024  # 10 MiB (default is 2 MiB, 10 MiB upper bound)
MIN_UPLOAD_CHUNK_SIZE = 1  # Allows small chunks for test cases / low bandwidth
MAX_VIDEO_DIMENSION = 4096  # Supports up to 4K UHD; blocks 8K/16K decompression bombs
MAX_VIDEO_FRAMES = 100_000  # Supports ~55 minutes at 30 fps; blocks extreme decompression loops
CHUNK_STREAM_BUFFER_SIZE = 64 * 1024  # 64 KiB bounded transfer chunk
MAX_TRACKS_WITH_CROPS = 50  # Matches crowd_control.max_tracked_people_per_camera = 50
MAX_CROPS_PER_TRACK = 120  # Matches existing 120 crop limit


def validate_person_id(person_id: str) -> str:
    """Validate and normalize a person_id for safe filesystem use.

    Layer A: Input format validation.
    Layer B: Structural path traversal detection (decoded and raw).

    Returns the stripped, validated person_id.
    Raises HTTPException(400) if the person_id is invalid.
    """
    if not person_id or not person_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="person_id is required",
        )

    normalized = person_id.strip()

    # ── Layer B-1: Reject null bytes and control characters ──
    if any(ord(c) < 32 for c in normalized) or "\x00" in normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="person_id contains invalid control characters",
        )

    # ── Layer B-2: Decode URL-encoded sequences and check for traversal ──
    # Decode up to 3 levels to catch double/triple encoding
    decoded = normalized
    for _ in range(3):
        prev = decoded
        decoded = unquote(decoded)
        if decoded == prev:
            break

    # ── Layer B-3: Reject any form of path traversal or absolute path ──
    _reject_path_traversal(normalized, "person_id")
    _reject_path_traversal(decoded, "person_id")

    # ── Layer A: Format validation ──
    if not _VALID_PERSON_ID_RE.match(normalized):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid person_id format. "
                "Allowed characters: letters, numbers, spaces, hyphens, underscores, and periods."
            ),
        )

    # Reject names that are only dots (e.g., "..", ".", "...")
    if all(c == "." for c in normalized):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="person_id cannot consist entirely of dots",
        )

    return normalized


def _reject_path_traversal(value: str, field_name: str) -> None:
    """Reject values containing path traversal sequences, absolute paths, or UNC paths."""
    # Check for path separator characters
    if "/" in value or "\\" in value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must not contain path separator characters",
        )

    # Check for drive-qualified paths (e.g., C:, D:)
    if len(value) >= 2 and value[1] == ":":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must not contain drive-qualified paths",
        )

    # Check for traversal sequences (even without separators, after decoding)
    if ".." in value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must not contain path traversal sequences",
        )


def validate_path_containment(base_dir: Path, target_path: Path) -> Path:
    """Verify that target_path is strictly contained within base_dir using canonical resolution.

    This is the mandatory Layer B filesystem containment check.
    It protects against any bypass of input validation.

    Returns the resolved target path.
    Raises HTTPException(400) if target_path escapes base_dir.
    """
    resolved_base = base_dir.resolve()
    resolved_target = target_path.resolve()

    if not resolved_target.is_relative_to(resolved_base):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path escapes the allowed storage directory",
        )

    return resolved_target


def sanitize_person_id_for_filename(person_id: str) -> str:
    """Convert a validated person_id into a filesystem-safe filename component.

    Replaces any non-safe characters with underscores.
    This is a secondary safety net for filename construction.
    """
    return "".join(c if c in _FILENAME_SAFE_CHARS else "_" for c in person_id)
