"""
ARGUS AI Real-Time Package & Large File Streaming Downloader.

Provides live terminal download monitoring with in-place progress bars,
download speed, ETA calculation, HTTP range resumption, SHA-256 validation,
and robust network auto-retry.

Usage:
    python scripts/download_package.py <URL> <OUTPUT_PATH> [--name <NAME>] [--version <VER>] [--platform <PLAT>] [--source <SRC>] [--retries <N>] [--sha256 <HASH>]
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def format_bytes(num_bytes: float) -> str:
    """Format bytes to human-readable string (KB, MB, GB)."""
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_time(seconds: float) -> str:
    """Format seconds to MM:SS or HH:MM:SS."""
    if seconds < 0 or seconds > 86400 * 7:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def render_progress_bar(
    package_name: str,
    downloaded_bytes: int,
    total_bytes: int | None,
    speed_bps: float,
    start_time: float,
    is_tty: bool,
    bar_width: int = 24,
) -> None:
    """Render a live progress bar to stdout."""
    elapsed = time.time() - start_time
    speed_str = f"{format_bytes(speed_bps)}/s"

    if total_bytes and total_bytes > 0:
        percent = min(100.0, (downloaded_bytes / total_bytes) * 100.0)
        filled = int(bar_width * (percent / 100.0))
        bar = "█" * filled + "░" * (bar_width - filled)
        remaining_bytes = max(0, total_bytes - downloaded_bytes)
        eta_seconds = (remaining_bytes / speed_bps) if speed_bps > 0 else 0
        eta_str = format_time(eta_seconds)

        line = (
            f"\r[ARGUS DOWNLOAD] [{bar}] {percent:5.1f}% | "
            f"{format_bytes(downloaded_bytes)} / {format_bytes(total_bytes)} | "
            f"Speed: {speed_str} | ETA: {eta_str} | Elapsed: {format_time(elapsed)}"
        )
    else:
        line = (
            f"\r[ARGUS DOWNLOAD] {format_bytes(downloaded_bytes)} downloaded | "
            f"Speed: {speed_str} | Elapsed: {format_time(elapsed)}"
        )

    if is_tty:
        sys.stdout.write(line)
        sys.stdout.flush()
    else:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def download_file(
    url: str,
    output_path: Path,
    package_name: str | None = None,
    version: str | None = None,
    platform: str | None = None,
    source: str | None = None,
    expected_sha256: str | None = None,
    max_retries: int = 5,
    chunk_size: int = 256 * 1024,
) -> bool:
    """
    Download a file with real-time visual progress, automatic resumption, and retries.
    """
    pkg_name = package_name or output_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")

    is_tty = sys.stdout.isatty()
    attempt = 0

    print("\n" + "=" * 60, flush=True)
    print("[DOWNLOAD]", flush=True)
    print(f"Package      : {pkg_name}", flush=True)
    if version:
        print(f"Version      : {version}", flush=True)
    if platform:
        print(f"Platform     : {platform}", flush=True)
    if source:
        print(f"Source       : {source}", flush=True)
    print(f"Destination  : {output_path}", flush=True)
    print("=" * 60 + "\n", flush=True)

    while attempt < max_retries:
        attempt += 1
        existing_bytes = part_path.stat().st_size if part_path.exists() else 0
        headers = {
            "User-Agent": "ARGUS-AI-Environment-Bootstrap/1.0",
        }

        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"
            print(f"[ARGUS DOWNLOAD] Existing partial data: {format_bytes(existing_bytes)}", flush=True)
            print(
                f"[ARGUS DOWNLOAD] Resuming from byte {existing_bytes} (Attempt {attempt}/{max_retries})...", flush=True
            )
        elif attempt > 1:
            print(f"[ARGUS DOWNLOAD] Retrying connection (Attempt {attempt}/{max_retries})...", flush=True)

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                status_code = getattr(response, "status", 200)
                content_length_header = response.headers.get("Content-Length")
                content_range_header = response.headers.get("Content-Range")

                total_bytes = None
                if content_range_header:
                    try:
                        total_bytes = int(content_range_header.split("/")[-1])
                    except (ValueError, IndexError):
                        pass
                elif content_length_header:
                    total_bytes = int(content_length_header)
                    if status_code == 206:
                        total_bytes += existing_bytes

                file_mode = "ab" if (existing_bytes > 0 and status_code == 206) else "wb"
                if file_mode == "wb":
                    existing_bytes = 0

                downloaded_in_session = 0
                start_time = time.time()
                last_render_time = 0.0
                speed_calc_window = []

                with open(part_path, file_mode) as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        f.write(chunk)
                        chunk_len = len(chunk)
                        downloaded_in_session += chunk_len
                        current_total = existing_bytes + downloaded_in_session

                        now = time.time()
                        speed_calc_window.append((now, chunk_len))
                        speed_calc_window = [(t, b) for (t, b) in speed_calc_window if now - t <= 2.0]
                        window_bytes = sum(b for (_, b) in speed_calc_window)
                        window_time = max(0.01, now - speed_calc_window[0][0]) if speed_calc_window else 0.01
                        speed_bps = window_bytes / window_time

                        if now - last_render_time >= (0.1 if is_tty else 1.5):
                            render_progress_bar(
                                pkg_name,
                                current_total,
                                total_bytes,
                                speed_bps,
                                start_time,
                                is_tty,
                            )
                            last_render_time = now

                # Final 100% progress render
                final_total = part_path.stat().st_size
                render_progress_bar(
                    pkg_name,
                    final_total,
                    final_total,
                    0.0,
                    start_time,
                    is_tty,
                )
                if is_tty:
                    sys.stdout.write("\n")
                sys.stdout.flush()

                # Validate SHA-256 if provided
                if expected_sha256:
                    print("[ARGUS DOWNLOAD] Validating SHA-256 checksum...", flush=True)
                    hasher = hashlib.sha256()
                    with open(part_path, "rb") as f:
                        while chunk := f.read(1024 * 1024):
                            hasher.update(chunk)
                    computed_hash = hasher.hexdigest()
                    if computed_hash.lower() != expected_sha256.lower():
                        print(f"[ARGUS ERROR] Checksum mismatch for {pkg_name}!", flush=True)
                        print(f"  Expected: {expected_sha256}", flush=True)
                        print(f"  Computed: {computed_hash}", flush=True)
                        part_path.unlink(missing_ok=True)
                        return False
                    print("[ARGUS DOWNLOAD] Checksum: VERIFIED (PASS)", flush=True)

                # Atomically replace part file to target destination
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except OSError:
                        pass
                part_path.rename(output_path)
                print(
                    f"[ARGUS DOWNLOAD] ✓ Successfully downloaded: {pkg_name} ({format_bytes(final_total)})\n",
                    flush=True,
                )
                return True

        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as err:
            print(f"\n[ARGUS DOWNLOAD] Connection interrupted ({err}).", flush=True)
            if attempt < max_retries:
                time.sleep(min(10, 2 * attempt))
            else:
                print(f"\n[ARGUS ERROR] Failed to download {pkg_name} after {max_retries} attempts.", flush=True)
                return False

    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="ARGUS AI Live Streaming Downloader")
    parser.add_argument("url", help="Download URL")
    parser.add_argument("output", help="Destination file path")
    parser.add_argument("--name", default=None, help="Display package name")
    parser.add_argument("--version", default=None, help="Package version string")
    parser.add_argument("--platform", default=None, help="Platform tag")
    parser.add_argument("--source", default=None, help="Download source label")
    parser.add_argument("--sha256", default=None, help="Expected SHA-256 hash")
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.environ.get("ARGUS_DOWNLOAD_RETRIES", "5")),
        help="Maximum retry attempts",
    )

    args = parser.parse_args()
    success = download_file(
        url=args.url,
        output_path=Path(args.output),
        package_name=args.name,
        version=args.version,
        platform=args.platform,
        source=args.source,
        expected_sha256=args.sha256,
        max_retries=args.retries,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
