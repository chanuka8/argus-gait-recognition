import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


class DownloadManager:
    CHUNK_SIZE = 1024 * 1024

    @staticmethod
    def _format_size(num_bytes: float) -> str:
        if num_bytes >= 1024**3:
            return f"{num_bytes / (1024**3):.2f} GB"
        if num_bytes >= 1024**2:
            return f"{num_bytes / (1024**2):.2f} MB"
        if num_bytes >= 1024:
            return f"{num_bytes / 1024:.2f} KB"
        return f"{num_bytes:.0f} B"

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 0 or seconds > 86400 * 7:
            return "--:--:--"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @classmethod
    def download_file(
        cls,
        url: str,
        dest_path: Path,
        package_name: str,
        expected_size: int | None = None,
        max_retries: int = 5,
        retry_delay_sec: float = 3.0,
    ) -> bool:
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dest_path.with_suffix(dest_path.suffix + ".part")

        for attempt in range(1, max_retries + 1):
            downloaded = 0
            if part_path.exists():
                downloaded = part_path.stat().st_size

            headers = {
                "User-Agent": "ARGUS-AI-Bootstrap/1.0",
            }
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"

            req = urllib.request.Request(url, headers=headers)

            print(f"\n[ARGUS DOWNLOAD] {package_name}")
            if attempt > 1:
                print(f"[ARGUS] Retry attempt {attempt}/{max_retries}...")

            if downloaded > 0:
                print(f"[ARGUS] Partial file preserved ({cls._format_size(downloaded)}). Resuming transfer...")

            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    status_code = getattr(response, "status", 200)
                    content_length = response.headers.get("Content-Length")

                    if status_code == 206:
                        total_size = downloaded + int(content_length) if content_length else (expected_size or 0)
                        mode = "ab"
                    else:
                        total_size = int(content_length) if content_length else (expected_size or 0)
                        mode = "wb"
                        downloaded = 0

                    start_time = time.monotonic()
                    bytes_this_session = 0
                    last_print_time = 0.0

                    with open(part_path, mode) as f_out:
                        while True:
                            chunk = response.read(cls.CHUNK_SIZE)
                            if not chunk:
                                break

                            f_out.write(chunk)
                            chunk_len = len(chunk)
                            downloaded += chunk_len
                            bytes_this_session += chunk_len

                            now = time.monotonic()
                            if now - last_print_time >= 0.25 or (total_size and downloaded >= total_size):
                                last_print_time = now
                                elapsed = now - start_time
                                speed = bytes_this_session / max(elapsed, 0.001)

                                percent = (downloaded / total_size * 100) if total_size > 0 else 0.0
                                percent = min(percent, 100.0)

                                remaining_bytes = max(0, total_size - downloaded) if total_size > 0 else 0
                                eta = (remaining_bytes / speed) if speed > 0 else 0.0

                                bar_len = 24
                                filled = int(bar_len * (percent / 100.0))
                                bar = "=" * filled + "-" * (bar_len - filled)

                                total_str = cls._format_size(total_size) if total_size > 0 else "Unknown"
                                down_str = cls._format_size(downloaded)
                                speed_str = f"{cls._format_size(speed)}/s"
                                eta_str = cls._format_time(eta)
                                el_str = cls._format_time(elapsed)

                                line = f"\r[{bar}] {percent:5.1f}% | {down_str} / {total_str} | Speed: {speed_str} | ETA: {eta_str} | Elapsed: {el_str}"
                                try:
                                    sys.stdout.write(line)
                                    sys.stdout.flush()
                                except UnicodeEncodeError:
                                    sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
                                    sys.stdout.flush()

                    sys.stdout.write("\n")
                    sys.stdout.flush()


                if total_size > 0 and part_path.stat().st_size < total_size:
                    print(
                        f"[ARGUS WARN] Downloaded size ({part_path.stat().st_size}) < expected ({total_size}). Retrying..."
                    )
                    time.sleep(retry_delay_sec)
                    continue

                if dest_path.exists():
                    dest_path.unlink()
                part_path.rename(dest_path)
                print(f"[ARGUS DOWNLOAD] Complete: {dest_path.name} ({cls._format_size(dest_path.stat().st_size)})\n")
                return True

            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as net_err:
                print(f"\n[ARGUS WARN] Transfer interrupted ({net_err}).")
                if attempt < max_retries:
                    print(f"[ARGUS] Retrying in {retry_delay_sec}s...")
                    time.sleep(retry_delay_sec)
                else:
                    print(f"[ARGUS ERROR] Download failed after {max_retries} attempts.")
                    return False
            except KeyboardInterrupt:
                print("\n[ARGUS] Download interrupted by user.")
                print(f"[ARGUS] Partial file preserved at {part_path.name}")
                raise

        return False
