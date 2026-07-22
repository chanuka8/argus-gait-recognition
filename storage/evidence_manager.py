"""Evidence manager for organized snapshot, GEI, and metadata storage with retention policy."""

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import cv2
import numpy as np

from monitoring.logging_config import get_logger


class EvidenceManager:
    """Manages creation, organized folder structure, and retention policy for evidence files."""

    def __init__(self, base_evidence_dir: str = "outputs/evidence", max_age_days: int = 30) -> None:
        self.base_dir = Path(base_evidence_dir)
        self.max_age_seconds = max_age_days * 86400.0
        self._logger = get_logger("evidence_manager")
        self._lock = Lock()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_evidence(
        self,
        target_id: str,
        camera_id: str,
        confidence: float,
        frame: Optional[np.ndarray] = None,
        gei: Optional[np.ndarray] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Save evidence snapshot, GEI, and metadata into organized target directory."""
        now_str = time.strftime("%Y%m%d_%H%M%S")
        timestamp = time.monotonic()
        folder_name = f"{now_str}_{camera_id}"
        target_dir = self.base_dir / target_id / folder_name

        count = 1
        while target_dir.exists():
            target_dir = self.base_dir / target_id / f"{folder_name}_{count}"
            count += 1

        saved_files: Dict[str, str] = {}

        with self._lock:
            target_dir.mkdir(parents=True, exist_ok=True)

            if frame is not None and frame.size > 0:
                frame_path = target_dir / "snapshot.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_files["snapshot"] = str(frame_path)

            if gei is not None and gei.size > 0:
                gei_path = target_dir / "gei.png"
                cv2.imwrite(str(gei_path), gei)
                saved_files["gei"] = str(gei_path)

            metadata = {
                "target_id": target_id,
                "camera_id": camera_id,
                "confidence": confidence,
                "timestamp": timestamp,
                "datetime": now_str,
                "files": saved_files,
                "extra": extra_metadata or {},
            }

            meta_path = target_dir / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            saved_files["metadata"] = str(meta_path)

            self._logger.info(f"Saved evidence for {target_id} in {target_dir}")

        return saved_files

    def enforce_retention_policy(self) -> int:
        """Purge evidence folders older than retention policy max age."""
        now = time.monotonic()
        deleted = 0

        with self._lock:
            if not self.base_dir.exists():
                return 0

            meta_files = list(self.base_dir.rglob("metadata.json"))
            for meta_file in meta_files:
                try:
                    if not meta_file.exists():
                        continue
                    with open(meta_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    ts = data.get("timestamp", 0.0)
                    if (now - ts) > self.max_age_seconds:
                        folder = meta_file.parent
                        if folder.exists():
                            for child in list(folder.iterdir()):
                                child.unlink(missing_ok=True)
                            folder.rmdir()
                            deleted += 1
                except Exception as e:
                    self._logger.error(f"Error purging evidence {meta_file}: {e}")

        return deleted
