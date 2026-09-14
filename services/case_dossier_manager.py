"""Case Dossier Manager for ARGUS AI.

Organizes missing persons and investigation cases into dedicated disk folders named:
    data/cases/{case_id}_{person_name}/

Inside each case folder:
    - case_details.json: Complete structured case dossier (ID, name, NIC, age, gender,
      case type, status, last seen location, GPS coordinates, biometrics summary, file inventory).
    - media/: Reference photos and reference videos associated with the case.
    - biometrics/: 256-dimensional ByGaitLight embeddings metadata, sequence metrics,
      and active gallery synchronization manifest.

Provides complete CRUD and synchronization between Firestore victims, local reference jobs,
EmbeddingDatabase, and disk storage.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from monitoring.logging_config import get_logger

logger = get_logger("argus.case_dossier_manager")


def _safe_atomic_write_json(file_path: Path, data: dict[str, Any], indent: int = 2) -> None:
    """Atomically writes JSON to file_path with a temporary swap to avoid corruption."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(f".tmp_{os.getpid()}_{time.time_ns()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    for attempt in range(6):
        try:
            tmp_path.replace(file_path)
            return
        except OSError:
            if attempt == 5:
                shutil.copyfile(str(tmp_path), str(file_path))
                tmp_path.unlink(missing_ok=True)
                return
            time.sleep(0.03 * (attempt + 1))


class CaseDossierManager:
    """Singleton manager for case folder organization and dossier artifacts."""

    _instance: CaseDossierManager | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        cases_dir: str | Path = "data/cases",
        jobs_dir: str | Path = "data/reference_jobs",
        videos_dir: str | Path = "data/reference_videos",
        photos_dir: str | Path = "data/reference_photos",
    ) -> None:
        self.cases_dir = Path(cases_dir)
        self.jobs_dir = Path(jobs_dir)
        self.videos_dir = Path(videos_dir)
        self.photos_dir = Path(photos_dir)

        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.photos_dir.mkdir(parents=True, exist_ok=True)

        self._cache_lock = threading.Lock()
        self._last_sync_time: float = 0.0

    @classmethod
    def get_instance(cls, cases_dir: str | Path = "data/cases") -> CaseDossierManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cases_dir=cases_dir)
            return cls._instance

    @staticmethod
    def sanitize_name(text: str) -> str:
        """Sanitize strings for filesystem directory and file names."""
        if not text:
            return "Unknown"
        # Replace non-alphanumeric (except dashes and underscores) with underscore
        cleaned = re.sub(r'[\\/*?:"<>| \t\n\r]+', "_", text.strip())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned or "Unknown"

    @classmethod
    def format_folder_name(cls, case_id: str, person_name: str) -> str:
        """Standardized naming convention: {case_id}_{person_name}."""
        safe_case = cls.sanitize_name(case_id or "Case")
        safe_name = cls.sanitize_name(person_name or "Unknown")
        return f"{safe_case}_{safe_name}"

    def find_case_folder(self, case_id: str) -> Path | None:
        """Locate existing folder for case_id (matches prefix `{case_id}_`)."""
        safe_case = self.sanitize_name(case_id)
        if not self.cases_dir.exists():
            return None

        # Exact match or prefix match
        for item in self.cases_dir.iterdir():
            if item.is_dir() and (item.name == safe_case or item.name.startswith(f"{safe_case}_")):
                return item
        return None

    def ensure_case_folder(self, case_id: str, person_name: str = "") -> Path:
        """Locates or creates dedicated folder `{case_id}_{person_name}` with media and biometrics dirs."""
        existing = self.find_case_folder(case_id)
        if existing:
            # If name is provided and different from current folder name, we can update or keep
            folder = existing
        else:
            folder_name = self.format_folder_name(case_id, person_name)
            folder = self.cases_dir / folder_name
            folder.mkdir(parents=True, exist_ok=True)

        # Ensure subdirectories
        (folder / "media").mkdir(exist_ok=True)
        (folder / "biometrics").mkdir(exist_ok=True)
        return folder

    def get_dossier_file(self, case_id: str, subpath: str) -> Path | None:
        """Safely returns absolute path to a file inside the case dossier, preventing path traversal."""
        folder = self.find_case_folder(case_id)
        if not folder or not folder.exists():
            return None

        target = (folder / subpath).resolve()
        folder_resolved = folder.resolve()

        try:
            target.relative_to(folder_resolved)
        except ValueError:
            logger.warning(f"Path traversal attempt rejected: {subpath} for case {case_id}")
            return None

        if target.exists() and target.is_file():
            return target
        return None

    def create_or_update_dossier(
        self,
        case_data: dict[str, Any],
        media_paths: list[str | Path] | None = None,
        biometrics_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Creates or updates the on-disk dossier folder and details for a case."""
        case_id = str(case_data.get("caseId") or case_data.get("case_id") or "Case_Unknown").strip()
        person_name = str(case_data.get("name") or case_data.get("person_name") or "Unknown").strip()

        folder = self.ensure_case_folder(case_id, person_name)
        media_dir = folder / "media"
        biometrics_dir = folder / "biometrics"

        # Copy any incoming media files
        if media_paths:
            for p in media_paths:
                src = Path(p)
                if src.exists() and src.is_file():
                    dest = media_dir / src.name
                    if not dest.exists():
                        try:
                            shutil.copy2(str(src), str(dest))
                        except (OSError, shutil.Error) as e:
                            logger.warning(f"Could not copy media {src} to {dest}: {e}")

        # Update biometrics manifest if provided
        if biometrics_data:
            bio_manifest_path = biometrics_dir / "biometrics_manifest.json"
            _safe_atomic_write_json(bio_manifest_path, biometrics_data)

        # Compile and save case_details.json
        dossier = self._build_dossier_dict(folder, case_data, biometrics_data)
        details_path = folder / "case_details.json"
        _safe_atomic_write_json(details_path, dossier)

        return dossier

    def _build_dossier_dict(
        self,
        folder: Path,
        raw_case_data: dict[str, Any] | None = None,
        biometrics_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Scans folder and merges raw case data into complete dossier representation."""
        details_path = folder / "case_details.json"
        existing_data: dict[str, Any] = {}
        if details_path.exists():
            try:
                with open(details_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"Could not read existing case details from {details_path}: {e}")

        data = dict(existing_data)
        if raw_case_data:
            data.update(raw_case_data)

        case_id = str(data.get("caseId") or data.get("case_id") or folder.name.split("_")[0])
        name = str(data.get("name") or data.get("person_name") or "Unknown")
        if name == "Unknown" and "_" in folder.name:
            # Infer from folder name
            parts = folder.name.split("_", 1)
            if len(parts) > 1:
                name = parts[1].replace("_", " ")

        # Scan folder files
        files_list = []
        total_bytes = 0
        media_photos = []
        media_videos = []

        for item in folder.rglob("*"):
            if item.is_file() and not item.name.startswith(".tmp"):
                rel_path = str(item.relative_to(folder)).replace("\\", "/")
                size = item.stat().st_size
                mtime = item.stat().st_mtime
                mime, _ = mimetypes.guess_type(item.name)
                mime = mime or "application/octet-stream"
                total_bytes += size

                file_info = {
                    "name": item.name,
                    "rel_path": rel_path,
                    "size_bytes": size,
                    "mime_type": mime,
                    "modified_at": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                    "download_url": f"/api/v1/cases/dossiers/{case_id}/files/{rel_path}",
                }
                files_list.append(file_info)

                if rel_path.startswith("media/"):
                    if mime.startswith("video/"):
                        media_videos.append(file_info)
                    elif mime.startswith("image/"):
                        media_photos.append(file_info)

        # Ensure case_details.json is represented in files_list if not yet on disk during first build
        has_case_details = any(f["rel_path"] == "case_details.json" for f in files_list)
        if not has_case_details and details_path.exists():
            size = details_path.stat().st_size
            mtime = details_path.stat().st_mtime
            files_list.append(
                {
                    "name": "case_details.json",
                    "rel_path": "case_details.json",
                    "size_bytes": size,
                    "mime_type": "application/json",
                    "modified_at": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                    "download_url": f"/api/v1/cases/dossiers/{case_id}/files/case_details.json",
                }
            )
            total_bytes += size

        # Check biometrics manifest
        bio_manifest_path = folder / "biometrics" / "biometrics_manifest.json"
        biometrics_info = biometrics_override or {}
        if not biometrics_info and bio_manifest_path.exists():
            try:
                with open(bio_manifest_path, "r", encoding="utf-8") as f:
                    biometrics_info = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"Could not read biometrics manifest from {bio_manifest_path}: {e}")

        # Parse location cleanly
        loc = data.get("lastSeenLocation") or {}
        if not loc or not isinstance(loc, dict):
            lat_str = str(data.get("latitude", "")).strip()
            lng_str = str(data.get("longitude", "")).strip()
            lat_val = float(lat_str) if lat_str else None
            lng_val = float(lng_str) if lng_str else None
            loc = {
                "name": data.get("locationName", "") or "Not Specified",
                "lat": lat_val,
                "lng": lng_val,
                "source": "Manual Entry / GPS",
            }

        reported_time = (
            data.get("reportedAt")
            or data.get("createdAt")
            or data.get("reported_at")
            or datetime.now(timezone.utc).isoformat()
        )

        dossier: dict[str, Any] = {
            "case_id": case_id,
            "person_name": name,
            "folder_name": folder.name,
            "folder_path": str(folder).replace("\\", "/"),
            "case_type": data.get("caseType") or data.get("case_type") or "Missing",
            "status": data.get("status") or "Investigating",
            "priority": data.get("priority") or "HIGH",
            "age": data.get("age", ""),
            "gender": data.get("gender", ""),
            "nic": data.get("nic", ""),
            "phone": data.get("contactNumber") or data.get("phone") or "",
            "description": data.get("description", ""),
            "location": loc,
            "reported_at": str(reported_time),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "biometrics": {
                "enrolled": bool(
                    biometrics_info.get("enrolled", False) or biometrics_info.get("gait_embeddings", 0) > 0
                ),
                "model": "ByGaitLight-256D + OSNet-512D",
                "gait_embeddings_count": biometrics_info.get("gait_embeddings_count")
                or biometrics_info.get("gait_embeddings", 0),
                "appearance_embeddings_count": biometrics_info.get("appearance_embeddings_count")
                or biometrics_info.get("appearance_embeddings", 0),
                "gallery_status": biometrics_info.get("gallery_status")
                or biometrics_info.get("status", "NOT_ENROLLED"),
                "job_id": biometrics_info.get("job_id", ""),
                "active_surveillance": biometrics_info.get("active_surveillance", False),
            },
            "media": {
                "photos_count": len(media_photos),
                "videos_count": len(media_videos),
                "photos": media_photos,
                "videos": media_videos,
            },
            "files_inventory": {
                "total_files": len(files_list),
                "total_size_bytes": total_bytes,
                "files": sorted(files_list, key=lambda x: x["rel_path"]),
            },
            "raw_metadata": {k: v for k, v in data.items() if k not in ("files_inventory", "media")},
        }

        return dossier

    def scan_and_sync_all_dossiers(self) -> list[dict[str, Any]]:
        """Synchronizes Firestore cases, reference jobs, embedding DB, and on-disk files.

        Returns all registered case dossiers.
        """
        with self._cache_lock:
            # 1. Fetch cases from Firestore if available
            firestore_cases: dict[str, dict[str, Any]] = {}
            try:
                from security_layer.auth import get_operator_store

                store = get_operator_store()
                client = store._get_firestore_client()
                if client:
                    docs = client.collection("victims").stream()
                    for d in docs:
                        c_dict = d.to_dict()
                        c_dict["caseId"] = d.id
                        firestore_cases[d.id] = c_dict
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not fetch cases from Firestore during sync: {e}")

            # 2. Fetch Embedding Database status
            embedding_stats: dict[str, dict[str, Any]] = {}
            try:
                from storage.embedding_database import EmbeddingDatabase

                emb_db = EmbeddingDatabase()
                for p in emb_db.list_all_persons():
                    embedding_stats[p.person_id] = {
                        "gait_embeddings": len(p.gait_embeddings),
                        "appearance_embeddings": len(p.appearance_embeddings),
                        "status": p.status,
                        "active": p.status == "ACTIVE",
                    }
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not read EmbeddingDatabase during sync: {e}")

            # 3. Fetch latest jobs for reference videos
            jobs_by_case: dict[str, dict[str, Any]] = {}
            if self.jobs_dir.exists():
                for j_file in self.jobs_dir.glob("*.json"):
                    try:
                        with open(j_file, "r", encoding="utf-8") as f:
                            j_data = json.load(f)
                            c_id = j_data.get("case_id") or j_data.get("person_id")
                            if c_id and (
                                c_id not in jobs_by_case
                                or j_data.get("updated_at", 0) > jobs_by_case[c_id].get("updated_at", 0)
                            ):
                                jobs_by_case[c_id] = j_data
                    except (json.JSONDecodeError, OSError) as e:
                        logger.debug(f"Could not load job manifest {j_file}: {e}")

            # 4. Synchronize all Firestore cases into disk folders
            for case_id, case_info in firestore_cases.items():
                name = case_info.get("name") or case_info.get("person_name") or "Unknown"
                folder = self.ensure_case_folder(case_id, name)
                media_dir = folder / "media"
                biometrics_dir = folder / "biometrics"

                # Check if there is media in data/reference_videos matching case_id
                if self.videos_dir.exists():
                    for v in self.videos_dir.iterdir():
                        if v.is_file() and case_id in v.name:
                            dest = media_dir / v.name
                            if not dest.exists():
                                try:
                                    shutil.copy2(str(v), str(dest))
                                except (OSError, shutil.Error) as e:
                                    logger.warning(f"Failed to copy video {v} to case media: {e}")

                # Check if there is media in data/reference_photos matching case_id
                if self.photos_dir.exists():
                    for p in self.photos_dir.iterdir():
                        if p.is_file() and case_id in p.name:
                            dest = media_dir / p.name
                            if not dest.exists():
                                try:
                                    shutil.copy2(str(p), str(dest))
                                except (OSError, shutil.Error) as e:
                                    logger.warning(f"Failed to copy photo {p} to case media: {e}")

                # Build biometrics data
                job_rec = jobs_by_case.get(case_id)
                emb_rec = embedding_stats.get(case_id, {})
                gait_count = emb_rec.get("gait_embeddings", 0)
                if not gait_count and job_rec:
                    gait_count = job_rec.get("progress", {}).get("embeddings_committed", 0)

                bio_manifest: dict[str, Any] = {
                    "enrolled": gait_count > 0 or emb_rec.get("active", False),
                    "gait_embeddings_count": gait_count,
                    "appearance_embeddings_count": emb_rec.get("appearance_embeddings", 0),
                    "gallery_status": emb_rec.get("status", "ACTIVE" if gait_count > 0 else "PENDING"),
                    "active_surveillance": emb_rec.get("active", False) or gait_count > 0,
                    "job_id": job_rec.get("job_id", "") if job_rec else "",
                    "job_status": job_rec.get("status", "") if job_rec else "",
                    "valid_sequences": job_rec.get("progress", {}).get("valid_sequences", 0) if job_rec else 0,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }
                _safe_atomic_write_json(biometrics_dir / "biometrics_manifest.json", bio_manifest)

                # Write case_details.json
                dossier_data = self._build_dossier_dict(folder, case_info, bio_manifest)
                _safe_atomic_write_json(folder / "case_details.json", dossier_data)

            # 5. Scan all directories in data/cases/ to include any offline/standalone cases
            all_dossiers: list[dict[str, Any]] = []
            if self.cases_dir.exists():
                for case_folder in self.cases_dir.iterdir():
                    if case_folder.is_dir():
                        details_file = case_folder / "case_details.json"
                        dossier = None
                        if details_file.exists():
                            try:
                                with open(details_file, "r", encoding="utf-8") as f:
                                    dossier = json.load(f)
                            except (json.JSONDecodeError, OSError) as e:
                                logger.debug(f"Could not read dossier details from {details_file}: {e}")

                        if not dossier:
                            # Build it from scratch
                            dossier = self._build_dossier_dict(case_folder)
                            _safe_atomic_write_json(details_file, dossier)

                        all_dossiers.append(dossier)

            # Sort dossiers by reported_at descending
            all_dossiers.sort(key=lambda d: str(d.get("reported_at", "")), reverse=True)
            self._last_sync_time = time.time()
            return all_dossiers

    def get_dossier(self, case_id: str) -> dict[str, Any] | None:
        """Retrieves a single case dossier by case_id."""
        folder = self.find_case_folder(case_id)
        if not folder:
            # Trigger sync to be certain
            dossiers = self.scan_and_sync_all_dossiers()
            for d in dossiers:
                if d.get("case_id") == case_id:
                    return d
            return None

        details_file = folder / "case_details.json"
        if details_file.exists():
            try:
                with open(details_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"Could not read dossier details from {details_file}: {e}")
        return self._build_dossier_dict(folder)
