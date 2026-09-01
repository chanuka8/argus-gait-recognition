import json
import shutil
import time
from pathlib import Path

import cv2

from core.logger import setup_logger
from enrollment.enrollment_lifecycle import EnrollmentLifecycleManager, EnrollmentStatus
from enrollment.enrollment_manager import EnrollmentManager
from pipeline.steps.live_gei import LiveGEI
from pipeline.steps.silhouette_step import SilhouetteStep
from pipeline.steps.tracking import TrackingStep
from preprocessing.video_quality_gate import DeterministicVideoQualityGate
from storage.embedding_database import EmbeddingDatabase
from storage.vector_store import VectorStore


class AutoEnrollmentService:
    def __init__(
        self,
        input_dir: str = "data/new_input",
        processed_dir: str = "data/auto_enrollment/gei",
        photo_processed_dir: str = "data/auto_enrollment/photos",
        marker_name: str = ".argus_enrolled.json",
        gei_frames: int = 15,
        video_stride: int = 10,
        scan_interval: int = 5,
        live_gallery_dir: str = "models/live_gallery",
        appearance_gallery_dir: str = "models/appearance_gallery",
        db_dir: str = "data/embedding_db",
        auto_delete_raw: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.processed_dir = Path(processed_dir)
        self.photo_processed_dir = Path(photo_processed_dir)
        self.marker_name = marker_name
        self.gei_frames = gei_frames
        self.video_stride = video_stride
        self.scan_interval = scan_interval
        self.auto_delete_raw = auto_delete_raw

        self.logger = setup_logger("ARGUS.AutoEnrollment")
        self.enrollment_manager = EnrollmentManager()
        self.tracker = TrackingStep()
        self.silhouette_step = SilhouetteStep()
        self.quality_gate = DeterministicVideoQualityGate()

        self.embedding_db = EmbeddingDatabase(
            db_dir=db_dir,
            gait_gallery_dir=live_gallery_dir,
            appearance_gallery_dir=appearance_gallery_dir,
        )
        self.lifecycle_manager = EnrollmentLifecycleManager(db=self.embedding_db)

        self.live_store = VectorStore(gallery_dir=live_gallery_dir)
        self.appearance_store = VectorStore(gallery_dir=appearance_gallery_dir)

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.photo_processed_dir.mkdir(parents=True, exist_ok=True)

    def _person_folders(self) -> list[Path]:
        return sorted(
            folder
            for folder in self.input_dir.iterdir()
            if folder.is_dir() and not folder.name.startswith("_") and not folder.name.startswith(".")
        )

    def _input_files(self, person_folder: Path) -> list[Path]:
        files: list[Path] = []
        for pattern in (
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.mp4",
            "*.avi",
            "*.mov",
        ):
            files.extend(person_folder.glob(pattern))
        return sorted(files)

    def _image_files(self, files: list[Path]) -> list[Path]:
        return [file for file in files if file.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    def _video_files(self, files: list[Path]) -> list[Path]:
        return [file for file in files if file.suffix.lower() in {".mp4", ".avi", ".mov"}]

    def _fingerprint(self, files: list[Path]) -> dict:
        return {
            str(path.name): {
                "size": path.stat().st_size,
                "modified": path.stat().st_mtime,
            }
            for path in files
        }

    def _marker_path(self, person_folder: Path) -> Path:
        return person_folder / self.marker_name

    def _load_marker(self, person_folder: Path) -> dict | None:
        marker = self._marker_path(person_folder)
        if not marker.exists():
            return None
        with open(marker, "r", encoding="utf-8") as file:
            return json.load(file)

    def _save_marker(self, person_folder: Path, fingerprint: dict, result: dict) -> None:
        if not person_folder.exists():
            return
        marker = self._marker_path(person_folder)
        payload = {
            "person_id": person_folder.name,
            "fingerprint": fingerprint,
            "result": result,
            "status": "enrolled" if result.get("success") else "failed",
            "updated_at": time.time(),
        }
        with open(marker, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)

    def _target_folder(self, root: Path, person_id: str) -> Path:
        folder = root / person_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _clear_target_folder(self, target_folder: Path) -> None:
        if target_folder.exists():
            shutil.rmtree(target_folder)
        target_folder.mkdir(parents=True, exist_ok=True)

    def _copy_image(self, image_path: Path, target_folder: Path) -> int:
        target = target_folder / image_path.name
        shutil.copy2(image_path, target)
        return 1

    def _crop_person(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _process_video(self, video_path: Path, target_folder: Path) -> list[Path]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.warning(f"Unable to open video: {video_path}")
            return []

        buffers: dict[int, LiveGEI] = {}
        crops_per_track: dict[int, list] = {}
        silhouettes_per_track: dict[int, list] = {}
        saved_paths: list[Path] = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            detections = self.tracker.track(frame)
            xyxy = detections.xyxy
            tracker_ids = detections.tracker_id

            if tracker_ids is None:
                continue

            for box, track_id in zip(xyxy, tracker_ids):
                track_id = int(track_id)
                crop = self._crop_person(frame, box)
                if crop is None:
                    continue

                silhouette = self.silhouette_step.extract_from_crop(crop)
                if silhouette is None:
                    continue

                if track_id not in buffers:
                    buffers[track_id] = LiveGEI(max_frames=self.gei_frames)
                    crops_per_track[track_id] = []
                    silhouettes_per_track[track_id] = []

                crops_per_track[track_id].append(crop)
                silhouettes_per_track[track_id].append(silhouette)
                buffers[track_id].add(silhouette)

                if not buffers[track_id].ready():
                    continue

                if frame_index % self.video_stride != 0:
                    continue

                q_res = self.quality_gate.assess_video_clip(
                    crops_per_track[track_id][-self.gei_frames :],
                    silhouettes_per_track[track_id][-self.gei_frames :],
                )

                if not q_res.passed:
                    if q_res.salvageable:
                        enhanced_crops = [
                            self.quality_gate.enhance_crop_deterministic(c)
                            for c in crops_per_track[track_id][-self.gei_frames :]
                        ]
                        enhanced_silhouettes = [
                            self.silhouette_step.extract_from_crop(ec) for ec in enhanced_crops if ec is not None
                        ]
                        valid_silhouettes = [s for s in enhanced_silhouettes if s is not None]
                        if len(valid_silhouettes) >= self.gei_frames:
                            rebuilt_gei = LiveGEI(max_frames=self.gei_frames)
                            for vs in valid_silhouettes:
                                rebuilt_gei.add(vs)
                            gei = rebuilt_gei.build()
                        else:
                            continue
                    else:
                        continue
                else:
                    gei = buffers[track_id].build()

                if gei is None:
                    continue

                output_name = f"{video_path.stem}_track{track_id}_frame{frame_index}.png"
                output_path = target_folder / output_name
                cv2.imwrite(str(output_path), gei)
                saved_paths.append(output_path)

        cap.release()
        return saved_paths

    def _prepare_photo_folder(self, person_id: str, image_files: list[Path]) -> tuple[Path, int]:
        target_folder = self._target_folder(self.photo_processed_dir, person_id)
        self._clear_target_folder(target_folder)
        prepared = 0
        for image_path in image_files:
            prepared += self._copy_image(image_path, target_folder)
        return target_folder, prepared

    def enroll_pending(self, force: bool = False) -> list[dict]:
        results: list[dict] = []

        for person_folder in self._person_folders():
            files = self._input_files(person_folder)
            if not files:
                continue

            image_files = self._image_files(files)
            video_files = self._video_files(files)
            fingerprint = self._fingerprint(files)
            person_id = person_folder.name

            self.logger.info(
                f"Auto enrollment processing '{person_id}': {len(video_files)} videos, {len(image_files)} photos"
            )

            generated_gei_paths: list[Path] = []
            if video_files:
                gait_target = self._target_folder(self.processed_dir, person_id)
                self._clear_target_folder(gait_target)
                for v_path in video_files:
                    geis = self._process_video(v_path, gait_target)
                    generated_gei_paths.extend(geis)


            job_result = self.lifecycle_manager.enroll_from_media(
                person_id=person_id,
                video_paths=video_files,
                photo_paths=image_files,
                gei_paths=generated_gei_paths,
                auto_delete_raw=self.auto_delete_raw,
            )

            res_dict = {
                "success": job_result.status == EnrollmentStatus.EMBEDDING_ONLY,
                "person_id": person_id,
                "status": job_result.status.value,
                "gait_embeddings_added": job_result.gait_embeddings_count,
                "appearance_embeddings_added": job_result.appearance_embeddings_count,
                "raw_files_deleted": job_result.raw_files_deleted,
                "raw_files_retained": job_result.raw_files_retained,
                "error_message": job_result.error_message,
            }

            if person_folder.exists():
                self._save_marker(person_folder, fingerprint, res_dict)

            results.append(res_dict)

            if res_dict["success"]:
                self.logger.info(
                    f"Enrollment completed for '{person_id}'. Raw files cleaned up. State: EMBEDDING_ONLY."
                )
            else:
                self.logger.warning(
                    f"Enrollment failed for '{person_id}'. Raw files retained. State: {res_dict['status']}."
                )

        return results

    def watch(self) -> None:
        print("\n=== ARGUS AUTO ENROLLMENT WATCHER ===")
        print(f"Watching: {self.input_dir}")
        print("Press CTRL + C to stop\n")

        while True:
            try:
                self.enroll_pending()
                time.sleep(self.scan_interval)
            except KeyboardInterrupt:
                print("\nAuto enrollment watcher stopped.")
                break
