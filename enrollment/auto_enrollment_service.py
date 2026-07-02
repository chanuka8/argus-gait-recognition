import json
import shutil
import time
from pathlib import Path

import cv2

from core.logger import setup_logger
from enrollment.enrollment_manager import EnrollmentManager
from pipeline.steps.live_gei import LiveGEI
from pipeline.steps.silhouette_step import SilhouetteStep
from pipeline.steps.tracking import TrackingStep
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
    ) -> None:
        self.input_dir = Path(input_dir)
        self.processed_dir = Path(processed_dir)
        self.photo_processed_dir = Path(photo_processed_dir)
        self.marker_name = marker_name
        self.gei_frames = gei_frames
        self.video_stride = video_stride
        self.scan_interval = scan_interval

        self.logger = setup_logger("ARGUS.AutoEnrollment")
        self.enrollment_manager = EnrollmentManager()
        self.tracker = TrackingStep()
        self.silhouette_step = SilhouetteStep()

        self.live_store = VectorStore(
            gallery_dir=live_gallery_dir,
        )

        self.appearance_store = VectorStore(
            gallery_dir=appearance_gallery_dir,
        )

        self.input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.photo_processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _person_folders(
        self,
    ) -> list[Path]:
        return sorted(
            folder
            for folder in self.input_dir.iterdir()
            if folder.is_dir()
            and not folder.name.startswith("_")
            and not folder.name.startswith(".")
        )

    def _input_files(
        self,
        person_folder: Path,
    ) -> list[Path]:
        files: list[Path] = []

        for pattern in (
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.mp4",
            "*.avi",
            "*.mov",
        ):
            files.extend(
                person_folder.glob(pattern)
            )

        return sorted(
            files,
        )

    def _image_files(
        self,
        files: list[Path],
    ) -> list[Path]:
        return [
            file
            for file in files
            if file.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
            }
        ]

    def _video_files(
        self,
        files: list[Path],
    ) -> list[Path]:
        return [
            file
            for file in files
            if file.suffix.lower()
            in {
                ".mp4",
                ".avi",
                ".mov",
            }
        ]

    def _fingerprint(
        self,
        files: list[Path],
    ) -> dict:
        return {
            str(path.name): {
                "size": path.stat().st_size,
                "modified": path.stat().st_mtime,
            }
            for path in files
        }

    def _marker_path(
        self,
        person_folder: Path,
    ) -> Path:
        return person_folder / self.marker_name

    def _load_marker(
        self,
        person_folder: Path,
    ) -> dict | None:
        marker = self._marker_path(
            person_folder,
        )

        if not marker.exists():
            return None

        with open(
            marker,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(
                file,
            )

    def _save_marker(
        self,
        person_folder: Path,
        fingerprint: dict,
        result: dict,
    ) -> None:
        marker = self._marker_path(
            person_folder,
        )

        payload = {
            "person_id": person_folder.name,
            "fingerprint": fingerprint,
            "result": result,
            "status": "enrolled" if result.get("success") else "failed",
            "updated_at": time.time(),
        }

        with open(
            marker,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=4,
            )

    def _metadata(
        self,
        store: VectorStore,
    ) -> dict:
        current = store.load()

        if current is None:
            return {}

        _, _, metadata = current

        return metadata

    def _already_enrolled(
        self,
        person_id: str,
        has_videos: bool,
        has_photos: bool,
    ) -> bool:
        gait_metadata = self._metadata(
            self.live_store,
        )

        appearance_metadata = self._metadata(
            self.appearance_store,
        )

        gait_done = (
            not has_videos
            or person_id in gait_metadata
        )

        appearance_done = (
            not has_photos
            or person_id in appearance_metadata
        )

        return gait_done and appearance_done

    def _needs_enrollment(
        self,
        person_folder: Path,
        fingerprint: dict,
        has_videos: bool,
        has_photos: bool,
        force: bool = False,
    ) -> bool:
        if force:
            return True

        person_id = person_folder.name

        if self._already_enrolled(
            person_id,
            has_videos,
            has_photos,
        ):
            return False

        marker = self._load_marker(
            person_folder,
        )

        if marker is None:
            return True

        if marker.get("status") != "enrolled":
            return True

        if marker.get("fingerprint") != fingerprint:
            return True

        return False

    def _target_folder(
        self,
        root: Path,
        person_id: str,
    ) -> Path:
        folder = root / person_id

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        return folder

    def _clear_target_folder(
        self,
        target_folder: Path,
    ) -> None:
        if target_folder.exists():
            shutil.rmtree(
                target_folder,
            )

        target_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _copy_image(
        self,
        image_path: Path,
        target_folder: Path,
    ) -> int:
        target = target_folder / image_path.name

        shutil.copy2(
            image_path,
            target,
        )

        return 1

    def _crop_person(
        self,
        frame,
        box,
    ):
        h, w = frame.shape[:2]

        x1, y1, x2, y2 = map(
            int,
            box,
        )

        x1 = max(
            0,
            x1,
        )
        y1 = max(
            0,
            y1,
        )
        x2 = min(
            w,
            x2,
        )
        y2 = min(
            h,
            y2,
        )

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def _process_video(
        self,
        video_path: Path,
        target_folder: Path,
    ) -> int:
        cap = cv2.VideoCapture(
            str(video_path),
        )

        if not cap.isOpened():
            self.logger.warning(
                f"Unable to open video: {video_path}"
            )

            return 0

        buffers: dict[int, LiveGEI] = {}
        saved = 0
        frame_index = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            detections = self.tracker.track(
                frame,
            )

            xyxy = detections.xyxy
            tracker_ids = detections.tracker_id

            if tracker_ids is None:
                continue

            for box, track_id in zip(
                xyxy,
                tracker_ids,
            ):
                track_id = int(
                    track_id,
                )

                crop = self._crop_person(
                    frame,
                    box,
                )

                if crop is None:
                    continue

                silhouette = self.silhouette_step.extract_from_crop(
                    crop,
                )

                if silhouette is None:
                    continue

                if track_id not in buffers:
                    buffers[track_id] = LiveGEI(
                        max_frames=self.gei_frames,
                    )

                buffers[track_id].add(
                    silhouette,
                )

                if not buffers[track_id].ready():
                    continue

                if frame_index % self.video_stride != 0:
                    continue

                gei = buffers[track_id].build()

                if gei is None:
                    continue

                output_name = (
                    f"{video_path.stem}_track{track_id}_"
                    f"frame{frame_index}.png"
                )

                output_path = target_folder / output_name

                cv2.imwrite(
                    str(output_path),
                    gei,
                )

                saved += 1

        cap.release()

        return saved

    def _prepare_photo_folder(
        self,
        person_id: str,
        image_files: list[Path],
    ) -> tuple[Path, int]:
        target_folder = self._target_folder(
            self.photo_processed_dir,
            person_id,
        )

        self._clear_target_folder(
            target_folder,
        )

        prepared = 0

        for image_path in image_files:
            prepared += self._copy_image(
                image_path,
                target_folder,
            )

        return target_folder, prepared

    def _prepare_gait_folder(
        self,
        person_id: str,
        video_files: list[Path],
    ) -> tuple[Path, int]:
        target_folder = self._target_folder(
            self.processed_dir,
            person_id,
        )

        self._clear_target_folder(
            target_folder,
        )

        prepared = 0

        for video_path in video_files:
            prepared += self._process_video(
                video_path,
                target_folder,
            )

        return target_folder, prepared

    def enroll_pending(
        self,
        force: bool = False,
    ) -> list[dict]:
        results: list[dict] = []

        for person_folder in self._person_folders():
            files = self._input_files(
                person_folder,
            )

            if not files:
                continue

            image_files = self._image_files(
                files,
            )

            video_files = self._video_files(
                files,
            )

            fingerprint = self._fingerprint(
                files,
            )

            person_id = person_folder.name

            # Skip photo-only folder
            if image_files and not video_files:
                print(
                    f"Skipped photo-only folder. Gait enrollment requires walking video: {person_id}"
                )
                continue

            if not video_files:
                continue

            if not self._needs_enrollment(
                person_folder=person_folder,
                fingerprint=fingerprint,
                has_videos=True,
                has_photos=False,
                force=force,
            ):
                print(
                    f"Skipped already enrolled identity: {person_id}"
                )

                continue

            print(
                f"\nAuto enrollment detected new gait data: {person_id}"
            )

            print(
                f"  Videos found: {len(video_files)}"
            )

            print(
                f"  Photos found: {len(image_files)}"
            )

            if image_files:
                print(
                    "  Photos ignored for gait-only enrollment."
                )

            result = {
                "success": False,
                "person_id": person_id,
                "message": "No usable enrollment data generated",
                "gait_embeddings_added": 0,
                "appearance_embeddings_added": 0,
            }

            gait_folder, gait_prepared = self._prepare_gait_folder(
                person_id,
                video_files,
            )

            if gait_prepared > 0:
                gait_result = self.enrollment_manager.enroll_gait_person(
                    str(gait_folder),
                )

                result["gait_result"] = gait_result
                result["gait_embeddings_added"] = gait_result.get(
                    "embeddings_added",
                    0,
                )

            total_embeddings = int(
                result.get(
                    "gait_embeddings_added",
                    0,
                )
            )

            result["success"] = total_embeddings > 0
            result["message"] = (
                "Gait enrollment completed"
                if result["success"]
                else "No valid embeddings generated"
            )

            self._save_marker(
                person_folder,
                fingerprint,
                result,
            )

            results.append(
                result,
            )

            if result["success"]:
                print(
                    f"  Gait enrollment completed: {person_id}"
                )
            else:
                print(
                    f"  Gait enrollment failed: {person_id} | "
                    f"No valid embeddings generated"
                )

        return results

    def watch(
        self,
    ) -> None:
        print("\n=== ARGUS AUTO ENROLLMENT WATCHER ===")
        print(f"Watching: {self.input_dir}")
        print("Press CTRL + C to stop\n")

        while True:
            try:
                self.enroll_pending()
                time.sleep(
                    self.scan_interval,
                )

            except KeyboardInterrupt:
                print("\nAuto enrollment watcher stopped.")
                break