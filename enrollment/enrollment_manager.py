from pathlib import Path

from core.logger import setup_logger
from enrollment.appearance_gallery_updater import AppearanceGalleryUpdater
from enrollment.enrollment_validator import EnrollmentValidator
from enrollment.gallery_updater import GalleryUpdater
from pipeline.steps.feature_extraction import FeatureExtractionStep
from pipeline.steps.reid_feature_extraction import ReIDFeatureExtractionStep


class EnrollmentManager:
    def __init__(
        self,
    ) -> None:
        self.logger = setup_logger("ARGUS.Enrollment")
        self.validator = EnrollmentValidator()
        self.gait_extractor = FeatureExtractionStep()
        self.appearance_extractor = ReIDFeatureExtractionStep()
        self.gallery_updater = GalleryUpdater()
        self.appearance_gallery_updater = AppearanceGalleryUpdater()
        try:
            from storage.embedding_database import EmbeddingDatabase

            self.embedding_db = EmbeddingDatabase()
        except (ImportError, RuntimeError, ValueError, TypeError, OSError):
            self.embedding_db = None

    def _collect_images(
        self,
        folder: Path,
    ) -> list[Path]:
        image_paths = []

        for ext in (
            "*.png",
            "*.jpg",
            "*.jpeg",
        ):
            image_paths.extend(
                folder.glob(
                    ext,
                )
            )

        return sorted(
            image_paths,
        )

    def _collect_videos(
        self,
        folder: Path,
    ) -> list[Path]:
        video_paths = []

        for ext in (
            "*.mp4",
            "*.avi",
            "*.mov",
        ):
            video_paths.extend(
                folder.glob(
                    ext,
                )
            )

        return sorted(
            video_paths,
        )

    def enroll_gait_person(
        self,
        person_folder: str,
    ) -> dict:
        folder = Path(
            person_folder,
        )

        person_id = folder.name

        valid, message = self.validator.validate_person_folder(
            str(folder),
        )

        if not valid:
            self.logger.error(f"Gait enrollment failed for {person_id}: {message}")

            return {
                "success": False,
                "person_id": person_id,
                "message": message,
                "embeddings_added": 0,
                "gallery": "gait",
            }

        image_paths = self._collect_images(
            folder,
        )

        embeddings = []

        for image_path in image_paths:
            try:
                embedding = self.gait_extractor.extract(
                    image_path,
                )

                embeddings.append(
                    embedding,
                )

            except (RuntimeError, ValueError, TypeError, OSError) as error:
                self.logger.warning(f"Failed to extract gait embedding from {image_path.name}: {error}")

        if not embeddings:
            return {
                "success": False,
                "person_id": person_id,
                "message": "No valid gait embeddings generated",
                "embeddings_added": 0,
                "gallery": "gait",
            }

        self.gallery_updater.add_person(
            person_id=person_id,
            embeddings=embeddings,
        )

        if self.embedding_db is not None:
            try:
                self.embedding_db.add_embeddings(
                    person_id=person_id,
                    gait_embeddings=embeddings,
                )
            except (RuntimeError, ValueError, TypeError, OSError, AttributeError) as db_err:
                self.logger.warning(f"EmbeddingDatabase sync warning for {person_id}: {db_err}")

        self.logger.info(f"Gait enrollment completed for {person_id}. Embeddings added: {len(embeddings)}")

        return {
            "success": True,
            "person_id": person_id,
            "message": "Gait enrollment completed",
            "embeddings_added": len(embeddings),
            "gallery": "gait",
        }

    def enroll_appearance_person(
        self,
        person_folder: str,
    ) -> dict:
        folder = Path(
            person_folder,
        )

        person_id = folder.name

        image_paths = self._collect_images(
            folder,
        )

        if not image_paths:
            return {
                "success": False,
                "person_id": person_id,
                "message": "No photo files found",
                "embeddings_added": 0,
                "gallery": "appearance",
            }

        embeddings = []

        for image_path in image_paths:
            try:
                embedding = self.appearance_extractor.extract(
                    image_path,
                )
                if embedding is not None:
                    embeddings.append(
                        embedding,
                    )

            except (RuntimeError, ValueError, TypeError, OSError) as error:
                self.logger.warning(f"Failed to extract appearance embedding from {image_path.name}: {error}")

        if not embeddings:
            return {
                "success": False,
                "person_id": person_id,
                "message": "No valid appearance embeddings generated",
                "embeddings_added": 0,
                "gallery": "appearance",
            }

        self.appearance_gallery_updater.add_person(
            person_id=person_id,
            embeddings=embeddings,
        )

        if self.embedding_db is not None:
            try:
                self.embedding_db.add_embeddings(
                    person_id=person_id,
                    appearance_embeddings=embeddings,
                )
            except (RuntimeError, ValueError, TypeError, OSError, AttributeError) as db_err:
                self.logger.warning(f"EmbeddingDatabase sync warning for {person_id}: {db_err}")

        self.logger.info(f"Appearance enrollment completed for {person_id}. Embeddings added: {len(embeddings)}")

        return {
            "success": True,
            "person_id": person_id,
            "message": "Appearance enrollment completed",
            "embeddings_added": len(embeddings),
            "gallery": "appearance",
        }

    def enroll_person(
        self,
        person_folder: str,
    ) -> dict:
        folder = Path(
            person_folder,
        )

        person_id = folder.name

        videos = self._collect_videos(
            folder,
        )

        images = self._collect_images(
            folder,
        )

        if videos:
            return {
                "success": False,
                "person_id": person_id,
                "message": "Gait enrollment requires video processing. Use AutoEnrollmentService to generate gait GEIs from video.",
                "embeddings_added": 0,
                "gallery": "none",
            }

        if images:
            return self.enroll_appearance_person(person_folder)

        return {
            "success": False,
            "person_id": person_id,
            "message": "No supported files found",
            "embeddings_added": 0,
            "gallery": "none",
        }
