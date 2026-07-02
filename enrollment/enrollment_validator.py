from pathlib import Path

import cv2


class EnrollmentValidator:
    def __init__(
        self,
        min_images: int = 5,
        min_resolution: tuple[int, int] = (64, 64),
    ) -> None:
        self.min_images = min_images
        self.min_resolution = min_resolution

    def validate_person_folder(
        self,
        person_folder: str,
    ) -> tuple[bool, str]:

        folder = Path(person_folder)

        if not folder.exists():
            return False, "Folder not found"

        images = []

        for ext in ("*.png", "*.jpg", "*.jpeg"):
            images.extend(folder.glob(ext))

        if len(images) < self.min_images:
            return (
                False,
                f"Need at least {self.min_images} images",
            )

        for image_path in images:

            image = cv2.imread(str(image_path))

            if image is None:
                return (
                    False,
                    f"Invalid image: {image_path.name}",
                )

            h, w = image.shape[:2]

            if (
                h < self.min_resolution[0]
                or
                w < self.min_resolution[1]
            ):
                return (
                    False,
                    f"Low resolution: {image_path.name}",
                )

        return True, "Validation passed"