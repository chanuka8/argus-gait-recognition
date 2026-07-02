from pathlib import Path

import cv2


def ensure_directory(path: str) -> None:
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def save_image(path: str, image) -> bool:
    ensure_directory(
        str(Path(path).parent)
    )

    return cv2.imwrite(path, image)