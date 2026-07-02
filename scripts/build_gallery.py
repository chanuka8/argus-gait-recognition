import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.architectures.bygait_light import ByGaitLight
from storage.vector_store import VectorStore


MODEL_PATH = "runs/exp_001/best_model.pth"
GEI_ROOT = "data/casia_processed/gei"
GALLERY_DIR = "models/gallery"


def load_backbone() -> ByGaitLight:
    model = ByGaitLight()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    filtered = {}

    for key, value in checkpoint.items():
        if key.startswith("backbone."):
            filtered[key.replace("backbone.", "")] = value

    model.load_state_dict(
        filtered,
        strict=True,
    )

    model.eval()

    return model


def image_to_tensor(
    path: Path,
) -> torch.Tensor:
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Failed to read image: {path}"
        )

    image = image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(
        image,
    ).unsqueeze(0).unsqueeze(0)

    return tensor


def main() -> None:
    model = load_backbone()

    features = []
    labels = []
    metadata = {}

    gei_root = Path(GEI_ROOT)

    for person_dir in tqdm(
        sorted(gei_root.iterdir()),
        desc="Gallery",
    ):
        if not person_dir.is_dir():
            continue

        person_id = person_dir.name
        count = 0

        for image_path in sorted(person_dir.glob("*.png")):
            tensor = image_to_tensor(
                image_path,
            )

            with torch.no_grad():
                embedding = (
                    model(tensor)
                    .cpu()
                    .numpy()
                    .flatten()
                    .astype(np.float32)
                )

            features.append(embedding)
            labels.append(person_id)
            count += 1

        if count > 0:
            metadata[person_id] = {
                "embeddings": count,
                "status": "ACTIVE",
                "enabled": True,
                "source": "CASIA_B_GALLERY",
                "updated_at": time.time(),
            }

    store = VectorStore(
        gallery_dir=GALLERY_DIR,
    )

    store.save(
        features,
        labels,
        metadata,
    )

    print("\nGallery build complete.")
    print(f"Embeddings: {len(features)}")
    print(f"People: {len(metadata)}")
    print(f"Gallery directory: {GALLERY_DIR}")


if __name__ == "__main__":
    main()