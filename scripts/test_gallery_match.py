import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from models.architectures.bygait_light import ByGaitLight
from pipeline.steps.matching_step import MatchingStep
from storage.vector_store import VectorStore


MODEL_PATH = "runs/exp_001/best_model.pth"
GEI_ROOT = "data/casia_processed/gei"


def load_backbone():

    model = ByGaitLight()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    filtered = {}

    for key, value in checkpoint.items():

        if key.startswith("backbone."):

            filtered[
                key.replace(
                    "backbone.",
                    "",
                )
            ] = value

    model.load_state_dict(
        filtered,
        strict=True,
    )

    model.eval()

    return model


def image_to_embedding(
    image_path,
    model,
):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    image = image.astype(
        np.float32
    ) / 255.0

    tensor = (
        torch.from_numpy(image)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    with torch.no_grad():

        embedding = (
            model(tensor)
            .cpu()
            .numpy()
            .flatten()
        )

    return embedding


def pick_random_image():

    all_images = []

    for person_dir in Path(
        GEI_ROOT
    ).iterdir():

        if not person_dir.is_dir():
            continue

        all_images.extend(
            list(
                person_dir.glob("*.png")
            )
        )

    return random.choice(
        all_images
    )


def main():

    model = load_backbone()

    store = VectorStore()

    gallery = store.load()

    if gallery is None:

        print(
            "Gallery not found."
        )

        return

    gallery_features, gallery_labels, metadata = gallery

    image_path = pick_random_image()

    query_embedding = image_to_embedding(
        image_path,
        model,
    )

    matcher = MatchingStep(
        threshold=0.75
    )

    identity, score = matcher.match(
        query_embedding,
        gallery_features,
        gallery_labels,
    )

    actual_person = (
        image_path.parent.name
    )

    print("\n=== MATCH RESULT ===")
    print(
        f"Query Image : {image_path.name}"
    )
    print(
        f"Actual ID   : {actual_person}"
    )
    print(
        f"Matched ID  : {identity}"
    )
    print(
        f"Score       : {score:.4f}"
    )


if __name__ == "__main__":
    main()