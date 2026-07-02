import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.inference_pipeline import (
    InferencePipeline,
)


GEI_ROOT = (
    "data/casia_processed/gei"
)


def random_image():

    images = []

    for person_dir in Path(
        GEI_ROOT
    ).iterdir():

        if person_dir.is_dir():

            images.extend(
                list(
                    person_dir.glob(
                        "*.png"
                    )
                )
            )

    return random.choice(
        images
    )


def main():

    image_path = (
        random_image()
    )

    pipeline = (
        InferencePipeline()
    )

    result = (
        pipeline.predict(
            image_path
        )
    )

    print(
        "\n=== ARGUS INFERENCE ==="
    )

    print(
        f"Image    : {image_path.name}"
    )

    print(
        f"Actual   : {image_path.parent.name}"
    )

    print(
        f"Identity : {result['identity']}"
    )

    print(
        f"Score    : {result['score']:.4f}"
    )


if __name__ == "__main__":
    main()