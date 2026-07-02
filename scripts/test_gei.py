import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from preprocessing.gei_builder import GEIBuilder


def main():
    gei_builder = GEIBuilder()

    for _ in range(20):

        frame = np.random.randint(
            0,
            2,
            (128, 64),
            dtype=np.uint8,
        ) * 255

        gei_builder.add_frame(frame)

    gei = gei_builder.build()

    if gei is not None:
        print("GEI generated")
        print("Shape:", gei.shape)
    else:
        print("GEI generation failed")


if __name__ == "__main__":
    main()