import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from models.architectures.bygait_light import ByGaitLight


def test_forward_pass():

    model = ByGaitLight()

    dummy = torch.randn(
        4,
        1,
        128,
        64,
    )

    output = model(dummy)

    assert output.shape == (4, 256)

    print(
        "PASS - Output shape:",
        output.shape,
    )


if __name__ == "__main__":
    test_forward_pass()