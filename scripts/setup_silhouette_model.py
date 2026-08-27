"""
Silhouette Model Setup and Asset Verification Helper for ARGUS AI.

Validates presence and contract of local learned silhouette segmentation ONNX asset
(models/engines/silhouette_segmenter.onnx).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.steps.silhouette_step import LearnedSilhouetteSegmenter


def setup_or_verify_silhouette_model(
    model_path: str = "models/weights/silhouette_segmenter.onnx",
) -> bool:
    target_path = Path(model_path)
    print(f"[*] Checking silhouette segmentation model asset at: {target_path}")

    segmenter = LearnedSilhouetteSegmenter(model_path=str(target_path))
    valid, reason = segmenter.validate_model()

    if valid:
        print(f"[SUCCESS] {reason}")
        return True

    print(f"[WARNING] {reason}")
    print("[INFO] EXTERNAL MODEL ASSET REQUIRED:")
    print(
        "       Please place a trained ONNX human segmentation model (UNet/SegFormer "
        "256x256 RGB input) at 'models/engines/silhouette_segmenter.onnx'."
    )
    print("       The ARGUS AI runtime will continue using Otsu thresholding fallback until this asset is supplied.")
    return False


if __name__ == "__main__":
    success = setup_or_verify_silhouette_model()
    sys.exit(0 if success else 1)
