import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT),
)

from training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ARGUS ByGaitLight model with metric learning. "
                    "Note: Previous short triplet run degraded accuracy, so triplet loss is optional and default --triplet-weight is set to 0.0."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--triplet-margin",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--triplet-weight",
        type=float,
        default=0.0,
        help="Weight for triplet loss. Default is 0.0 to prevent training degradation observed in previous runs."
    )

    parser.add_argument(
        "--loss-mode",
        type=str,
        choices=["ce", "ce_arcface"],
        default="ce",
        help="Loss mode to use. Default is stable cross-entropy ('ce')."
    )

    parser.add_argument(
        "--arcface-scale",
        type=float,
        default=30.0,
        help="Scale parameter for ArcFace loss."
    )

    parser.add_argument(
        "--arcface-margin",
        type=float,
        default=0.50,
        help="Margin parameter for ArcFace loss."
    )

    args = parser.parse_args()

    trainer = Trainer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_classes=args.max_classes,
        max_samples=args.max_samples,
        triplet_margin=args.triplet_margin,
        triplet_weight=args.triplet_weight,
        loss_mode=args.loss_mode,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
    )

    trainer.train()



if __name__ == "__main__":
    main()