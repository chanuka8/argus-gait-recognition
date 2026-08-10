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
        description="Train ARGUS ByGaitLight model with metric learning (HPP + ArcFace + Triplet)."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
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
        "--run-dir",
        type=str,
        default="runs/exp_002_hpp_arcface",
        help="Directory to save experiment checkpoints and logs.",
    )

    parser.add_argument(
        "--part-bins",
        type=int,
        default=4,
        help="Horizontal Part Pooling (HPP) part bins.",
    )

    parser.add_argument(
        "--split-config",
        type=str,
        default="configs/subject_split.json",
        help="Subject split manifest configuration path.",
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
        default=0.5,
        help="Weight for Batch-Hard Triplet loss."
    )

    parser.add_argument(
        "--loss-mode",
        type=str,
        choices=["ce", "ce_arcface"],
        default="ce_arcface",
        help="Loss mode to use. Default is ArcFace ('ce_arcface')."
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
        run_dir=args.run_dir,
        part_bins=args.part_bins,
        split_config_path=args.split_config,
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
