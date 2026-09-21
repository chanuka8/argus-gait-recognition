import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset_split import load_or_create_subject_split
from evaluation.evaluator import SubjectDisjointEvaluator
from evaluation.open_set_evaluator import SubjectDisjointOpenSetEvaluator
from evaluation.threshold_calibrator import ThresholdCalibrator


def evaluate_checkpoint(
    model_path: str,
    output_dir: str,
    gei_root: str = "data/casia_processed/gei",
    split_config: str = "configs/subject_split.json",
    margin_threshold: float = 0.05,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print(f"  EVALUATING: {model_path}")
    print(f"  Margin Threshold: {margin_threshold}")
    print("=======================================================\n")

    split_manifest = load_or_create_subject_split(config_path=split_config, data_dir=gei_root)
    val_subs = split_manifest["val_subjects"]

    evaluator_base = SubjectDisjointEvaluator(
        gei_root=gei_root,
        model_path=model_path,
        split_config_path=split_config,
        report_dir=str(out_dir),
    )

    calibrator = ThresholdCalibrator(
        val_subjects=val_subs,
        feature_extractor_fn=evaluator_base.image_to_embedding,
    )
    calib_res = calibrator.calibrate(
        criterion="min_eer",
        margin_threshold=margin_threshold,
        gei_root=gei_root,
        output_dir=str(out_dir),
    )
    calibrated_threshold = float(calib_res["selected_threshold"])
    print(f"  -> Calibrated Threshold (Val set min_eer): {calibrated_threshold:.4f}")

    evaluator = SubjectDisjointEvaluator(
        gei_root=gei_root,
        model_path=model_path,
        split_config_path=split_config,
        threshold=calibrated_threshold,
        report_dir=str(out_dir),
    )
    closed_set_res = evaluator.evaluate()

    rank1 = float(closed_set_res["rank1_accuracy"])
    rank5 = float(closed_set_res["rank5_accuracy"])
    rank10 = float(closed_set_res.get("rank10_accuracy", 0.0))

    cond_accs = closed_set_res.get("condition_wise_accuracy", {})
    nm_acc = float(cond_accs.get("NM", {}).get("rank1_accuracy", 0.0))
    bg_acc = float(cond_accs.get("BG", {}).get("rank1_accuracy", 0.0))
    cl_acc = float(cond_accs.get("CL", {}).get("rank1_accuracy", 0.0))

    print(f"  -> Rank-1 Accuracy:  {rank1 * 100:.2f}%")
    print(f"  -> Rank-5 Accuracy:  {rank5 * 100:.2f}%")
    print(f"  -> NM Accuracy:      {nm_acc * 100:.2f}%")
    print(f"  -> BG Accuracy:      {bg_acc * 100:.2f}%")
    print(f"  -> CL Accuracy:      {cl_acc * 100:.2f}%")

    open_set_evaluator = SubjectDisjointOpenSetEvaluator(
        gei_root=gei_root,
        model_path=model_path,
        split_config_path=split_config,
        threshold=calibrated_threshold,
        known_ratio=0.5,
        report_dir=str(out_dir),
        margin_threshold=margin_threshold,
    )
    open_set_res = open_set_evaluator.evaluate_open_set_protocol()

    roc_auc = float(open_set_res.get("ROC_AUC", 0.0))
    eer = float(open_set_res.get("EER", 0.0))
    op_metrics = open_set_res.get("operating_metrics", {})
    far = float(op_metrics.get("FAR", 0.0))
    frr = float(op_metrics.get("FRR", 0.0))
    tar = float(op_metrics.get("TAR", 0.0))

    margin_metrics = open_set_res.get("margin_aware_metrics", {})
    margin_far = float(margin_metrics.get("FAR", 0.0))
    margin_frr = float(margin_metrics.get("FRR", 0.0))
    margin_tar = float(margin_metrics.get("TAR", 0.0))

    print(f"  -> ROC-AUC:          {roc_auc:.4f}")
    print(f"  -> EER:              {eer * 100:.2f}%")
    print(f"  -> [Score-Only] FAR: {far * 100:.2f}%  FRR: {frr * 100:.2f}%  TAR: {tar * 100:.2f}%")
    print(
        f"  -> [Margin M>={margin_threshold}] FAR: {margin_far * 100:.2f}%  FRR: {margin_frr * 100:.2f}%  TAR: {margin_tar * 100:.2f}%"
    )

    summary = {
        "model_path": model_path,
        "threshold": round(calibrated_threshold, 4),
        "margin_threshold": margin_threshold,
        "rank1": round(rank1, 4),
        "rank5": round(rank5, 4),
        "rank10": round(rank10, 4),
        "NM": round(nm_acc, 4),
        "BG": round(bg_acc, 4),
        "CL": round(cl_acc, 4),
        "ROC_AUC": round(roc_auc, 4),
        "EER": round(eer, 4),
        "score_only_FAR": round(far, 4),
        "score_only_FRR": round(frr, 4),
        "score_only_TAR": round(tar, 4),
        "margin_aware_FAR": round(margin_far, 4),
        "margin_aware_FRR": round(margin_frr, 4),
        "margin_aware_TAR": round(margin_tar, 4),
        "condition_wise": cond_accs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved evaluation summary to: {summary_path}\n")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gei-root", default="data/casia_processed/gei")
    parser.add_argument("--split-config", default="configs/subject_split.json")
    parser.add_argument(
        "--margin-threshold", type=float, default=0.08, help="Top1/Top2 margin threshold for EXP-004B open-set policy"
    )
    args = parser.parse_args()

    evaluate_checkpoint(
        model_path=args.model_path,
        output_dir=args.output_dir,
        gei_root=args.gei_root,
        split_config=args.split_config,
        margin_threshold=args.margin_threshold,
    )
