import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SPLIT_CONFIG_PATH = "configs/subject_split.json"

DEFAULT_TRAIN_RANGE = (1, 62)    # 001 - 062
DEFAULT_VAL_RANGE = (63, 74)     # 063 - 074
DEFAULT_TEST_RANGE = (75, 124)   # 075 - 124


def format_subject_id(subject_num: int) -> str:
    return f"{subject_num:03d}"


def generate_subject_split_manifest(
    data_dir: str = "data/casia_processed/gei",
    train_range: tuple[int, int] = DEFAULT_TRAIN_RANGE,
    val_range: tuple[int, int] = DEFAULT_VAL_RANGE,
    test_range: tuple[int, int] = DEFAULT_TEST_RANGE,
    output_path: str = DEFAULT_SPLIT_CONFIG_PATH,
    seed: int = 42,
) -> dict:
    data_path = Path(data_dir)
    
    train_subjects = [format_subject_id(i) for i in range(train_range[0], train_range[1] + 1)]
    val_subjects = [format_subject_id(i) for i in range(val_range[0], val_range[1] + 1)]
    test_subjects = [format_subject_id(i) for i in range(test_range[0], test_range[1] + 1)]

    # Count actual available sample sequences per subject if directory exists
    def count_samples(subject_list: list[str]) -> dict[str, int]:
        counts = {}
        for sub in subject_list:
            sub_dir = data_path / sub
            if sub_dir.exists() and sub_dir.is_dir():
                counts[sub] = len(list(sub_dir.glob("*.png")))
            else:
                counts[sub] = 0
        return counts

    manifest = {
        "seed": seed,
        "protocol": "CASIA-B Subject-Disjoint Standard Partition (001-074 Train/Val, 075-124 Test)",
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "subject_counts": {
            "train": len(train_subjects),
            "val": len(val_subjects),
            "test": len(test_subjects),
            "total": len(train_subjects) + len(val_subjects) + len(test_subjects),
        },
        "sample_counts": {
            "train": count_samples(train_subjects),
            "val": count_samples(val_subjects),
            "test": count_samples(test_subjects),
        },
    }

    validate_disjoint_splits(manifest)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    return manifest


def validate_disjoint_splits(manifest: dict) -> None:
    train_set = set(manifest["train_subjects"])
    val_set = set(manifest["val_subjects"])
    test_set = set(manifest["test_subjects"])

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    errors = []
    if train_val_overlap:
        errors.append(f"Train and Val subjects overlap: {sorted(list(train_val_overlap))}")
    if train_test_overlap:
        errors.append(f"Train and Test subjects overlap: {sorted(list(train_test_overlap))}")
    if val_test_overlap:
        errors.append(f"Val and Test subjects overlap: {sorted(list(val_test_overlap))}")

    if errors:
        raise ValueError("DATA LEAKAGE DETECTED in subject split manifest:\n" + "\n".join(errors))


def load_or_create_subject_split(
    config_path: str = DEFAULT_SPLIT_CONFIG_PATH,
    data_dir: str = "data/casia_processed/gei",
) -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        validate_disjoint_splits(manifest)
        return manifest
    return generate_subject_split_manifest(data_dir=data_dir, output_path=config_path)


if __name__ == "__main__":
    m = generate_subject_split_manifest()
    print(f"Manifest created successfully at {DEFAULT_SPLIT_CONFIG_PATH}:")
    print(f"Train: {len(m['train_subjects'])}, Val: {len(m['val_subjects'])}, Test: {len(m['test_subjects'])}")
