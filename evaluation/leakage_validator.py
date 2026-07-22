import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def assert_subject_disjointness(
    train_subjects: list[str],
    val_subjects: list[str],
    test_subjects: list[str],
) -> None:
    train_set = set(str(s) for s in train_subjects)
    val_set = set(str(s) for s in val_subjects)
    test_set = set(str(s) for s in test_subjects)

    tv_overlap = train_set & val_set
    tt_overlap = train_set & test_set
    vt_overlap = val_set & test_set

    errors = []
    if tv_overlap:
        errors.append(f"Train and Val subjects overlap: {sorted(list(tv_overlap))}")
    if tt_overlap:
        errors.append(f"Train and Test subjects overlap: {sorted(list(tt_overlap))}")
    if vt_overlap:
        errors.append(f"Val and Test subjects overlap: {sorted(list(vt_overlap))}")

    if errors:
        raise ValueError("DATA LEAKAGE FAILURE: Subject identity overlap detected!\n" + "\n".join(errors))


def assert_gallery_probe_disjointness(
    gallery_paths: list[str | Path],
    probe_paths: list[str | Path],
    train_subjects: list[str] | None = None,
    unknown_subjects: list[str] | None = None,
    gallery_subjects: list[str] | None = None,
) -> None:
    gal_path_set = set(str(Path(p).resolve()) for p in gallery_paths)
    prb_path_set = set(str(Path(p).resolve()) for p in probe_paths)

    path_overlap = gal_path_set & prb_path_set
    if path_overlap:
        raise ValueError(f"DATA LEAKAGE FAILURE: {len(path_overlap)} sample paths exist in BOTH gallery and probe sets!")

    if train_subjects is not None:
        train_set = set(str(s) for s in train_subjects)
        
        # Check gallery does not contain training subjects
        gal_train_overlap = [p for p in gal_path_set if any(f"/{ts}/" in p.replace("\\", "/") or f"_{ts}_" in p for ts in train_set)]
        if gal_train_overlap:
            raise ValueError(f"DATA LEAKAGE FAILURE: Gallery contains {len(gal_train_overlap)} samples from training subjects!")

        # Check probe does not contain training subjects
        prb_train_overlap = [p for p in prb_path_set if any(f"/{ts}/" in p.replace("\\", "/") or f"_{ts}_" in p for ts in train_set)]
        if prb_train_overlap:
            raise ValueError(f"DATA LEAKAGE FAILURE: Probe contains {len(prb_train_overlap)} samples from training subjects!")

    if unknown_subjects is not None and gallery_subjects is not None:
        unk_set = set(str(s) for s in unknown_subjects)
        gal_sub_set = set(str(s) for s in gallery_subjects)

        unk_gal_overlap = unk_set & gal_sub_set
        if unk_gal_overlap:
            raise ValueError(f"DATA LEAKAGE FAILURE: Open-set unknown subjects {sorted(list(unk_gal_overlap))} exist in gallery!")


def assert_no_test_threshold_calibration(
    calibration_subjects: list[str],
    test_subjects: list[str],
) -> None:
    cal_set = set(str(s) for s in calibration_subjects)
    test_set = set(str(s) for s in test_subjects)

    cal_test_overlap = cal_set & test_set
    if cal_test_overlap:
        raise ValueError(f"DATA LEAKAGE FAILURE: Threshold calibration used test subjects {sorted(list(cal_test_overlap))}!")
