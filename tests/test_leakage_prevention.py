import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset_split import load_or_create_subject_split
from evaluation.leakage_validator import (
    assert_subject_disjointness,
    assert_gallery_probe_disjointness,
    assert_no_test_threshold_calibration,
)


class TestLeakagePrevention:

    def test_repository_subject_split_disjointness(self):
        manifest = load_or_create_subject_split()
        train_subs = manifest["train_subjects"]
        val_subs = manifest["val_subjects"]
        test_subs = manifest["test_subjects"]

        # Should pass with no exception
        assert_subject_disjointness(train_subs, val_subs, test_subs)

    def test_detects_train_test_overlap(self):
        train_subs = ["001", "002", "003"]
        val_subs = ["004", "005"]
        test_subs = ["003", "006"]  # "003" overlaps

        with pytest.raises(ValueError, match="Train and Test subjects overlap"):
            assert_subject_disjointness(train_subs, val_subs, test_subs)

    def test_detects_gallery_probe_path_overlap(self):
        gal_paths = ["data/casia/075/075_nm-01_090.png", "data/casia/075/075_nm-02_090.png"]
        prb_paths = ["data/casia/075/075_nm-02_090.png"]  # Duplicate path

        with pytest.raises(ValueError, match="sample paths exist in BOTH gallery and probe"):
            assert_gallery_probe_disjointness(gal_paths, prb_paths)

    def test_detects_test_threshold_calibration_leakage(self):
        cal_subs = ["063", "064", "075"]  # 075 is a test subject
        test_subs = ["075", "076", "077"]

        with pytest.raises(ValueError, match="Threshold calibration used test subjects"):
            assert_no_test_threshold_calibration(cal_subs, test_subs)

    def test_detects_open_set_unknown_in_gallery(self):
        unknown_subs = ["100", "101"]
        gallery_subs = ["075", "076", "100"]  # 100 in gallery

        with pytest.raises(ValueError, match="Open-set unknown subjects .* exist in gallery"):
            assert_gallery_probe_disjointness(
                gallery_paths=["path1"],
                probe_paths=["path2"],
                unknown_subjects=unknown_subs,
                gallery_subjects=gallery_subs,
            )
