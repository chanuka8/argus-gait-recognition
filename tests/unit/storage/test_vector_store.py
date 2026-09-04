import inspect
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import cli
from storage.vector_store import VectorStore


class TestVectorStoreSecurity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.gallery_dir = Path(self.temp_dir.name)
        self.store = VectorStore(gallery_dir=str(self.gallery_dir))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_vector_store_exists(self):
        self.assertIsNotNone(self.store)

    def test_valid_numeric_features_and_numeric_labels_load_successfully(self):
        feats = np.random.randn(5, 256).astype(np.float32)
        lbls = np.array([101, 102, 103, 104, 105], dtype=np.int64)
        np.save(self.store.features_file, feats)
        np.save(self.store.labels_file, lbls)

        loaded = self.store.load()
        self.assertIsNotNone(loaded)
        loaded_feats, loaded_lbls, _ = loaded
        self.assertEqual(loaded_feats.dtype, np.float32)
        self.assertEqual(loaded_lbls.dtype, np.int64)

    def test_valid_numeric_features_and_unicode_string_labels_load_successfully(self):
        feats = np.random.randn(5, 256).astype(np.float32)
        lbls = np.array(["P1", "P2", "P3", "P4", "P5"])
        np.save(self.store.features_file, feats)
        np.save(self.store.labels_file, lbls)

        loaded = self.store.load()
        self.assertIsNotNone(loaded)
        loaded_feats, loaded_lbls, _ = loaded
        self.assertEqual(loaded_feats.shape, (5, 256))
        self.assertEqual(list(loaded_lbls), ["P1", "P2", "P3", "P4", "P5"])

    def test_missing_features_file_returns_none(self):
        np.save(self.store.labels_file, np.array(["P1"]))
        self.assertIsNone(self.store.load())

    def test_missing_labels_file_returns_none(self):
        np.save(self.store.features_file, np.random.randn(1, 256).astype(np.float32))
        self.assertIsNone(self.store.load())

    def test_object_dtype_feature_array_is_rejected(self):
        obj_feats = np.array([{"a": 1}], dtype=object)
        lbls = np.array(["P1"])
        np.save(self.store.features_file, obj_feats, allow_pickle=True)
        np.save(self.store.labels_file, lbls)

        with self.assertRaises(ValueError):
            self.store.load()

    def test_object_dtype_label_array_is_rejected(self):
        feats = np.random.randn(1, 256).astype(np.float32)
        obj_lbls = np.array([{"key": "val"}], dtype=object)
        np.save(self.store.features_file, feats)
        np.save(self.store.labels_file, obj_lbls, allow_pickle=True)

        with self.assertRaises(ValueError):
            self.store.load()

    def test_pickled_arrays_are_rejected_with_allow_pickle_false(self):
        obj_feats = np.array([object()], dtype=object)
        lbls = np.array(["P1"])
        np.save(self.store.features_file, obj_feats, allow_pickle=True)
        np.save(self.store.labels_file, lbls)

        with self.assertRaises(ValueError) as ctx:
            self.store.load()
        msg = str(ctx.exception).lower()
        self.assertTrue("pickle" in msg or "prohibited" in msg)

    def test_corrupted_feature_file_is_rejected(self):
        with open(self.store.features_file, "wb") as f:
            f.write(b"CORRUPTED_GARBAGE_BINARY")
        np.save(self.store.labels_file, np.array(["P1"]))

        with self.assertRaises(ValueError):
            self.store.load()

    def test_corrupted_label_file_is_rejected(self):
        np.save(self.store.features_file, np.random.randn(1, 256).astype(np.float32))
        with open(self.store.labels_file, "wb") as f:
            f.write(b"CORRUPTED_GARBAGE_BINARY")

        with self.assertRaises(ValueError):
            self.store.load()

    def test_non_numeric_feature_dtype_is_rejected(self):
        str_feats = np.array([["f1", "f2"]], dtype=str)
        lbls = np.array(["P1"])
        np.save(self.store.features_file, str_feats)
        np.save(self.store.labels_file, lbls)

        with self.assertRaises(ValueError) as ctx:
            self.store.load()
        self.assertIn("numeric", str(ctx.exception).lower())

    def test_feature_array_with_ndim_not_equal_2_is_rejected(self):
        feats_1d = np.random.randn(256).astype(np.float32)
        lbls = np.array(["P1"])
        np.save(self.store.features_file, feats_1d)
        np.save(self.store.labels_file, lbls)

        with self.assertRaises(ValueError) as ctx:
            self.store.load()
        self.assertIn("2-dimensional", str(ctx.exception).lower())

    def test_label_array_with_ndim_not_equal_1_is_rejected(self):
        feats = np.random.randn(1, 256).astype(np.float32)
        lbls_2d = np.array([["P1"]])
        np.save(self.store.features_file, feats)
        np.save(self.store.labels_file, lbls_2d)

        with self.assertRaises(ValueError) as ctx:
            self.store.load()
        self.assertIn("1-dimensional", str(ctx.exception).lower())

    def test_mismatched_feature_label_counts_are_rejected(self):
        feats = np.random.randn(5, 256).astype(np.float32)
        lbls = np.array(["P1", "P2"])
        np.save(self.store.features_file, feats)
        np.save(self.store.labels_file, lbls)

        with self.assertRaises(ValueError) as ctx:
            self.store.load()
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_normal_save_load_round_trip(self):
        feats = np.random.randn(3, 256).astype(np.float32)
        lbls = np.array(["A", "B", "C"])
        meta = {"A": {"status": "ACTIVE"}}

        self.store.save(feats, lbls, meta)
        loaded = self.store.load()

        self.assertIsNotNone(loaded)
        loaded_feats, loaded_lbls, loaded_meta = loaded
        self.assertEqual(loaded_feats.shape, (3, 256))
        self.assertEqual(list(loaded_lbls), ["A", "B", "C"])
        self.assertIn("A", loaded_meta)

    def test_cli_docs_check_uses_allow_pickle_false(self):
        source = inspect.getsource(cli)
        self.assertNotIn("allow_pickle=True", source)
        self.assertIn("allow_pickle=False", source)

    def test_no_active_allow_pickle_true_remains_in_first_party_code(self):
        cmd = ["git", "grep", "-n", "allow_pickle=True"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        prod_matches = [line for line in res.stdout.splitlines() if ".py:" in line and not line.startswith("tests/")]
        self.assertEqual(len(prod_matches), 0, f"Found allow_pickle=True in production code: {prod_matches}")


if __name__ == "__main__":
    unittest.main()
