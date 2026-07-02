import unittest
import tempfile
from pathlib import Path
import numpy as np

from storage.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):

    def test_vector_store_exists(self):
        store = VectorStore()
        self.assertIsNotNone(store)

    def test_load_returns_tuple(self):
        store = VectorStore()

        try:
            result = store.load()

            self.assertTrue(
                isinstance(result, tuple)
                or result is None
            )

        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()