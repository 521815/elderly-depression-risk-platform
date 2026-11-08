import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from elderly_depression.data import feature_cache


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        self._patchers = [
            mock.patch.object(feature_cache, "PARQUET_DIR", self._base / "parquet"),
            mock.patch.object(feature_cache, "MANIFEST_JSON", self._base / "manifest.json"),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        self._tmp.cleanup()

    def test_parquet_roundtrip(self):
        df = pd.DataFrame(
            {
                "subject_id": ["1", "2"],
                "fold": ["train", "test"],
                "hamd_score": [8.0, 12.0],
                "face_0": [0.1, 0.2],
                "face_1": [0.3, 0.4],
                "audio_0": [0.5, 0.6],
                "personality_0": [0.7, 0.8],
            }
        )
        feature_cache.save_parquet_features(df, "test_features")
        loaded = feature_cache.load_parquet_features("test_features")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape[0], 2)
        for col in ["subject_id", "fold", "hamd_score", "face_0", "audio_0", "personality_0"]:
            self.assertIn(col, loaded.columns)
        np.testing.assert_allclose(loaded["face_0"], df["face_0"], rtol=1e-6)

    def test_manifest_freshness(self):
        root = self._base / "raw"
        root.mkdir()
        (root / "a.npy").write_bytes(np.zeros(4, dtype=np.float32).tobytes())
        (root / "b.npy").write_bytes(np.zeros(8, dtype=np.float32).tobytes())

        feature_cache.write_manifest(root)
        self.assertTrue(feature_cache.manifest_fresh(root))

        (root / "c.npy").write_bytes(np.zeros(4, dtype=np.float32).tobytes())
        self.assertFalse(feature_cache.manifest_fresh(root))


if __name__ == "__main__":
    unittest.main()
