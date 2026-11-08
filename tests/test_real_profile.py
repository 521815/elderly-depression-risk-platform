import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from elderly_depression import real_profile
from elderly_depression import synthetic


class RealProfileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        self._patchers = [
            mock.patch.object(real_profile, "PROFILE_DIR", self._base),
            mock.patch.object(real_profile, "PROFILE_JSON", self._base / "profile.json"),
            mock.patch.object(synthetic, "SYNTHETIC_DIR", self._base),
            mock.patch.object(synthetic, "TABULAR_CSV", self._base / "tabular.csv"),
            mock.patch.object(synthetic, "MULTIMODAL_PARQUET", self._base / "mpdd.parquet"),
            mock.patch.object(synthetic, "MULTIMODAL_CSV", self._base / "mpdd.csv"),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        self._tmp.cleanup()

    def test_tabular_profile_and_enhanced_generation(self):
        real = pd.DataFrame(
            {
                "age": [61, 65, 72, 78, 84, 62, 68, 75, 81, 66],
                "comorbidity_count": [0, 1, 2, 3, 4, 1, 2, 3, 4, 0],
                "sleep_quality": [1, 2, 3, 4, 2, 2, 3, 4, 3, 1],
                "depression": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            }
        )
        profile = real_profile.analyze_tabular(
            real,
            feature_cols=["age", "comorbidity_count", "sleep_quality"],
            label_col="depression",
        )
        self.assertGreaterEqual(len(profile["age_bands"]), 2)
        self.assertIn("missing_rate", profile)

        syn = synthetic.generate_tabular_synthetic_enhanced(
            real,
            feature_cols=["age", "comorbidity_count", "sleep_quality"],
            label_col="depression",
            n_per_class=4,
            seed=7,
        )
        self.assertEqual(len(syn), 8)
        self.assertGreater(syn["age"].std(), 0)

    def test_multimodal_profile_and_enhanced_generation(self):
        rng = __import__("numpy").random.default_rng(3)
        rows = []
        for i in range(40):
            rows.append(
                {
                    "subject_id": str(i % 10),
                    "segment": f"s{i}",
                    "hamd_score": 12.0 if i % 2 else 4.0,
                    "face_0": float(rng.normal(0.2 if i % 2 else -0.2, 0.1)),
                    "face_1": float(rng.normal(0.1 if i % 2 else -0.1, 0.1)),
                    "audio_0": float(rng.normal(0.3 if i % 2 else -0.2, 0.1)),
                }
            )
        df = pd.DataFrame(rows)
        profile = real_profile.analyze_multimodal(df)
        self.assertIn("face", profile)
        self.assertGreater(profile["face"]["n_features"], 0)

        syn = synthetic.generate_multimodal_synthetic_enhanced(
            df, n_per_class=5, seed=9
        )
        self.assertEqual(len(syn), 10)
        self.assertGreater(syn["face_0"].std(), 0)


if __name__ == "__main__":
    unittest.main()
