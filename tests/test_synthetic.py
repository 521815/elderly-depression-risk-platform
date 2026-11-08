import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from elderly_depression import synthetic


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        self._patchers = [
            mock.patch.object(synthetic, "SYNTHETIC_DIR", self._base),
            mock.patch.object(synthetic, "DIALOGUE_CSV", self._base / "dialogue.csv"),
            mock.patch.object(synthetic, "TABULAR_CSV", self._base / "tabular.csv"),
            mock.patch.object(synthetic, "MULTIMODAL_PARQUET", self._base / "mpdd.parquet"),
            mock.patch.object(synthetic, "MULTIMODAL_CSV", self._base / "mpdd.csv"),
            mock.patch.object(synthetic, "REPORT_JSON", self._base / "report.json"),
            mock.patch.object(
                synthetic, "DIALOGUE_VALIDATION_JSON", self._base / "dialogue_validation.json"
            ),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        self._tmp.cleanup()

    def test_dialogue_generation(self):
        df = synthetic.generate_dialogue_samples(n=20, seed=1)
        self.assertEqual(len(df), 20)
        self.assertTrue(set(df["risk_level"]).issubset({"low", "medium", "high"}))
        self.assertIn("transcript_json", df.columns)

    def test_tabular_generation_is_balanced_and_separate(self):
        real = pd.DataFrame(
            {
                "age": [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                "comorbidity_count": [0, 1, 2, 3, 0, 1, 2, 3, 4, 1],
                "depression": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            }
        )
        syn = synthetic.generate_tabular_synthetic(
            real,
            feature_cols=["age", "comorbidity_count"],
            label_col="depression",
            n_per_class=5,
            seed=3,
        )
        self.assertEqual(len(syn), 10)
        self.assertEqual(float(syn["depression"].mean()), 0.5)
        self.assertNotIn("id", syn.columns)

    def test_multimodal_generation(self):
        real = pd.DataFrame(
            {
                "subject_id": ["1", "2", "3", "4"],
                "hamd_score": [4.0, 5.0, 12.0, 16.0],
                "face_0": [0.1, 0.2, 0.8, 0.9],
                "audio_0": [0.3, 0.4, 0.7, 0.8],
            }
        )
        syn = synthetic.generate_multimodal_synthetic(real, n_per_class=4, seed=5)
        self.assertEqual(len(syn), 8)
        self.assertIn("face_0", syn.columns)
        self.assertEqual(set(syn["fold"]), {"synthetic"})

    def test_delete_synthetic(self):
        synthetic.REPORT_JSON.write_text("{}", encoding="utf-8")
        self.assertTrue(synthetic.delete_synthetic())
        self.assertFalse(synthetic.REPORT_JSON.exists())

    def test_dialogue_validation(self):
        df = synthetic.generate_dialogue_samples(n=30, seed=2)
        result = synthetic.validate_dialogue_samples(df)
        self.assertIn("gates", result)
        self.assertIn("feasible", result)
        self.assertIn("sentence_length_ks_d", result["metrics"])


if __name__ == "__main__":
    unittest.main()
