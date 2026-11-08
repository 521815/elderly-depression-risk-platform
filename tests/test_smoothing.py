import tempfile
import unittest
from pathlib import Path

import pandas as pd

from elderly_depression import smoothing


class SmoothingTests(unittest.TestCase):
    def test_beta_binomial_shrinkage_moves_small_group(self):
        result = smoothing.beta_binomial_shrink(
            positive=1,
            total=2,
            global_rate=0.4,
            prior_strength=20,
        )
        self.assertLess(result["shrunk_rate"], result["raw_rate"])
        self.assertGreater(result["shrunk_rate"], 0.4)
        self.assertLessEqual(result["ci_low"], result["shrunk_rate"])
        self.assertGreaterEqual(result["ci_high"], result["shrunk_rate"])

    def test_ema_is_bounded(self):
        values = smoothing.exponential_moving_average([0.1, 0.9, 0.2], alpha=0.5)
        self.assertEqual(len(values), 3)
        self.assertTrue(all(0 <= value <= 1 for value in values))
        self.assertAlmostEqual(values[0], 0.1)

    def test_robust_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "utterances.csv"
            pd.DataFrame(
                {
                    "subject_id": ["s1", "s1", "s1", "s2"],
                    "hamd_score": [10, 10, 10, 2],
                    "face_0": [0.0, 0.0, 100.0, 1.0],
                    "audio_0": [1.0, 2.0, 3.0, 4.0],
                }
            ).to_csv(path, index=False)
            med = smoothing._aggregate_subjects(path, "median")
            mean = smoothing._aggregate_subjects(path, "mean")
            self.assertEqual(float(med.loc[med["subject_id"] == "s1", "face_0"].iloc[0]), 0.0)
            self.assertAlmostEqual(
                float(mean.loc[mean["subject_id"] == "s1", "face_0"].iloc[0]),
                100 / 3,
                places=5,
            )

    def test_subgroup_shrinkage_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "subgroup.csv"
            pd.DataFrame(
                [
                    {"group": "age", "value": "<70", "n": 100, "positive_rate": 0.2},
                    {"group": "age", "value": "70+", "n": 10, "positive_rate": 0.8},
                ]
            ).to_csv(source, index=False)
            result = smoothing.run_subgroup_shrinkage(
                input_path=source,
                output_dir=root,
                prior_strength=10,
            )
            self.assertTrue((root / "charls_subgroup_shrunk.csv").exists())
            small = next(row for row in result["report"] if row["n"] == 10)
            self.assertLess(small["shrunk_rate"], small["raw_rate"])


if __name__ == "__main__":
    unittest.main()
