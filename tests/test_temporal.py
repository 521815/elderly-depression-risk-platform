import unittest

import numpy as np
import pandas as pd

from elderly_depression.models.tabular import (
    train_longitudinal_baseline,
    train_temporal_baseline,
)


class TemporalValidationTests(unittest.TestCase):
    def test_later_wave_is_used_as_validation(self):
        rng = np.random.default_rng(11)
        rows = []
        for wave, n in [(1, 180), (2, 160)]:
            age = rng.uniform(60, 90, n)
            chronic = rng.integers(0, 5, n)
            sleep = rng.integers(1, 5, n)
            logit = -3.5 + 0.04 * (age - 60) + 0.35 * chronic + 0.4 * sleep
            prob = 1.0 / (1.0 + np.exp(-logit))
            rows.append(
                pd.DataFrame(
                    {
                        "id": [f"{wave}-{i}" for i in range(n)],
                        "wave": wave,
                        "age": age,
                        "comorbidity_count": chronic,
                        "sleep_quality": sleep,
                        "depression": (rng.uniform(size=n) < prob).astype(int),
                    }
                )
            )
        df = pd.concat(rows, ignore_index=True)
        result = train_temporal_baseline(
            df,
            feature_cols=["age", "comorbidity_count", "sleep_quality"],
            label_col="depression",
            resample_method="none",
            output_dir=None,
        )
        report = result["report"]
        self.assertIn(2, set(report["test_wave"]))
        self.assertNotIn(1, set(report["test_wave"]))
        self.assertTrue({"random_forest", "xgboost", "logistic"}.issubset(set(report["model"])))

    def test_longitudinal_person_level_split(self):
        rng = np.random.default_rng(23)
        rows = []
        for i in range(120):
            pid = f"p{i}"
            base_age = rng.uniform(60, 64)
            prev_cesd = float(rng.integers(0, 25))
            for wave in [1, 2]:
                age = base_age + wave
                logit = -4.0 + 0.12 * prev_cesd + 0.04 * (age - 60)
                prob = 1.0 / (1.0 + np.exp(-logit))
                rows.append(
                    {
                        "id": pid,
                        "wave": wave,
                        "age": age,
                        "comorbidity_count": int(prev_cesd % 5),
                        "sleep_quality": int(prev_cesd % 4) + 1,
                        "cesd_score": prev_cesd,
                        "depression": int(rng.uniform() < prob),
                    }
                )
        df = pd.DataFrame(rows)
        result = train_longitudinal_baseline(
            df,
            feature_cols=["age", "comorbidity_count", "sleep_quality"],
            label_col="depression",
            resample_method="none",
            output_dir=None,
        )
        report = result["report"]
        self.assertEqual(set(report["split"]), {"person-level"})
        self.assertGreater(int(report["train_ids"].iloc[0]), 0)
        self.assertGreater(int(report["test_ids"].iloc[0]), 0)
        self.assertTrue({"random_forest", "xgboost", "logistic"}.issubset(set(report["model"])))
        self.assertIn("prev_cesd_score", result["features"])


if __name__ == "__main__":
    unittest.main()
