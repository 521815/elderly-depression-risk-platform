import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from elderly_depression.models.multimodal import (
    predict_subject_pipeline,
    train_late_fusion,
    train_subject_late_fusion,
)


class MultimodalTests(unittest.TestCase):
    def test_threshold_is_selected_without_evaluation_fold(self):
        rng = np.random.default_rng(7)
        y = np.array([0] * 28 + [1] * 12, dtype=int)
        face = y + rng.normal(0, 0.15, size=len(y))
        audio = y + rng.normal(0, 0.3, size=len(y))
        gait = y + rng.normal(0, 0.5, size=len(y))
        df = pd.DataFrame(
            {
                "subject_id": [f"s{i}" for i in range(len(y))],
                "hamd_score": y,
                "hamd_binary": y,
                "face_0": face,
                "audio_0": audio,
                "gait_0": gait,
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = train_late_fusion(
                df,
                prefixes=["face", "audio", "gait"],
                label_col="hamd_binary",
                output_dir=Path(tmp),
            )

        report = result["report"]
        self.assertIn("threshold", report.columns)
        self.assertTrue((report["recall"] > 0).any())
        self.assertTrue((report["f1"] > 0).any())

    def test_subject_aware_cv_and_pipeline(self):
        rng = np.random.default_rng(11)
        subjects = [f"s{i}" for i in range(12)]
        y = np.array([0] * 6 + [1] * 6)
        rows = []
        for subject, label in zip(subjects, y):
            for _ in range(4):
                rows.append(
                    {
                        "subject_id": subject,
                        "hamd_binary": label,
                        "face_0": label + rng.normal(0, 0.1),
                        "face_1": label + rng.normal(0, 0.1),
                        "audio_0": label + rng.normal(0, 0.2),
                        "gait_0": label + rng.normal(0, 0.3),
                    }
                )
        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp:
            result = train_subject_late_fusion(
                df,
                prefixes=["face", "audio", "gait"],
                label_col="hamd_binary",
                output_dir=Path(tmp),
                n_splits=3,
                n_repeats=2,
                pca_dims=2,
            )
            out = predict_subject_pipeline(
                result["pipeline"],
                df[df["subject_id"] == "s0"],
            )

        self.assertIn("repeat", result["report"].columns)
        self.assertIn("fold", result["report"].columns)
        self.assertFalse(result["summary"].empty)
        fusion_spec = next(iter(result["pipeline"]["fusion"].values()))
        self.assertEqual(fusion_spec["mode"], "mean")
        self.assertEqual(fusion_spec["threshold"], 0.5)
        self.assertEqual(len(out), 1)
        self.assertIn("fusion_prob", out.columns)
        self.assertIn("prediction", out.columns)


if __name__ == "__main__":
    unittest.main()
