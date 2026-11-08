import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import elderly_depression.multimodal_synthetic_v3 as v3
from elderly_depression.multimodal_synthetic_v3 import (
    generate_segment_augmented,
    generate_hierarchical_synthetic,
    split_subject_holdout,
    validate_vs_real,
)


class MultimodalSyntheticV3Tests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        self._patchers = [
            mock.patch.object(
                v3, "SYNTHETIC_V3_PARQUET",
                self._base / "syn.parquet",
            ),
            mock.patch.object(
                v3, "VALIDATION_JSON",
                self._base / "validation.json",
            ),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        self._tmp.cleanup()

    def test_hierarchical_generation_and_validation(self):
        rng = np.random.default_rng(1)
        rows = []
        for i in range(20):
            subj = f"s{i}"
            for k in range(2):
                rows.append(
                    {
                        "subject_id": subj,
                        "segment": f"{subj}-{k}",
                        "hamd_score": 12.0 if i % 2 else 4.0,
                        "face_0": float(rng.normal(0.3 if i % 2 else -0.2, 0.1)),
                        "face_1": float(rng.normal(0.2 if i % 2 else -0.1, 0.1)),
                        "audio_0": float(rng.normal(0.4 if i % 2 else -0.1, 0.1)),
                    }
                )
        real = pd.DataFrame(rows)
        syn = generate_hierarchical_synthetic(
            real,
            n_subjects=6,
            segments_per_subject=3,
            seed=7,
            n_latent=4,
            n_residual=2,
        )
        self.assertEqual(len(syn), 18)
        self.assertEqual(syn["subject_id"].nunique(), 6)
        self.assertIn("face_0", syn.columns)

        validation = validate_vs_real(syn, real)
        self.assertIn("gates", validation)
        self.assertIn("feasible", validation)

    def test_segment_augmentation(self):
        real = pd.DataFrame(
            {
                "subject_id": ["1", "1", "2", "2"],
                "segment": ["a", "b", "c", "d"],
                "hamd_score": [4.0, 4.0, 12.0, 12.0],
                "face_0": [0.1, 0.2, 0.8, 0.9],
                "audio_0": [0.3, 0.4, 0.7, 0.8],
            }
        )
        aug = generate_segment_augmented(real, k=2, noise_scale=0.01, seed=3)
        self.assertEqual(len(aug), 8)
        self.assertEqual(aug["subject_id"].nunique(), 2)
        self.assertEqual(set(aug["fold"]), {"synthetic"})

    def test_subject_holdout_split(self):
        real = pd.DataFrame(
            {
                "subject_id": [str(i) for i in range(20)],
                "segment": [f"s{i}" for i in range(20)],
                "hamd_score": [12.0 if i % 2 else 4.0 for i in range(20)],
                "face_0": [0.1] * 20,
            }
        )
        tune, test = split_subject_holdout(real, test_ratio=0.2, seed=5)
        self.assertGreater(len(tune), len(test))
        self.assertTrue(set(tune["subject_id"]).isdisjoint(set(test["subject_id"])))


if __name__ == "__main__":
    unittest.main()
