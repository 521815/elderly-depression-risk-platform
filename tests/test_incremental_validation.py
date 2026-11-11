import unittest

import numpy as np
import pandas as pd

from elderly_depression.models.incremental_validation import (
    _paired_wilcoxon_p,
    cross_source_audit,
    run_charls_theory_baselines,
    run_mpdd_modality_ablation,
)


def _charls_fixture() -> pd.DataFrame:
    rows = []
    for subject in range(30):
        label = int(subject >= 15)
        for wave in (1, 2):
            rows.append(
                {
                    "id": subject,
                    "wave": wave,
                    "age": 60 + subject % 15,
                    "gender": 1 + subject % 2,
                    "education": 2 + subject % 5,
                    "marital_status": 1 + subject % 3,
                    "self_rated_health": 1 + subject % 5,
                    "adl_score": subject % 4,
                    "mobility_score": subject % 5,
                    "sleep_quality": 1 + subject % 4,
                    "comorbidity_count": subject % 5,
                    "vigorous_activity": subject % 2,
                    "moderate_activity": (subject + 1) % 2,
                    "light_activity": 1,
                    "smoker": subject % 2,
                    "drinker": (subject + 1) % 2,
                    "income_earn": subject * 100,
                    "income_pension": subject * 50,
                    "social_work": subject % 2,
                    "self_memory": 1 + subject % 5,
                    "immediate_recall": subject % 8,
                    "delayed_recall": subject % 8,
                    "serial7": subject % 5,
                    "cesd_score": 15 if label else 5,
                    "depression": label,
                }
            )
    return pd.DataFrame(rows)


def _mpdd_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = []
    for subject in range(24):
        label = int(subject >= 12)
        for segment in range(2):
            row = {
                "subject_id": str(subject),
                "segment": f"{subject}-{segment}",
                "label2": label,
                "label3": 2 if label else 0,
                "phq9_score": 14 if label else 3,
            }
            for prefix in ("face", "audio", "gait", "personality"):
                row[f"{prefix}_0"] = label + rng.normal(0, 0.2)
            rows.append(row)
    return pd.DataFrame(rows)


class IncrementalValidationTests(unittest.TestCase):
    def test_charls_theory_baselines_are_subject_grouped(self):
        result = run_charls_theory_baselines(
            _charls_fixture(),
            n_splits=3,
            n_bootstrap=5,
        )
        summary = result["summary"]
        self.assertEqual(summary["baseline_id"].tolist(), ["B0", "B1", "B2", "B3", "B4"])
        self.assertNotIn("cesd_score", result["feature_sets"]["theory_full"])
        self.assertTrue((summary["n_subjects"] == 30).all())
        self.assertIn("auc_ci_low", summary.columns)

    def test_mpdd_modality_ablation_covers_all_subsets(self):
        data = _mpdd_fixture()
        subject = (
            data.groupby("subject_id", sort=False)
            .agg(
                {
                    "label2": "first",
                    "label3": "first",
                    "phq9_score": "first",
                    "face_0": "mean",
                    "audio_0": "mean",
                    "gait_0": "mean",
                    "personality_0": "mean",
                }
            )
            .reset_index()
        )
        result = run_mpdd_modality_ablation(
            subject,
            n_splits=3,
            n_repeats=1,
            pca_dims=1,
        )
        summary = result["summary"]
        self.assertEqual(len(summary), 16)
        self.assertIn("face+audio+gait+personality", summary["modalities"].tolist())
        self.assertEqual(
            result["fine_tuning_boundary"]["clinical_end_to_end_finetuning"],
            "not_estimable",
        )

    def test_cross_source_audit_forbids_individual_fusion(self):
        audit = cross_source_audit(
            _charls_fixture(),
            _mpdd_fixture(),
        )
        self.assertFalse(audit["pairing"]["CHARLS__MPDD"])
        self.assertTrue(audit["sources"]["MPDD"]["fields"]["audio"])
        self.assertTrue(
            any("不同受试者" in item for item in audit["forbidden"])
        )

    def test_degenerate_paired_differences_return_neutral_p_value(self):
        self.assertEqual(_paired_wilcoxon_p(np.zeros(25)), 1.0)


if __name__ == "__main__":
    unittest.main()
