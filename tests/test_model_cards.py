import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from elderly_depression import services


class ModelCardTests(unittest.TestCase):
    def test_model_cards_describe_roles_metrics_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            tabular = base / "tabular.csv"
            pd.DataFrame(
                [
                    {
                        "model": "logistic",
                        "auc": 0.81,
                        "accuracy": 0.75,
                        "f1": 0.70,
                        "recall": 0.72,
                    }
                ]
            ).to_csv(tabular, index=False)
            multimodal = base / "multimodal.csv"
            pd.DataFrame(
                [
                    {
                        "modality": "fusion",
                        "auc_mean": 0.73,
                        "pr_auc_mean": 0.52,
                        "f1_mean": 0.56,
                        "sensitivity_mean": 0.82,
                        "specificity_mean": 0.69,
                    }
                ]
            ).to_csv(multimodal, index=False)
            cmdc_summary = base / "cmdc_summary.json"
            cmdc_summary.write_text(
                json.dumps(
                    {
                        "primary_result": {
                            "auc_mean": 0.96,
                            "pr_auc_mean": 0.97,
                            "sensitivity_mean": 0.92,
                            "specificity_mean": 0.96,
                            "brier_mean": 0.05,
                        },
                        "dataset_fingerprint": "a" * 64,
                        "schema_version": 1,
                    }
                ),
                encoding="utf-8",
            )
            holdout = base / "holdout.json"
            holdout.write_text(
                json.dumps({"holdout_metrics": {"auc": 1.0}}),
                encoding="utf-8",
            )
            fusion = base / "fusion.json"
            fusion.write_text(json.dumps({"variants": {}}), encoding="utf-8")
            mpdd = base / "mpdd.csv"
            mpdd.write_text("subject_id,label\nS1,1\n", encoding="utf-8")
            with (
                mock.patch.object(services, "MODEL_REPORT", tabular),
                mock.patch.object(services, "MULTIMODAL_SUBJECT_REPORT", multimodal),
                mock.patch.object(services, "CMDC_MODEL_SUMMARY", cmdc_summary),
                mock.patch.object(services, "CMDC_HOLDOUT_SUMMARY", holdout),
                mock.patch.object(services, "FUSION_WEIGHTS_JSON", fusion),
                mock.patch.object(
                    services,
                    "SYNTHETIC_AUGMENTATION_REPORT",
                    base / "synthetic.csv",
                ),
                mock.patch.object(services, "MPDD_CSV", mpdd),
                mock.patch.object(services, "MPDD_UPLOAD_CSV", base / "none.csv"),
            ):
                cards = services.model_cards()

        by_id = {card["id"]: card for card in cards}
        self.assertTrue(
            {"charls_tabular", "mpdd_multimodal", "cmdc_external", "assessment_fusion"}
            .issubset(by_id)
        )
        self.assertEqual(len(by_id["cmdc_external"]["artifact"]["sha256"]), 64)
        self.assertEqual(by_id["cmdc_external"]["dataset_fingerprint"], "a" * 64)
        self.assertIn("临床部署", by_id["cmdc_external"]["not_intended_for"])


if __name__ == "__main__":
    unittest.main()
