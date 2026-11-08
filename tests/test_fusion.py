import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from elderly_depression.models.fusion import train_fusion
from elderly_depression.models.llm_assessor import DialogueAssessor


class FusionTests(unittest.TestCase):
    def test_learned_fusion_weights_are_used_by_assessor(self):
        rng = np.random.default_rng(7)
        n = 600
        signals = pd.DataFrame(
            {
                "text_risk": rng.uniform(0, 1, n),
                "scale_risk": rng.uniform(0, 1, n),
                "context_risk": rng.uniform(0, 1, n),
                "expression_risk": rng.uniform(0, 1, n),
                "mpdd_risk": rng.uniform(0, 1, n),
            }
        )
        logit = (
            -2.0
            + 2.2 * signals["text_risk"]
            + 3.0 * signals["scale_risk"]
            + 1.5 * signals["context_risk"]
            - 1.2 * signals["expression_risk"]
            + 2.4 * signals["mpdd_risk"]
            + rng.normal(0, 0.2, n)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        signals["depression"] = (rng.uniform(size=n) < prob).astype(int)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = train_fusion(signals, label_col="depression", output_dir=out)
            self.assertTrue((out / "fusion_weights.json").exists())
            self.assertTrue((out / "fusion_no_mpdd_calibrator.pkl").exists())
            self.assertTrue((out / "calibration_curve.csv").exists())
            self.assertIn("no_mpdd", result["weights"]["variants"])
            self.assertIn("with_mpdd", result["weights"]["variants"])
            self.assertIn("ece_calibrated", result["report"].columns)

            import elderly_depression.models.llm_assessor as assessor_module

            with mock.patch.object(assessor_module, "FUSION_WEIGHTS_JSON", out / "fusion_weights.json"):
                assessment = DialogueAssessor().assess(
                    transcript=[{"role": "user", "text": "最近睡不好，也不想出门"}],
                    scale_answers={"cesd10": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]},
                    demographics={
                        "age": 78,
                        "gender": "female",
                        "married": False,
                        "comorbidity_count": 2,
                    },
                    expression_summary={"risk": 0.7},
                    mpdd_risk=0.8,
                )
            self.assertEqual(assessment["fusion_mode"], "learned")
            self.assertEqual(assessment["fusion_variant"], "with_mpdd")
            self.assertTrue(assessment["language_cues"])

    def test_language_cues_blend_into_text_risk(self):
        assessor = DialogueAssessor()
        base = assessor.assess(
            transcript=[{"role": "user", "text": "今天天气不错，出去走了走"}],
            scale_answers={"cesd10": [0] * 10},
            demographics={},
            expression_summary={},
            language_cues=[],
        )
        enhanced = assessor.assess(
            transcript=[{"role": "user", "text": "最近睡不好，也不想出门"}],
            scale_answers={"cesd10": [0] * 10},
            demographics={},
            expression_summary={},
            language_cues=[{"cue": "自杀意念", "confidence": 1.0}],
        )
        self.assertEqual(base["text_method"], "rule")
        self.assertEqual(enhanced["text_method"], "rule_llm")
        self.assertGreater(enhanced["text_risk"], base["text_risk"])


if __name__ == "__main__":
    unittest.main()
