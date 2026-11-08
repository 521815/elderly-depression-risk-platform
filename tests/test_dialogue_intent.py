import tempfile
import unittest
from pathlib import Path
from unittest import mock

from elderly_depression import dialogue_intent


class DialogueIntentTests(unittest.TestCase):
    def test_train_and_predict_small_intent_model(self):
        rows = []
        for index in range(12):
            rows.append(
                {
                    "text": f"最近总是睡不着，半夜会醒{index}",
                    "cue": "睡眠差",
                    "group": f"sleep-{index % 3}",
                }
            )
            rows.append(
                {
                    "text": f"现在什么都不想做，也提不起兴趣{index}",
                    "cue": "兴趣减退",
                    "group": f"interest-{index % 3}",
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.pkl"
            report_path = root / "report.json"
            with mock.patch.object(dialogue_intent, "MODEL_PATH", model_path):
                dialogue_intent._CACHE["mtime"] = None
                dialogue_intent._CACHE["bundle"] = None
                result = dialogue_intent.train_intent_model(
                    rows,
                    output_path=model_path,
                    report_path=report_path,
                )
                self.assertTrue(model_path.exists())
                self.assertTrue(report_path.exists())
                self.assertGreaterEqual(result["samples"], 20)
                prediction = dialogue_intent.predict_intent("晚上睡不着，容易醒")
                self.assertIsNotNone(prediction)
                self.assertEqual(prediction["topic"], "sleep")


if __name__ == "__main__":
    unittest.main()
