import unittest

from elderly_depression.dialogue_response_policy import (
    MODEL_PATH,
    build_strategy_response,
    predict_sleep_strategy,
    rule_sleep_strategy,
)


class DialogueResponsePolicyTests(unittest.TestCase):
    def test_late_sleep_with_enough_duration_is_not_insomnia(self):
        text = "昨晚看电影睡晚了，但是因为时长睡够了所以精神还好"
        self.assertEqual(rule_sleep_strategy(text), "late_compensated")
        response = build_strategy_response("late_compensated")
        self.assertIn("总时长睡够", response)
        self.assertIn("精神也没受影响", response)
        self.assertIn("偶尔一次", response)
        self.assertNotIn("入睡困难", response)

    def test_compensated_sleep_but_fatigued_uses_different_strategy(self):
        text = "昨晚追剧睡晚了，虽然睡够了，但第二天还是很累"
        self.assertEqual(rule_sleep_strategy(text), "late_compensated_tired")
        response = build_strategy_response("late_compensated_tired")
        self.assertIn("夜里睡得踏不踏实", response)

    def test_pretrained_model_loads_and_predicts(self):
        self.assertTrue(MODEL_PATH.exists())
        prediction = predict_sleep_strategy(
            "昨天看电影睡得比较晚，不过总时长补够了，白天精神不错"
        )
        self.assertEqual(prediction["strategy"], "late_compensated")
        self.assertIn(prediction["source"], {"rules+pretrained", "pretrained"})


if __name__ == "__main__":
    unittest.main()
