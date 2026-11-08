import unittest

from elderly_depression.dialogue_policy import (
    build_sleep_response,
    build_policy_response,
    classify_answer_archetype,
    classify_sleep_subtype,
    is_situational_late_sleep,
)


class DialoguePolicyTests(unittest.TestCase):
    def test_answer_archetypes(self):
        self.assertEqual(classify_answer_archetype("最近睡得挺好"), "positive")
        self.assertEqual(classify_answer_archetype("偶尔会醒一下"), "mild")
        self.assertEqual(classify_answer_archetype("最近经常睡不好"), "moderate")
        self.assertEqual(classify_answer_archetype("最近总睡不着"), "moderate")
        self.assertEqual(classify_answer_archetype("每天半夜都醒"), "severe")
        self.assertEqual(classify_answer_archetype("说不清楚"), "uncertain")
        self.assertEqual(classify_answer_archetype("不想聊这个"), "avoidant")
        self.assertEqual(classify_answer_archetype("没有失眠"), "positive")
        self.assertEqual(classify_answer_archetype("没有那么难受"), "positive")

    def test_policy_responses_are_humanized(self):
        response = build_policy_response("sleep", "moderate", variant_key=3)
        self.assertIsNotNone(response)
        self.assertIn("睡", response)
        self.assertIn("？", response)
        avoidant = build_policy_response("mood", "avoidant", variant_key=1)
        self.assertIsNotNone(avoidant)
        self.assertTrue(
            any(word in avoidant for word in ["没关系", "谢谢您", "不勉强"])
        )
        medication = build_policy_response("medication", "moderate", variant_key=2)
        self.assertIsNotNone(medication)
        self.assertIn("漏服", medication)
        self.assertTrue(
            any(word in medication for word in ["分格药盒", "提醒"])
        )

    def test_medication_side_effect_is_not_mistaken_for_forgetting(self):
        response = build_policy_response(
            "medication",
            "mild",
            focus_text="吃完药会头晕",
            positive_context="吃药基本正常",
        )
        self.assertIn("头晕", response)
        self.assertIn("不要自行停药", response)
        self.assertNotIn("漏服后不要自行加倍", response)

    def test_professional_responses_do_not_reask_disclosed_facts(self):
        cases = [
            (
                "function",
                "洗澡穿衣需要帮忙",
                ["需要帮忙", "最费劲"],
                "最近有没有需要家人帮忙",
            ),
            (
                "falls",
                "最近站不稳，怕跌倒",
                ["站不稳", "跌倒", "更明显"],
                "最近有没有差点摔倒",
            ),
            (
                "diet",
                "最近吃不下，体重也轻了",
                ["吃不下", "体重下降", "持续多久"],
                "最近吃得下吗",
            ),
            (
                "sensory",
                "最近听东西不清楚",
                ["听不清楚", "逐渐", "突然"],
                "听声音还清楚吗",
            ),
            (
                "care_support",
                "平时靠老伴照顾",
                ["照顾", "紧急联系人", "马上联系"],
                "平时谁在照顾您",
            ),
        ]
        for topic, text, expected, repeated_question in cases:
            with self.subTest(topic=topic):
                response = build_policy_response(
                    topic,
                    "mild",
                    focus_text=text,
                )
                for phrase in expected:
                    self.assertIn(phrase, response)
                self.assertNotIn(repeated_question, response)

    def test_sleep_rhythm_is_not_midnight_waking(self):
        text = "还好，就是作息有点难调"
        self.assertEqual(classify_sleep_subtype(text), "rhythm")
        response = build_sleep_response(
            "sleep",
            text,
            classify_answer_archetype(text),
            variant_key=4,
        )
        self.assertIsNotNone(response)
        self.assertIn("作息", response)
        self.assertNotIn("中途醒", response)
        disrupted = build_sleep_response(
            "sleep",
            "最近作息乱了",
            "unknown",
            variant_key=5,
        )
        self.assertIn("几点睡", disrupted)

    def test_sleep_rhythm_preserves_positive_sleep_and_daytime(self):
        text = "睡的还行，白天也还好，就是睡的我觉得有点晚，调整作息有困难"
        response = build_sleep_response(
            "sleep",
            text,
            classify_answer_archetype(text),
            variant_key=6,
        )
        self.assertIsNotNone(response)
        self.assertIn("都还可以", response)
        self.assertIn("几点上床", response)
        self.assertNotIn("白天不太舒服", response)

    def test_situational_late_sleep_is_not_a_sleep_disorder(self):
        cases = [
            "昨天看了个电影所以睡晚了",
            "昨晚追剧看到很晚",
            "前天和朋友聊到很晚才睡",
            "最近总熬夜",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertTrue(is_situational_late_sleep(text))
                response = build_sleep_response(
                    "sleep",
                    text,
                    classify_answer_archetype(text),
                    1,
                    focus_text=text,
                )
                self.assertIsNotNone(response)
                self.assertNotIn("作息比较规律", response)
                self.assertTrue(
                    any(word in response for word in ["晚睡", "熬夜", "上床时间"])
                )

    def test_exercise_related_late_sleep_gets_specific_advice(self):
        text = (
            "睡的不错，就是作息还没调整好，入睡的相对较晚，"
            "可能是因为健身导致的"
        )
        response = build_sleep_response(
            "sleep",
            text,
            classify_answer_archetype(text),
            1,
            focus_text=text,
            positive_context="睡的不错",
        )
        self.assertIn("睡眠本身不错", response)
        self.assertIn("健身时间", response)
        self.assertIn("几点结束健身", response)


if __name__ == "__main__":
    unittest.main()
