import unittest

from elderly_depression.clinical_triage import assess_red_flags


class ClinicalTriageTests(unittest.TestCase):
    def test_chest_pain_with_breathing_is_emergency(self):
        result = assess_red_flags("胸口疼，喘不上气，还冒汗")
        self.assertIsNotNone(result)
        self.assertEqual(result["level"], "emergency")

    def test_stroke_sign_is_emergency(self):
        result = assess_red_flags("突然半边无力，说话不清")
        self.assertIsNotNone(result)
        self.assertEqual(result["level"], "emergency")

    def test_repeated_falls_is_urgent(self):
        result = assess_red_flags("最近反复跌倒，站不稳加重")
        self.assertIsNotNone(result)
        self.assertEqual(result["level"], "urgent")

    def test_shoulder_discomfort_is_not_red_flag(self):
        self.assertIsNone(assess_red_flags("肩膀有点不舒服"))

    def test_negated_emergency_terms_do_not_trigger(self):
        self.assertIsNone(
            assess_red_flags("没有胸口疼，也没有喘不上气，不想死")
        )


if __name__ == "__main__":
    unittest.main()
