import unittest

from elderly_depression.clinical_interview import (
    build_clinical_summary,
    meets_two_week_threshold,
    update_clinical_findings,
)


class ClinicalInterviewTests(unittest.TestCase):
    def test_extracts_frequency_duration_and_function(self):
        findings = update_clinical_findings(
            {},
            "最近一个月几乎每天心里都难受，已经影响吃饭和出门",
        )
        self.assertEqual(findings["duration"], "一个月")
        self.assertEqual(findings["frequency"], "几乎每天")
        self.assertEqual(findings["functional_impact"], "有")
        self.assertEqual(findings["appetite"], "受影响")
        self.assertIn("持续：一个月", build_clinical_summary(findings))

    def test_negative_functional_impact_is_not_misread(self):
        findings = update_clinical_findings(
            {},
            "偶尔会难过，但不影响吃饭和睡觉",
        )
        self.assertEqual(findings["functional_impact"], "无")
        self.assertEqual(findings["frequency"], "偶尔")

    def test_extracts_two_week_duration(self):
        findings = update_clinical_findings({}, "膝盖疼已经两周了")
        self.assertEqual(findings["duration"], "两周")

    def test_two_week_threshold(self):
        self.assertFalse(meets_two_week_threshold("一周"))
        self.assertTrue(meets_two_week_threshold("两周"))
        self.assertTrue(meets_two_week_threshold("一个月"))

    def test_summary_includes_exercise_and_bedtime_when_confirmed(self):
        findings = {
            "exercise_end": "21点",
            "sleep_time": "23点",
        }
        summary = build_clinical_summary(findings)
        self.assertIn("健身结束：21点", summary)
        self.assertIn("上床时间：23点", summary)


if __name__ == "__main__":
    unittest.main()
