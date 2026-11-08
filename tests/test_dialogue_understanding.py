import unittest

from elderly_depression.dialogue_understanding import analyze_utterance


class DialogueUnderstandingTests(unittest.TestCase):
    def test_contrast_marker_selects_the_unresolved_concern(self):
        analysis = analyze_utterance("平时有人陪，不过还是觉得孤单")
        self.assertEqual(analysis.focus_text, "还是觉得孤单")
        self.assertIn("平时有人陪", analysis.positive_context)
        self.assertTrue(analysis.has_focused_issue)

    def test_implicit_contrast_keeps_positive_sleep_context(self):
        analysis = analyze_utterance("睡的还不错，调作息有困难")
        self.assertEqual(analysis.focus_text, "调作息有困难")
        self.assertIn("睡的还不错", analysis.positive_context)
        self.assertTrue(analysis.requests_help)

    def test_correction_selects_the_replacement_statement(self):
        analysis = analyze_utterance("不是睡不着，是早上醒太早")
        self.assertEqual(analysis.focus_text, "早上醒太早")
        self.assertTrue(analysis.is_correction)

    def test_correction_without_repeated_issue_is_a_request(self):
        analysis = analyze_utterance("不是，就是想调整作息")
        self.assertEqual(analysis.focus_text, "想调整作息")
        self.assertTrue(analysis.is_correction)
        self.assertTrue(analysis.requests_help)

    def test_explicit_correction_preserves_the_replacement_focus(self):
        analysis = analyze_utterance(
            "你理解错了，我说的是没胃口，不是睡不着"
        )
        self.assertEqual(analysis.focus_text, "没胃口")
        self.assertEqual(analysis.focus_reason, "explicit_correction")
        self.assertTrue(analysis.is_correction)

    def test_single_clause_remains_unchanged(self):
        analysis = analyze_utterance("最近经常睡不好")
        self.assertEqual(analysis.focus_text, "最近经常睡不好")
        self.assertEqual(analysis.focus_reason, "whole_text")


if __name__ == "__main__":
    unittest.main()
