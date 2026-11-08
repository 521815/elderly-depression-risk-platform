import unittest

from elderly_depression.dialogue_knowledge import (
    find_general_medical_guidance,
    format_guidance_for_prompt,
    retrieve_guidance,
)


class DialogueKnowledgeTests(unittest.TestCase):
    def test_mood_guidance_is_retrieved(self):
        items = retrieve_guidance("mood", "最近心里堵，偶尔想哭", "mild")
        self.assertTrue(items)
        self.assertEqual(items[0]["topic"], "mood")
        self.assertIn("可信的人", format_guidance_for_prompt(items))

    def test_medication_guidance_prioritizes_safe_use(self):
        items = retrieve_guidance("medication", "吃完药头晕，还漏服", "moderate")
        prompt = format_guidance_for_prompt(items)
        self.assertIn("不要自行加倍或停药", prompt)
        self.assertIn("医生或药师", prompt)

    def test_general_medical_knowledge_matches_common_symptoms(self):
        diabetes = find_general_medical_guidance("最近总是口渴、尿多，体重也下降了")
        self.assertEqual(diabetes[0]["id"], "general-diabetes")
        urinary = find_general_medical_guidance("尿频尿痛，还有点发热")
        self.assertEqual(urinary[0]["id"], "general-urinary")


if __name__ == "__main__":
    unittest.main()
