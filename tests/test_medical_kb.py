import json
import unittest

from elderly_depression.medical_kb import (
    KB_PATH,
    format_medical_kb_for_prompt,
    search_medical_knowledge,
)


class MedicalKnowledgeBaseTests(unittest.TestCase):
    def test_offline_medlineplus_cache_is_attributed(self):
        self.assertTrue(KB_PATH.exists())
        payload = json.loads(KB_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["source"], "MedlinePlus.gov")
        self.assertGreaterEqual(payload["topic_count"], 18)
        self.assertTrue(payload["retrieved_at"])

    def test_chinese_triggers_retrieve_relevant_topics(self):
        blood_pressure = search_medical_knowledge("最近血压高，后脑勺胀")
        self.assertEqual(blood_pressure[0]["id"], "high_blood_pressure")
        delirium = search_medical_knowledge("老人突然糊涂，意识不清")
        self.assertEqual(delirium[0]["id"], "delirium")

    def test_prompt_format_names_medlineplus(self):
        items = search_medical_knowledge("最近总是失眠")
        prompt = format_medical_kb_for_prompt(items)
        self.assertIn("MedlinePlus", prompt)
        self.assertIn("Insomnia", prompt)


if __name__ == "__main__":
    unittest.main()
