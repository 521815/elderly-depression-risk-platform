import json
import unittest

from elderly_depression.research_dialogues import (
    build_session_frame,
    extract_structured_labels,
    subject_level_split,
)


class ResearchDialogueTests(unittest.TestCase):
    def test_structured_labels(self):
        transcript = json.dumps(
            [
                {"role": "user", "text": "最近一个月总睡不着，白天也没有力气"},
                {"role": "user", "text": "这些情况影响生活，很多事做不了"},
            ],
            ensure_ascii=False,
        )
        labels = extract_structured_labels(transcript)
        self.assertEqual(labels["dlg_sleep"], 1)
        self.assertEqual(labels["dlg_somatic"], 1)
        self.assertEqual(labels["dlg_functional_impact"], 1)
        self.assertIn("个月", labels["dlg_duration_text"])

    def test_subject_level_split_has_no_overlap(self):
        records = []
        for subject in range(12):
            for session in range(2):
                records.append(
                    {
                        "subject_id": f"s{subject}",
                        "session_id": f"s{subject}-{session}",
                        "age": 65 + subject,
                        "gender": "female" if subject % 2 else "male",
                        "comorbidity_count": subject % 4,
                        "scale_name": "demo",
                        "scale_score": 4 + subject,
                        "depression_label": int(subject >= 6),
                        "transcript_json": json.dumps(
                            [{"role": "user", "text": "最近睡不好"}],
                            ensure_ascii=False,
                        ),
                        "consent_code": f"consent-{subject}",
                    }
                )
        frame = build_session_frame(records)
        split = subject_level_split(frame)
        sets = {
            name: set(group["subject_id"])
            for name, group in split.groupby("split")
        }
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])
        self.assertEqual(set(split["split"]), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
