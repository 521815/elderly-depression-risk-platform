import tempfile
import unittest
from pathlib import Path
from unittest import mock

from elderly_depression import records


class RecordsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._user_dir = Path(self._tmp.name) / "user_records"
        self._patchers = [
            mock.patch.object(records, "USER_DIR", self._user_dir),
            mock.patch.object(records, "USERS_FILE", self._user_dir / "users.json"),
            mock.patch.object(
                records, "USER_RECORDS_DB", self._user_dir / "user_records.sqlite"
            ),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        self._tmp.cleanup()

    def test_create_user_and_save_assessment(self):
        user = records.create_user(
            "张奶奶",
            age=76,
            gender="female",
            comorbidity_count=2,
        )
        self.assertEqual(records.get_user(user["id"])["name"], "张奶奶")

        records.save_assessment(
            user["id"],
            {
                "depression_probability": 0.65,
                "risk_level": "medium",
                "scale_score": 12,
                "scale_name": "cesd10",
            },
        )
        history = records.list_assessments(user["id"])
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["depression_probability"], 0.65)

        updated = records.update_user(user["id"], comorbidity_count=3)
        self.assertEqual(updated["comorbidity_count"], 3)

    def test_query_and_delete_user(self):
        user = records.create_user("李大爷", note="独居", age=80)
        records.save_assessment(
            user["id"],
            {
                "depression_probability": 0.8,
                "risk_level": "high",
                "scale_score": 16,
                "scale_name": "phq9",
            },
        )

        found = records.list_users(query="李大爷")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["id"], user["id"])
        self.assertEqual(len(records.list_users(query="不存在")), 0)

        filtered = records.list_assessments(user["id"], risk="high", scale="phq9")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(records.list_assessments(user["id"], risk="low"), [])

        self.assertTrue(records.delete_user(user["id"]))
        self.assertIsNone(records.get_user(user["id"]))
        self.assertEqual(records.list_assessments(user["id"]), [])

    def test_export_import_roundtrip_merge_and_replace(self):
        user = records.create_user(
            "王奶奶",
            age=75,
            gender="female",
            comorbidity_count=3,
            note="备份测试",
        )
        records.save_assessment(
            user["id"],
            {
                "depression_probability": 0.72,
                "risk_level": "high",
                "scale_score": 15,
                "scale_name": "gds15",
            },
        )
        backup = records.export_all()
        self.assertEqual(backup["format"], records.BACKUP_FORMAT)
        self.assertEqual(len(backup["users"]), 1)
        self.assertEqual(len(backup["assessments"][user["id"]]), 1)

        other = records.create_user("赵爷爷", age=80)
        records.delete_user(user["id"])
        merged = records.import_backup(backup, mode="merge")
        self.assertEqual(merged["users"], 1)
        self.assertEqual(merged["assessments"], 1)
        self.assertEqual(len(records.list_users()), 2)
        self.assertEqual(records.get_user(user["id"])["note"], "备份测试")

        replaced = records.import_backup(backup, mode="replace")
        self.assertEqual(replaced["mode"], "replace")
        self.assertEqual(len(records.list_users()), 1)
        self.assertIsNone(records.get_user(other["id"]))
        self.assertEqual(len(records.list_assessments(user["id"])), 1)

    def test_import_rejects_invalid_backup(self):
        with self.assertRaises(ValueError):
            records.import_backup({"users": []}, mode="merge")

    def test_update_user_can_clear_nullable_fields(self):
        user = records.create_user(
            "清空测试",
            age=70,
            note="旧备注",
            followup_label=1,
        )
        updated = records.update_user(user["id"], note=None, followup_label=None)
        self.assertIsNone(updated["note"])
        self.assertIsNone(updated["followup_label"])

    def test_save_assessment_protects_generated_fields(self):
        user = records.create_user("字段保护", age=72)
        entry = records.save_assessment(
            user["id"],
            {
                "id": "fake-id",
                "user_id": "fake-user",
                "created_at": "fake-time",
                "depression_probability": 0.5,
            },
        )
        self.assertNotEqual(entry["id"], "fake-id")
        self.assertEqual(entry["user_id"], user["id"])
        self.assertNotEqual(entry["created_at"], "fake-time")

    def test_export_all_includes_more_than_200_assessments(self):
        user = records.create_user("大记录量", age=70)
        for _ in range(205):
            records.save_assessment(
                user["id"],
                {"depression_probability": 0.5, "risk_level": "low"},
            )
        backup = records.export_all()
        self.assertEqual(len(backup["assessments"][user["id"]]), 205)


if __name__ == "__main__":
    unittest.main()
