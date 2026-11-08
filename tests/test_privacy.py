import unittest

from elderly_depression.privacy import (
    memory_for_cloud,
    redact_known_values,
    redact_messages,
    redact_text,
)


class PrivacyTests(unittest.TestCase):
    def test_direct_identifiers_are_redacted(self):
        text = "我叫王阿姨，住在上海市浦东新区，电话13800138000，邮箱a@example.com"
        redacted = redact_text(text)
        self.assertNotIn("王阿姨", redacted)
        self.assertNotIn("浦东新区", redacted)
        self.assertNotIn("13800138000", redacted)
        self.assertNotIn("a@example.com", redacted)

    def test_messages_are_redacted_without_dropping_roles(self):
        messages = [
            {"role": "user", "text": "我叫李叔，住在地铁站附近"},
            {"role": "assistant", "text": "我在听"},
        ]
        cleaned = redact_messages(messages)
        self.assertEqual(cleaned[0]["role"], "user")
        self.assertNotIn("李叔", cleaned[0]["text"])
        self.assertEqual(cleaned[1]["text"], "我在听")

    def test_known_session_identifiers_are_removed_anywhere_in_text(self):
        redacted = redact_known_values(
            "王阿姨今天说住在徐汇区，王阿姨睡不好",
            ["王阿姨", "徐汇区"],
        )
        self.assertNotIn("王阿姨", redacted)
        self.assertNotIn("徐汇区", redacted)

    def test_memory_drops_direct_identity_fields(self):
        memory = {
            "姓名": "王阿姨",
            "居住地": "上海",
            "近期目标": "调整作息",
        }
        cloud_memory = memory_for_cloud(memory)
        self.assertNotIn("姓名", cloud_memory)
        self.assertNotIn("居住地", cloud_memory)
        self.assertEqual(cloud_memory["近期目标"], "调整作息")


if __name__ == "__main__":
    unittest.main()
