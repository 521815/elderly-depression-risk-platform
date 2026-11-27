import unittest

from backend_api.agent_adapter import AgentAdapter
from elderly_depression.dialogue_policy import HUMAN_OPENERS


class DialogueIterationV3Tests(unittest.TestCase):
    def test_moderate_openers_do_not_use_fixed_service_thanks(self):
        for opener in HUMAN_OPENERS["moderate"]:
            self.assertNotIn("谢谢您愿意", opener)
            self.assertNotIn("我听到了", opener)

    def test_mood_reply_is_grounded_without_service_thanks(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "最近心里很难受，总想哭"}],
            {"mode": "interview"},
            session_id="iteration-v3-mood",
        )
        self.assertNotIn("谢谢您愿意", reply)
        self.assertIn("难受", reply)
        self.assertLessEqual(reply.count("？"), 1)

    def test_safety_reply_is_direct_and_does_not_thank_user(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "有时候觉得活着没意思"}],
            {"mode": "interview"},
            session_id="iteration-v3-safety",
        )
        self.assertNotIn("谢谢您愿意", reply)
        self.assertIn("安全", reply)
        self.assertIn("能马上联系到的人", reply)

    def test_alliance_repair_does_not_quote_the_complaint_back(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [
                {
                    "role": "user",
                    "text": "你答非所问，我不想继续这样聊",
                },
            ],
            {"mode": "interview"},
            session_id="iteration-v3-repair",
        )
        self.assertIn("没接准", reply)
        self.assertNotIn("“你答非所问，我不想继续这样聊”", reply)
        self.assertNotIn("“您刚才说的话”", reply)
        self.assertIn("换个话题", reply)

    def test_companion_positive_health_does_not_immediately_probe_symptoms(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "今天身体还行"}],
            {"mode": "companion"},
            session_id="iteration-v3-health",
        )
        self.assertIn("松快", reply)
        self.assertNotIn("哪里特别不舒服", reply)

    def test_action_support_gives_one_low_burden_step(self):
        agent = AgentAdapter(api_key=None)
        session = "iteration-v3-action"
        agent.chat(
            [
                {
                    "role": "user",
                    "text": "今天心里很烦，什么都不想做",
                }
            ],
            {"mode": "companion"},
            session_id=session,
        )
        reply = agent.chat(
            [
                {
                    "role": "user",
                    "text": "今天心里很烦，什么都不想做",
                },
                {"role": "assistant", "text": "您更想被听着，还是一起想办法？"},
                {"role": "user", "text": "一起想办法"},
            ],
            {"mode": "companion"},
            session_id=session,
        )
        self.assertIn("今天能完成", reply)
        self.assertNotIn("专业评估", reply)
        self.assertLessEqual(reply.count("？"), 1)


if __name__ == "__main__":
    unittest.main()
