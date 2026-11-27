import json
import unittest
from types import SimpleNamespace

from backend_api.agent_adapter import AGENT_CONTRACT, AgentAdapter
from elderly_depression.prompt_registry import (
    load_prompt,
    prompt_versions,
)


class PromptRegistryTests(unittest.TestCase):
    def test_interview_prompt_is_versioned_and_complete(self):
        prompt, version = load_prompt(
            "interview",
            current_topic="睡眠与精力",
            completed_topics="无",
            planned_topic="兴趣与愉悦",
        )
        self.assertEqual(version, "interview_v3")
        self.assertIn("睡眠与精力", prompt)
        self.assertIn("兴趣与愉悦", prompt)
        self.assertIn("最多一个问号", prompt)
        self.assertIn("不诊断抑郁症", prompt)
        self.assertIn("只输出一个合法 JSON", prompt)
        self.assertIn("不重复同一开场", prompt)
        self.assertIn("用户想一起想办法", prompt)
        self.assertNotIn("[[", prompt)

    def test_companion_prompt_stays_out_of_screening(self):
        prompt, version = load_prompt("companion")
        self.assertEqual(version, "companion_v3")
        self.assertIn("不做筛查量表", prompt)
        self.assertIn("最多一个问号", prompt)
        self.assertIn("120、110或12356", prompt)
        self.assertIn("用户想停止", prompt)
        self.assertIn("先顺着积极含义回应", prompt)
        self.assertNotIn("[[", prompt)

    def test_prompt_versions_are_exposed(self):
        versions = prompt_versions()
        self.assertEqual(versions["interview"], "interview_v3")
        self.assertEqual(versions["companion"], "companion_v3")
        self.assertEqual(
            AGENT_CONTRACT["prompt_versions"]["interview"],
            "interview_v3",
        )

    def test_llm_prompt_contains_structured_runtime_state(self):
        class FakeCompletions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                content = json.dumps(
                    {
                        "reply": "睡眠本身还可以，我们先看入睡时间。您通常几点上床？",
                        "topic": "sleep",
                        "risk_cues": [],
                        "cue_confidence": {},
                        "next_question": "您通常几点上床？",
                    },
                    ensure_ascii=False,
                )
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=content)
                        )
                    ]
                )

        completions = FakeCompletions()
        agent = AgentAdapter(api_key="test-key")
        agent._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        state = agent._session(
            "prompt-runtime",
            {"mode": "interview"},
        )
        reply = agent._try_llm_reply(
            [
                {
                    "role": "user",
                    "text": "睡得还行，只是最近入睡比较晚",
                }
            ],
            {"cloud_consent": True},
            state,
            "睡得还行，只是最近入睡比较晚",
            mode="interview",
        )
        self.assertIn("几点上床", reply)
        system_prompt = completions.kwargs["messages"][0]["content"]
        self.assertIn("interview_v3", agent.last_prompt_version)
        self.assertIn("## 本轮可核对状态", system_prompt)
        self.assertIn("当前主题：睡眠与精力", system_prompt)
        self.assertIn("本轮真正焦点", system_prompt)


if __name__ == "__main__":
    unittest.main()
