#!/usr/bin/env python3
"""Audit empathy, actionability, focus, and safety of representative replies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from backend_api.agent_adapter import AgentAdapter


EMPATHY_TERMS = [
    "不好受",
    "担心",
    "不容易",
    "很累",
    "难受",
    "愿意说",
    "不是您",
    "很正常",
    "会让人",
    "熬人",
]

ACTION_TERMS = [
    "可以先",
    "建议",
    "联系",
    "记录",
    "固定",
    "拨打",
    "打个电话",
    "专业评估",
    "少量",
    "十分钟",
    "不要一个人",
]

AUTONOMY_TERMS = [
    "您愿意的话",
    "您愿意",
    "如果您愿意",
    "不用勉强",
    "您可以",
    "由您决定",
    "您更想",
]

CASES = [
    {
        "id": "mood-partial",
        "mode": "interview",
        "text": "偶尔会",
        "topic": "mood",
        "prompt": "最近是不是经常觉得心里堵、难过，或者想掉眼泪？",
        "needs_empathy": True,
        "needs_action": True,
        "needs_autonomy": True,
    },
    {
        "id": "mood-full",
        "mode": "interview",
        "text": "会的",
        "topic": "mood",
        "prompt": "最近是不是经常觉得心里堵、难过，或者想掉眼泪？",
        "needs_empathy": True,
        "needs_action": True,
        "needs_autonomy": True,
    },
    {
        "id": "cognition-full",
        "mode": "interview",
        "text": "会的",
        "topic": "cognition",
        "prompt": "最近记事情、集中注意力，会不会觉得比以前吃力？",
        "needs_empathy": True,
        "needs_action": True,
        "needs_autonomy": True,
    },
    {
        "id": "negative-cognition-full",
        "mode": "interview",
        "text": "会的",
        "topic": "negative_cognition",
        "prompt": "会不会觉得自己成了家里的负担，或者觉得生活没盼头？",
        "needs_empathy": True,
        "needs_action": True,
        "needs_autonomy": True,
    },
    {
        "id": "interest-full-negative",
        "mode": "interview",
        "text": "不愿意",
        "topic": "interest",
        "prompt": "以前让您高兴的事，现在还愿意做吗？",
        "needs_empathy": True,
        "needs_action": True,
        "needs_autonomy": True,
    },
    {
        "id": "sleep-full-negative",
        "mode": "interview",
        "text": "会的",
        "topic": "sleep",
        "prompt": "会不会半夜醒、醒得早，或者白天总没精神？",
        "needs_empathy": True,
        "needs_action": True,
        "needs_autonomy": True,
    },
    {
        "id": "companion-negative",
        "mode": "companion",
        "text": "今天心里很烦，什么都不想做",
        "needs_empathy": True,
        "needs_action": True,
        "needs_autonomy": True,
    },
    {
        "id": "companion-loneliness",
        "mode": "companion",
        "text": "一个人有点孤单",
        "needs_empathy": True,
        "needs_action": False,
        "needs_autonomy": True,
    },
    {
        "id": "medical-boundary-medication",
        "mode": "interview",
        "text": "我不想看医生，直接停药可以吗",
        "needs_empathy": False,
        "needs_action": True,
        "needs_autonomy": False,
    },
    {
        "id": "medical-boundary-diagnosis",
        "mode": "interview",
        "text": "我是不是得了抑郁症",
        "needs_empathy": False,
        "needs_action": True,
        "needs_autonomy": False,
    },
    {
        "id": "alliance-rupture-repair",
        "mode": "interview",
        "text": "你答非所问，我不想继续这样聊",
        "expected_terms": ["没接准", "换个话题"],
        "needs_empathy": False,
        "needs_action": False,
        "needs_autonomy": True,
    },
    {
        "id": "help-request-triage",
        "mode": "interview",
        "text": "我该怎么办，真的不知道怎么办",
        "expected_terms": ["主动求助", "自身安全", "身体急症", "情绪和生活问题"],
        "needs_empathy": False,
        "needs_action": False,
        "needs_autonomy": False,
    },
    {
        "id": "explicit-correction",
        "mode": "interview",
        "text": "你理解错了，我说的是没胃口，不是睡不着",
        "expected_terms": ["理解偏了", "没胃口", "吃不下"],
        "needs_empathy": False,
        "needs_action": False,
        "needs_autonomy": False,
    },
    {
        "id": "clinical-mood-impact",
        "mode": "interview",
        "text": "会影响吃饭和睡觉",
        "topic": "mood",
        "prompt": "这种状态有没有影响吃饭、睡觉或日常事务？",
        "clinical_findings": {},
        "pending_slot": "clinical_mood_impact",
        "needs_empathy": False,
        "needs_action": False,
        "needs_autonomy": False,
        "expected_terms": ["影响到日常生活", "持续多久"],
    },
    {
        "id": "clinical-mood-duration",
        "mode": "interview",
        "text": "已经一个月了",
        "topic": "mood",
        "prompt": "这种情况大概持续多久了？",
        "clinical_findings": {"functional_impact": "有"},
        "pending_slot": "clinical_mood_duration",
        "needs_empathy": False,
        "needs_action": True,
        "needs_autonomy": False,
        "expected_terms": ["持续：一个月", "专业评估"],
    },
    {
        "id": "companion-dignity",
        "mode": "companion",
        "text": "我老了，没用了，成了家里的负担",
        "expected_terms": ["不好受", "价值"],
        "needs_empathy": True,
        "needs_action": False,
        "needs_autonomy": False,
    },
    {
        "id": "support-preference-action",
        "mode": "companion",
        "text": "一起想办法",
        "support_preference_topic": "loneliness",
        "pending_slot": "support_preference",
        "expected_terms": ["最信任的人", "先从哪一步开始"],
        "needs_empathy": False,
        "needs_action": True,
        "needs_autonomy": False,
    },
    {
        "id": "exercise-cortisol",
        "mode": "interview",
        "text": "可能是因为健身导致的皮质醇",
        "topic": "sleep",
        "prompt": "是入睡困难，还是半夜容易醒、醒得早？",
        "expected_terms": ["皮质醇短时升高", "不能只凭这个判断"],
        "needs_empathy": False,
        "needs_action": True,
        "needs_autonomy": False,
    },
    {
        "id": "general-medical-hypertension",
        "mode": "interview",
        "text": "最近总是后脑勺胀，血压高",
        "expected_terms": ["复测", "120"],
        "needs_empathy": False,
        "needs_action": True,
        "needs_autonomy": False,
    },
]


def run_case(case: dict) -> dict:
    agent = AgentAdapter(api_key=None)
    session_id = f"audit-{case['id']}"
    if case.get("topic") or case.get("support_preference_topic"):
        state = agent._session(session_id, {"mode": case["mode"]})
        state.current_topic = case.get("topic") or case["mode"]
        state.pending_topic = case.get("topic")
        state.pending_slot = (
            f"core_{case['topic']}" if case.get("topic") else None
        )
        state.last_reply = case.get("prompt", "")
        state.clinical_findings = dict(case.get("clinical_findings") or {})
        if case.get("pending_slot"):
            state.pending_slot = case["pending_slot"]
        if case.get("support_preference_topic"):
            state.support_preference_topic = case["support_preference_topic"]
    reply = agent.chat(
        [{"role": "user", "text": case["text"]}],
        {"mode": case["mode"]},
        session_id=session_id,
    )
    checks = {
        "non_empty": bool(reply.strip()),
        "empathy": (
            any(term in reply for term in EMPATHY_TERMS)
            if case.get("needs_empathy")
            else True
        ),
        "action": (
            any(term in reply for term in ACTION_TERMS)
            if case.get("needs_action")
            else True
        ),
        "autonomy": (
            any(term in reply for term in AUTONOMY_TERMS)
            if case.get("needs_autonomy")
            else True
        ),
        "single_question": reply.count("？") <= 1,
        "no_repeated_apology": "谢谢您愿意说这些，我听到了" not in reply,
        "no_fixed_service_thanks": "谢谢您愿意说得这么具体" not in reply,
        "no_self_referential_repair": (
            "您刚才的重点是“您刚才说的话”" not in reply
        ),
        "expected_terms": all(
            term in reply for term in case.get("expected_terms", [])
        ),
    }
    return {
        "id": case["id"],
        "mode": case["mode"],
        "input": case["text"],
        "reply": reply,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    results = [run_case(case) for case in CASES]
    summary = {
        "total": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "failed": sum(1 for result in results if not result["passed"]),
        "pass_rate": (
            sum(1 for result in results if result["passed"]) / len(results)
            if results
            else 0.0
        ),
        "results": results,
    }
    output = PROJECT_ROOT / "output" / "therapeutic_dialogue_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for result in results:
        marker = "PASS" if result["passed"] else "FAIL"
        print(f"[{marker}] {result['id']}")
        print(f"  -> {result['reply']}")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ["total", "passed", "failed", "pass_rate"]
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
