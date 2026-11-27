#!/usr/bin/env python3
"""Run scripted multi-turn dialogue checks against the current Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from backend_api.agent_adapter import AgentAdapter


INTERVIEW_CASES = [
    ("您好", ["您好", "我在这儿"]),
    ("最近总睡不着，半夜醒", ["睡", "醒"]),
    ("最近药有时候忘记吃", ["吃药", "按时", "漏服", "忘记", "提醒"]),
    ("洗澡穿衣需要帮忙", ["穿衣", "洗澡", "日常"]),
    ("最近站不稳，怕跌倒", ["跌倒", "站不稳", "头晕"]),
    ("最近吃不下，体重也轻了", ["胃口", "体重", "吃"]),
    ("肩膀不舒服", ["肩膀", "酸痛", "僵硬"]),
    ("以前爱听戏，现在觉得没意思", ["没意思", "兴趣", "高兴", "以前"]),
    ("最近心里很难受，总想哭", ["难受", "哭", "心情"]),
    ("总是记不住事情", ["记", "注意", "脑子"]),
    ("平时一个人，没人说话", ["一个人", "说话", "孤单"]),
    ("觉得自己拖累孩子", ["负担", "拖累", "自责", "感受", "想法"]),
    ("高血压最近不稳定", ["血压", "复诊", "慢病"]),
    ("最近听东西不清楚", ["听力", "听", "清楚"]),
    ("平时靠老伴照顾", ["照顾", "家人", "帮忙"]),
    ("有时候觉得活着没意思", ["安全", "家人", "医生", "急救"]),
]


COMPANION_CASES = [
    ("你好呀", ["您好", "我在这儿"]),
    ("今天天气不错", ["天气", "穿着"]),
    ("今天吃了饺子", ["吃饭", "吃"]),
    ("孩子昨天来看我了", ["家人", "来看", "电话"]),
    ("我平时喜欢下棋", ["喜欢", "下棋", "高兴"]),
    ("今天身体还行", ["身体", "舒服", "精神"]),
    ("肩膀不舒服", ["肩膀", "不舒服", "酸痛"]),
    ("想起年轻时候的事情", ["以前", "故事", "想起"]),
    ("今天收到孩子的消息，很开心", ["高兴", "开心", "好"]),
    ("今天遇到点烦心事", ["烦", "不痛快", "难受"]),
    ("一个人有点孤单", ["一个人", "说话", "孤单"]),
    ("快过节了，家里准备包饺子", ["过节", "家里", "节日"]),
    ("家里养了一只猫", ["猫", "小动物", "陪"]),
    ("最近在听老歌", ["听歌", "电视", "电影"]),
    ("今天和邻居聊了一会儿", ["邻居", "串门", "散步"]),
    ("今天先聊到这", ["下次", "照顾", "聊到"]),
]

RHYTHM_CASES = [
    ("还好，就是作息有点难调", ["作息", "睡得晚", "补觉", "几点睡"]),
]

RHYTHM_SEQUENCE_CASES = [
    (
        "睡的还行，白天也还好，就是睡的我觉得有点晚，调整作息有困难",
        ["都还可以", "几点上床", "几点起"],
    ),
    (
        "一两点，早上自然醒的话会到十点",
        ["小时", "自然醒", "夜里"],
    ),
    (
        "夜里不太醒，白天也还行",
        ["质量", "作息"],
    ),
]

RHYTHM_CLARIFICATION_CASES = [
    ("睡的还不错，调作息有困难", ["睡眠本身还可以", "几点"]),
    ("不是，就是想调整作息", ["明白", "真正想解决", "固定起床"]),
]

COMPOSITE_CASES = [
    (
        "interview",
        "我心情还行，就是晚上老醒",
        ["睡不好", "醒", "白天"],
    ),
    (
        "interview",
        "平时有人陪，不过还是觉得孤单",
        ["孤单", "心里话", "打个电话"],
    ),
    (
        "interview",
        "不是睡不着，是早上醒太早",
        ["醒得太早", "再睡着"],
    ),
    (
        "interview",
        "吃药基本正常，就是总忘记",
        ["忘记", "药盒", "提醒"],
    ),
    (
        "companion",
        "今天天气不错，不过心里有点堵",
        ["不痛快", "难受", "心里"],
    ),
]

PROFESSIONAL_SLOT_CASES = [
    ("最近药有时候忘记吃", ["早上还是晚上"]),
    ("早上", ["提醒", "睡"]),
]

SITUATIONAL_SLEEP_CASES = [
    ("最近睡得怎么样", ["睡"]),
    ("昨天看了个电影所以睡晚了", ["晚睡", "电影", "第二天"]),
    ("并没有少点乐趣啊", ["理解偏", "看电影", "第二天"]),
]

COMPENSATED_LATE_SLEEP_CASES = [
    (
        "昨晚看电影睡晚了，但是因为时长睡够了所以精神还好",
        ["睡得晚", "总时长睡够", "精神也没受影响"],
    ),
]

EXERCISE_LATE_SLEEP_CASES = [
    (
        "睡的不错，就是作息还没调整好，入睡的相对较晚，可能是因为健身导致的",
        ["睡眠本身不错", "健身时间", "几点结束健身"],
    ),
]

CORTISOL_FOLLOWUP_CASES = [
    ("最近睡得怎么样", ["睡"]),
    (
        "可能是因为健身导致的皮质醇",
        ["皮质醇短时升高", "不能只凭这个判断", "几点结束健身"],
    ),
]

SLEEP_EXERCISE_TIMING_CASES = [
    ("最近睡得怎么样", ["睡"]),
    ("可能是因为健身导致的皮质醇", ["几点结束健身"]),
    ("九点", ["21点结束", "几点上床"]),
    ("十一点", ["大约间隔2小时", "健身提前", "降低强度"]),
    ("把健身提前", ["提前三十分钟", "连续记录一周", "以前让您高兴"]),
]

INTEREST_AFFIRMATIVE_CASES = [
    ("我以前喜欢下棋，现在也愿意继续做", ["喜欢", "下棋"]),
    ("愿意的", ["那就好", "乐趣", "支持"]),
    ("偶尔会", ["不好受", "愿意说出来", "打个电话"]),
    ("会影响吃饭和睡觉", ["影响到日常生活", "持续多久"]),
    ("已经一个月了", ["持续：一个月", "专业评估"]),
]

HELP_REQUEST_CASES = [
    ("我该怎么办，真的不知道怎么办", ["主动求助", "自身安全", "身体急症"]),
    ("是情绪和生活问题", ["信任的人", "发消息或打电话"]),
]

SUPPORT_PREFERENCE_CASES = [
    ("今天心里很烦，什么都不想做", ["先听", "想办法"]),
    ("一起想办法", ["可信的人", "先从哪一步开始"]),
]

DIGNITY_LIFE_REVIEW_CASES = [
    ("我老了，没用了，成了家里的负担", ["不好受", "价值"]),
    ("我想说说这一辈子最骄傲的事", ["人生", "经历"]),
]

ITERATION_V3_CASES = [
    ("interview", [("最近心里很难受，总想哭", ["难受", "持续", "影响"])]),
    (
        "interview",
        [("你答非所问，我不想继续这样聊", ["没接准", "换个话题"])],
    ),
    (
        "companion",
        [("今天身体还行", ["松快", "精神", "出去走走"])],
    ),
    (
        "companion",
        [
            ("今天心里很烦，什么都不想做", ["先听", "想办法"]),
            ("一起想办法", ["今天能完成", "先写下来", "先联系"]),
        ],
    ),
]


def _run_case(agent: AgentAdapter, mode: str, cases: list[tuple[str, list[str]]], session: str) -> list[dict]:
    messages = []
    results = []
    for text, expected_any in cases:
        messages.append({"role": "user", "text": text})
        reply = agent.chat(messages, {"mode": mode}, session_id=session)
        messages.append({"role": "assistant", "text": reply})
        passed = bool(reply.strip()) and any(word in reply for word in expected_any)
        results.append(
            {
                "mode": mode,
                "input": text,
                "reply": reply,
                "engine": agent.last_engine,
                "expected_any": expected_any,
                "passed": passed,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-env-key", action="store_true")
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "output" / "dialogue_self_evaluation.json"),
    )
    args = parser.parse_args()
    agent = AgentAdapter() if args.use_env_key else AgentAdapter(api_key=None)
    results = _run_case(agent, "interview", INTERVIEW_CASES, "self-interview")
    for index, case in enumerate(RHYTHM_CASES):
        results += _run_case(
            agent,
            "interview",
            [case],
            f"self-interview-rhythm-{index}",
        )
    results += _run_case(
        agent,
        "interview",
        RHYTHM_SEQUENCE_CASES,
        "self-interview-rhythm-sequence",
    )
    results += _run_case(
        agent,
        "interview",
        RHYTHM_CLARIFICATION_CASES,
        "self-interview-rhythm-clarification",
    )
    for index, (mode, text, expected_any) in enumerate(COMPOSITE_CASES):
        results += _run_case(
            agent,
            mode,
            [(text, expected_any)],
            f"self-composite-{index}",
        )
    results += _run_case(
        agent,
        "interview",
        PROFESSIONAL_SLOT_CASES,
        "self-professional-slot",
    )
    results += _run_case(
        agent,
        "interview",
        SITUATIONAL_SLEEP_CASES,
        "self-situational-sleep",
    )
    results += _run_case(
        agent,
        "interview",
        COMPENSATED_LATE_SLEEP_CASES,
        "self-compensated-late-sleep",
    )
    results += _run_case(
        agent,
        "interview",
        EXERCISE_LATE_SLEEP_CASES,
        "self-exercise-late-sleep",
    )
    results += _run_case(
        agent,
        "interview",
        CORTISOL_FOLLOWUP_CASES,
        "self-cortisol-followup",
    )
    results += _run_case(
        agent,
        "interview",
        SLEEP_EXERCISE_TIMING_CASES,
        "self-sleep-exercise-timing",
    )
    results += _run_case(
        agent,
        "interview",
        INTEREST_AFFIRMATIVE_CASES,
        "self-interest-affirmative",
    )
    results += _run_case(
        agent,
        "interview",
        HELP_REQUEST_CASES,
        "self-help-request",
    )
    results += _run_case(
        agent,
        "companion",
        SUPPORT_PREFERENCE_CASES,
        "self-support-preference",
    )
    for index, case in enumerate(DIGNITY_LIFE_REVIEW_CASES):
        results += _run_case(
            agent,
            "companion",
            [case],
            f"self-dignity-life-review-{index}",
        )
    for index, (mode, turns) in enumerate(ITERATION_V3_CASES):
        results += _run_case(
            agent,
            mode,
            turns,
            f"self-iteration-v3-{index}",
        )
    results += _run_case(agent, "companion", COMPANION_CASES, "self-companion")

    replies = [item["reply"] for item in results]
    duplicates = sum(
        1 for current, previous in zip(replies[1:], replies[:-1]) if current == previous
    )
    passed = sum(1 for item in results if item["passed"])
    summary = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "duplicate_consecutive_replies": duplicates,
        "pass_rate": passed / len(results) if results else 0.0,
        "results": results,
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in results:
        marker = "PASS" if item["passed"] else "FAIL"
        print(f"[{marker}] {item['mode']} | {item['engine']} | {item['input']}")
        print(f"  -> {item['reply'][:160].replace(chr(10), ' ')}")
    print(
        json.dumps(
            {k: summary[k] for k in ["total", "passed", "failed", "duplicate_consecutive_replies", "pass_rate"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
