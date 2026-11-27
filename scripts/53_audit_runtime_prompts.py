#!/usr/bin/env python3
"""Deterministic checks for approved runtime prompt templates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.prompt_registry import load_prompt, prompt_versions


CHECKS = {
    "interview": {
        "max_one_question": "最多一个问号",
        "no_diagnosis": "不诊断抑郁症",
        "medical_boundary": "不建议自行停药",
        "safety_escalation": "120、110或12356",
        "topic_switch": "中途换题",
        "user_correction": "用户纠正",
        "stop_control": "用户明确表示停止",
        "no_repeated_opener": "不重复同一开场",
        "action_preference": "用户想一起想办法",
        "structured_output": "只输出一个合法 JSON",
    },
    "companion": {
        "max_one_question": "最多一个问号",
        "not_screening": "不做筛查量表",
        "medical_boundary": "不建议自行停药",
        "safety_escalation": "120、110或12356",
        "support_choice": "继续听",
        "stop_control": "用户想停止",
        "positive_valence": "先顺着积极含义回应",
        "no_repeated_opener": "不连续重复",
        "structured_output": "只输出一个合法 JSON",
    },
}


def main() -> None:
    results = []
    for mode, checks in CHECKS.items():
        prompt, version = load_prompt(
            mode,
            current_topic="睡眠与精力",
            completed_topics="无",
            planned_topic="兴趣与愉悦",
        )
        check_results = {
            check_id: marker in prompt
            for check_id, marker in checks.items()
        }
        check_results["no_unresolved_placeholders"] = "[[" not in prompt
        check_results["reasonable_length"] = 300 <= len(prompt) <= 5000
        results.append(
            {
                "mode": mode,
                "version": version,
                "characters": len(prompt),
                "checks": check_results,
                "passed": all(check_results.values()),
            }
        )

    summary = {
        "runtime_versions": prompt_versions(),
        "total": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "failed": sum(1 for item in results if not item["passed"]),
        "results": results,
    }
    output = PROJECT_ROOT / "output" / "prompt_template_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "total": summary["total"],
                "passed": summary["passed"],
                "failed": summary["failed"],
                "versions": summary["runtime_versions"],
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
