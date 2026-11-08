"""End-to-end demo: dialogue + scale answers + demographics -> risk probability."""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.models.llm_assessor import DialogueAssessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="Call OpenAI for a summary")
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    args = parser.parse_args()

    transcript = [
        {
            "role": "assistant",
            "text": "最近睡得怎么样？白天还有没有喜欢做的事情？",
        },
        {
            "role": "user",
            "text": "睡不着，经常到半夜醒。以前喜欢下棋，现在不想动，没意思，感觉拖累家里人。",
        },
        {
            "role": "assistant",
            "text": "身体上有什么不舒服吗？",
        },
        {
            "role": "user",
            "text": "腰疼，没力气，也不太想出门，心里难受。",
        },
    ]
    scale_answers = {
        "cesd10": [1, 1, 2, 1, 1, 1, 2, 0, 1, 1],
        "phq9": [1, 1, 1, 1, 1, 1, 1, 1, 0],
    }
    demographics = {
        "age": 74,
        "gender": "female",
        "married": False,
        "comorbidity_count": 3,
        "sleep_quality": 3,
    }
    expression_summary = {"risk": 0.62}

    completer = None
    if args.llm:
        from elderly_depression.models.llm_assessor import openai_completer

        completer = openai_completer(model=args.llm_model)
    assessor = DialogueAssessor(llm_complete=completer)
    result = assessor.assess(
        transcript=transcript,
        scale_answers=scale_answers,
        demographics=demographics,
        expression_summary=expression_summary,
        scale_name="cesd10",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
