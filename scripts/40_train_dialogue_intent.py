#!/usr/bin/env python3
"""Train a lightweight intent model from synthetic dialogue transcripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from backend_api.agent_adapter import CUE_RULES
from elderly_depression.dialogue_intent import (
    CUE_TO_TOPIC,
    train_mode_intent_models,
)


CUE_TO_INTENT = {
    "睡眠差": "sleep",
    "兴趣减退": "interest",
    "情绪低落": "mood",
    "躯体不适": "somatic",
    "认知下降": "cognition",
    "社交孤立": "social",
    "无望感": "negative_cognition",
    "自责/负担": "negative_cognition",
    "自杀意念": "safety",
}

PREFIXES = [
    "",
    "最近",
    "这阵子",
    "这段时间",
    "我最近",
    "我总是",
    "我常常",
    "有时候",
    "老是",
    "这几天",
]

SUFFIXES = [
    "",
    "。",
    "，挺难受的。",
    "，不知道怎么办。",
    "，家里人也担心。",
    "，晚上更明显。",
    "，白天也没精神。",
]


def _extract_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for transcript in df.get("transcript_json", []):
        try:
            messages = json.loads(transcript)
        except (TypeError, json.JSONDecodeError):
            continue
        for message in messages:
            if message.get("role") != "user":
                continue
            text = str(message.get("text") or "").strip()
            if not text:
                continue
            matches = []
            for label, keywords in CUE_RULES:
                score = sum(1 for keyword in keywords if keyword in text)
                if score:
                    matches.append((score, label))
            if not matches:
                continue
            matches.sort(key=lambda item: (-item[0], item[1]))
            label = matches[0][1]
            intent = CUE_TO_INTENT.get(label)
            if not intent:
                continue
            rows.append(
                {
                    "text": text,
                    "cue": f"interview:{intent}",
                    "group": f"base:{len(rows)}",
                }
            )
    return rows


def _generate_rows(existing: list[dict]) -> list[dict]:
    rows = list(existing)
    seen = {(row["text"].strip(), row["cue"]) for row in rows}
    for label, keywords in CUE_RULES:
        intent = CUE_TO_INTENT.get(label)
        if not intent:
            continue
        for keyword in keywords:
            for prefix in PREFIXES:
                for suffix in SUFFIXES:
                    text = f"{prefix}{keyword}{suffix}".strip()
                    cue = f"interview:{intent}"
                    key = (text, cue)
                    if not text or key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "text": text,
                            "cue": cue,
                            "group": f"rule:{intent}:{keyword}",
                            "source": "synthetic_paraphrase",
                        }
                    )
    return rows


def _generate_seed_rows(root: Path) -> list[dict]:
    seeds = {}
    for filename in [
        "dialogue_intent_seeds.json",
        "dialogue_intent_seeds_extra.json",
    ]:
        seed_path = root / "data" / filename
        if not seed_path.exists():
            continue
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
        for mode, intents in payload.items():
            mode_seeds = seeds.setdefault(mode, {})
            for intent, phrases in intents.items():
                mode_seeds.setdefault(intent, []).extend(phrases)
    if not seeds:
        return []
    rows = []
    for mode, intents in seeds.items():
        for intent, phrases in intents.items():
            for seed_index, phrase in enumerate(phrases):
                phrase = str(phrase).strip()
                if not phrase:
                    continue
                group = f"seed:{mode}:{intent}:{seed_index}"
                rows.append(
                    {
                        "text": phrase,
                        "cue": f"{mode}:{intent}",
                        "group": group,
                        "source": "seed",
                    }
                )
                for prefix in PREFIXES[1:]:
                    rows.append(
                        {
                            "text": f"{prefix}{phrase}",
                            "cue": f"{mode}:{intent}",
                            "group": group,
                            "source": "synthetic_paraphrase",
                        }
                    )
                for suffix in SUFFIXES[1:]:
                    rows.append(
                        {
                            "text": f"{phrase}{suffix}",
                            "cue": f"{mode}:{intent}",
                            "group": group,
                            "source": "synthetic_paraphrase",
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        default=str(PROJECT_ROOT / "data" / "synthetic" / "dialogue_samples.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "output" / "dialogue_intent_model.pkl"),
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "output" / "dialogue_intent_report.json"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.samples)
    base_rows = _extract_rows(df)
    rows = _generate_rows(base_rows)
    rows.extend(_generate_seed_rows(PROJECT_ROOT))
    training_frame = pd.DataFrame(rows)
    training_path = PROJECT_ROOT / "output" / "dialogue_intent_training.csv"
    training_frame.to_csv(training_path, index=False, encoding="utf-8-sig")
    print(
        f"Base utterances: {len(base_rows)}; "
        f"synthetic paraphrase training rows: {len(rows)}"
    )
    print(training_frame["cue"].value_counts().to_string())
    result = train_mode_intent_models(
        rows,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
