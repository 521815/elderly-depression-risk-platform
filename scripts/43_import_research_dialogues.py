#!/usr/bin/env python3
"""Import and structure de-identified research dialogue sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.research_dialogues import (
    SESSION_FILE,
    build_session_frame,
    load_sessions,
    save_sessions,
)


def _read_input(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload.get("sessions", [])
    return pd.read_csv(path).to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=str(SESSION_FILE))
    parser.add_argument("--mode", choices=["merge", "replace"], default="merge")
    args = parser.parse_args()

    incoming = build_session_frame(_read_input(Path(args.input)))
    target = Path(args.output)
    if args.mode == "merge" and target.exists():
        existing = load_sessions(target)
        combined = pd.concat([existing, incoming], ignore_index=True)
        combined = combined.drop_duplicates(subset=["session_id"], keep="last")
    else:
        combined = incoming
    saved = save_sessions(combined, target)
    print(
        json.dumps(
            {
                "saved": str(saved),
                "sessions": int(len(combined)),
                "subjects": int(combined["subject_id"].nunique()),
                "positive_sessions": int(combined["depression_label"].sum()),
                "structured_columns": [
                    column
                    for column in combined.columns
                    if column.startswith("dlg_")
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
