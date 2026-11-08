#!/usr/bin/env python3
"""Train the local domain response-policy model used by the dialogue fast path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.dialogue_response_policy import (
    MODEL_PATH,
    REPORT_PATH,
    train_response_policy,
)


def main() -> None:
    result = train_response_policy(MODEL_PATH, REPORT_PATH)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
