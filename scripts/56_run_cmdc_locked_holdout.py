#!/usr/bin/env python3
"""Run one locked subject-level CMDC holdout evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.services import run_cmdc_locked_holdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    summary = run_cmdc_locked_holdout(
        repeats=args.repeats,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
