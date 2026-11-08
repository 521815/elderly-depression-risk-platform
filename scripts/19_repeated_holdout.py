"""Repeated 80/20 subject holdout with fixed best augmentation parameters."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.multimodal_synthetic_v3 import run_repeated_holdout


def main() -> None:
    result = run_repeated_holdout(n_repeats=5)
    print("Summary (mean AUC over repeats):")
    print(result["summary"].to_string(index=False))
    print("\nReport rows:", len(result["report"]))


if __name__ == "__main__":
    main()
