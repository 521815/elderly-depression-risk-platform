"""Tune augmentation on tuning subjects, then evaluate on held-out test subjects."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.multimodal_synthetic_v3 import run_augmentation_optimization


def main() -> None:
    result = run_augmentation_optimization()
    print("Tuning grid:")
    print(result["tuning_grid"].to_string(index=False))
    print("\nBest:", result["best"])
    print("\nFinal held-out test report:")
    print(result["final_report"].to_string(index=False))


if __name__ == "__main__":
    main()
