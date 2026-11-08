"""Segment-level augmentation experiment: within-subject small-noise copies."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.multimodal_synthetic_v3 import run_augmentation_pipeline


def main() -> None:
    result = run_augmentation_pipeline(noise_scale=0.03, k=3)
    print("Validation feasible:", result["validation"]["feasible"])
    print(result["validation"]["gates"])
    print(result["report"].to_string(index=False))


if __name__ == "__main__":
    main()
