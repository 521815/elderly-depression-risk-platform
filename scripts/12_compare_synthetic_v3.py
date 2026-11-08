"""Run hierarchical cross-modal synthetic generation, validation and subject-aware comparison."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.multimodal_synthetic_v3 import run_v3_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--separation-scale", type=float, default=0.6)
    parser.add_argument("--residual-scale", type=float, default=0.7)
    args = parser.parse_args()
    result = run_v3_pipeline(
        n_subjects=120,
        segments_per_subject=5,
        separation_scale=args.separation_scale,
        residual_scale=args.residual_scale,
    )
    report = result["report"]
    out = PROJECT_ROOT / "output" / "synthetic_v3_subject_report.csv"
    report.to_csv(out, index=False, encoding="utf-8-sig")
    print("Validation feasible:", result["validation"]["feasible"])
    print(result["validation"]["gates"])
    print(report.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
