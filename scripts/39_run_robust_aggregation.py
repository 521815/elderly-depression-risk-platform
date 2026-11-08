#!/usr/bin/env python3
"""Compare mean, median, and trimmed-mean aggregation on MPDD utterances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.smoothing import run_robust_aggregation_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--utterance-csv",
        default=str(PROJECT_ROOT / "data" / "mpdd_utterance_features.csv"),
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--pca-dims", type=int, default=32)
    args = parser.parse_args()

    result = run_robust_aggregation_ablation(
        utterance_path=args.utterance_csv,
        output_dir=args.output,
        repeats=args.repeats,
        splits=args.splits,
        pca_dims=args.pca_dims,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "report"}, ensure_ascii=False, indent=2))
    print("\nFusion comparison:")
    for row in result["fusion_comparison"]:
        print(
            f"{row['aggregation']}: AUC={row['fusion_auc']:.4f} "
            f"F1={row['fusion_f1']:.4f} Brier={row['fusion_brier']:.4f}"
        )


if __name__ == "__main__":
    main()
