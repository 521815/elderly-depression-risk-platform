#!/usr/bin/env python3
"""Apply empirical-Bayes shrinkage to the CHARLS subgroup report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.smoothing import run_subgroup_shrinkage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "output" / "charls_subgroup_report.csv"),
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output"))
    parser.add_argument("--prior-strength", type=float, default=None)
    args = parser.parse_args()

    result = run_subgroup_shrinkage(
        input_path=args.input,
        output_dir=args.output,
        prior_strength=args.prior_strength,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "report"}, ensure_ascii=False, indent=2))
    print("\nSubgroup shrinkage report:")
    for row in result["report"]:
        print(
            f"{row['group']}={row['value']} n={row['n']} "
            f"raw={row['raw_rate']:.3f} shrunk={row['shrunk_rate']:.3f} "
            f"95%CI=[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        )


if __name__ == "__main__":
    main()
