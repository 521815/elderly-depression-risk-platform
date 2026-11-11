"""Run theory baselines, modality ablations, and cross-source validation audits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.models.incremental_validation import (
    run_incremental_validation,
)
from elderly_depression.smoothing import aggregate_mpdd_subjects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--charls",
        default=str(PROJECT_ROOT / "data" / "charls_standardized.csv"),
    )
    parser.add_argument(
        "--mpdd-utterance",
        default=str(PROJECT_ROOT / "data" / "mpdd_utterance_features.csv"),
    )
    parser.add_argument(
        "--cmdc",
        default=str(PROJECT_ROOT / "data" / "cmdc_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output" / "incremental_validation"),
    )
    parser.add_argument("--charls-splits", type=int, default=5)
    parser.add_argument("--mpdd-splits", type=int, default=5)
    parser.add_argument("--mpdd-repeats", type=int, default=5)
    args = parser.parse_args()

    charls = pd.read_csv(args.charls)
    mpdd = aggregate_mpdd_subjects(args.mpdd_utterance, method="mean")
    cmdc_path = Path(args.cmdc)
    cmdc = pd.read_csv(cmdc_path) if cmdc_path.exists() else None

    result = run_incremental_validation(
        charls=charls,
        mpdd_subject=mpdd,
        cmdc=cmdc,
        output_dir=args.output_dir,
        charls_splits=args.charls_splits,
        mpdd_splits=args.mpdd_splits,
        mpdd_repeats=args.mpdd_repeats,
    )
    print("\nCHARLS theory baselines")
    print(
        result["charls"][
            [
                "baseline_id",
                "model",
                "feature_set",
                "auc",
                "pr_auc",
                "brier",
                "f1",
            ]
        ].to_string(index=False)
    )
    print("\nMPDD modality ablations")
    print(
        result["mpdd"][
            [
                "modalities",
                "auc_mean",
                "pr_auc_mean",
                "brier_mean",
                "f1_mean",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
