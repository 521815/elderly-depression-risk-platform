#!/usr/bin/env python3
"""Test whether MODMA audio pretraining improves MPDD target-domain audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from elderly_depression.models.modma_transfer import (
    build_mpdd_raw_audio_features,
    run_modma_to_mpdd_transfer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "output" / "modma" / "modma_audio_subject_features.csv"
)
DEFAULT_MPDD_AUDIO = (
    PROJECT_ROOT
    / "data"
    / "mpdd"
    / "privacy-constrained-raw-Elder-train"
    / "audio"
)
DEFAULT_MPDD_LABELS = (
    PROJECT_ROOT
    / "data"
    / "mpdd"
    / "Train-MPDD-Elder"
    / "Elder"
    / "split_labels_train.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "modma_transfer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-features", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mpdd-audio", type=Path, default=DEFAULT_MPDD_AUDIO)
    parser.add_argument("--mpdd-labels", type=Path, default=DEFAULT_MPDD_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--source-weight", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source_features.exists():
        raise FileNotFoundError(
            "MODMA subject features not found; run "
            "scripts/65_train_modma_audio_baseline.py first"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.source_features, dtype={"subject_id": str})
    utterance, target, extraction_errors = build_mpdd_raw_audio_features(
        args.mpdd_audio,
        args.mpdd_labels,
    )
    fold_report, comparisons, meta = run_modma_to_mpdd_transfer(
        source,
        target,
        n_splits=args.splits,
        n_repeats=args.repeats,
        source_weight=args.source_weight,
    )
    utterance.drop(columns=["wav_path"], errors="ignore").to_csv(
        args.output_dir / "mpdd_audio_utterance_features.csv",
        index=False,
    )
    target.to_csv(
        args.output_dir / "mpdd_audio_subject_features.csv",
        index=False,
    )
    fold_report.to_csv(
        args.output_dir / "modma_to_mpdd_fold_report.csv",
        index=False,
    )
    comparisons.to_csv(
        args.output_dir / "modma_to_mpdd_comparisons.csv",
        index=False,
    )
    (args.output_dir / "mpdd_audio_extraction_errors.json").write_text(
        json.dumps(extraction_errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = (
        fold_report.groupby("condition")
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            brier_mean=("brier", "mean"),
            ece_mean=("ece", "mean"),
            f1_mean=("f1", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(
        args.output_dir / "modma_to_mpdd_summary.csv",
        index=False,
    )
    (args.output_dir / "modma_to_mpdd_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"target subjects: {len(target)}")
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
