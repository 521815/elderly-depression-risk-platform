#!/usr/bin/env python3
"""Train a subject-grouped acoustic baseline on the MODMA auxiliary cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elderly_depression.data.modma import (
    build_modma_subject_index,
    materialize_modma_audio_root,
)
from elderly_depression.features.modma_audio import (
    aggregate_modma_subject_features,
    build_modma_audio_features,
    train_modma_subject_baseline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "modma"
DEFAULT_METADATA = (
    DEFAULT_ROOT / "metadata" / "subjects_information_audio_lanzhou_2015.xlsx"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "modma"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with materialize_modma_audio_root(args.root) as (audio_root, _):
        subject_index, audit = build_modma_subject_index(
            audio_root,
            args.metadata,
        )
        utterance, errors = build_modma_audio_features(
            audio_root,
            subject_index,
        )
    subject_features = aggregate_modma_subject_features(utterance)
    fold_report, summary = train_modma_subject_baseline(
        subject_features,
        n_splits=args.splits,
        n_repeats=args.repeats,
    )

    utterance_output = utterance.drop(columns=["wav_path"], errors="ignore")
    utterance_output.to_csv(
        args.output_dir / "modma_audio_utterance_features.csv",
        index=False,
    )
    subject_features.to_csv(
        args.output_dir / "modma_audio_subject_features.csv",
        index=False,
    )
    fold_report.to_csv(
        args.output_dir / "modma_audio_fold_report.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir / "modma_audio_summary.csv",
        index=False,
    )
    (args.output_dir / "modma_audio_extraction_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "modma_audio_training_meta.json").write_text(
        json.dumps(
            {
                "source": "MODMA",
                "role": "auxiliary_cross_age_pretraining_and_robustness",
                "is_older_adult_primary_dataset": False,
                "subjects": int(len(subject_features)),
                "readable_segments": int(len(utterance)),
                "feature_extractor": "numpy_scipy_mfcc_spectral_v1",
                "model": "standard_scaler_l2_logistic_C0.1",
                "threshold": 0.5,
                "splits": args.splits,
                "repeats": args.repeats,
                "grouping": "subject_id",
                "audio_audit_complete": audit["audio"]["complete_dataset"],
                "warnings": audit["warnings"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"utterance features: {len(utterance)}")
    print(f"subjects: {len(subject_features)}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
