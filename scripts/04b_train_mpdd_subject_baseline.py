"""Train subject-aware utterance-level MPDD baseline with repeated group CV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.data.mpdd import build_utterance_feature_table
from elderly_depression.models.multimodal import train_subject_late_fusion
from elderly_depression.smoothing import aggregate_mpdd_subjects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utterance-csv", default=str(PROJECT_ROOT / "data" / "mpdd_utterance_features.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output"))
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--pca-dims", type=int, default=64)
    parser.add_argument("--train", default="real", choices=["real", "synthetic"])
    parser.add_argument(
        "--subject-aggregation",
        default="none",
        choices=["none", "mean", "median", "trimmed_mean"],
        help="none keeps utterance-level rows; other modes aggregate features per subject.",
    )
    args = parser.parse_args()

    if args.train == "synthetic":
        from elderly_depression.multimodal_synthetic_v3 import run_augmentation_pipeline

        result = run_augmentation_pipeline(noise_scale=0.03, k=3)
        print("Synthetic-train / real-eval validation feasible:", result["validation"]["feasible"])
        print(result["report"].to_string(index=False))
        return

    utterance_csv = Path(args.utterance_csv)
    if not utterance_csv.exists():
        print("Building utterance feature table first...")
        build_utterance_feature_table(output_path=utterance_csv)

    if args.subject_aggregation == "none":
        df = pd.read_csv(utterance_csv).dropna(subset=["hamd_score"])
        df["hamd_binary"] = (df["hamd_score"] >= args.threshold).astype(int)
        label_col = "hamd_binary"
    else:
        df = aggregate_mpdd_subjects(
            utterance_csv,
            method=args.subject_aggregation,
            threshold=args.threshold,
        )
        label_col = "_label"
    prefixes = []
    for prefix in ["face", "audio", "gait", "personality"]:
        if any(c.startswith(prefix + "_") for c in df.columns):
            prefixes.append(prefix)
    print(f"Rows: {len(df)}, subjects: {df['subject_id'].nunique()}, modalities: {prefixes}")

    result = train_subject_late_fusion(
        df,
        prefixes=prefixes,
        label_col=label_col,
        output_dir=args.output,
        n_splits=args.splits,
        n_repeats=args.repeats,
        pca_dims=args.pca_dims,
    )
    print("\nSubject-level summary (mean over repeated CV folds):")
    print(result["summary"].to_string(index=False))
    print(f"\nSaved: {Path(args.output) / 'multimodal_subject_report.csv'}")


if __name__ == "__main__":
    main()
