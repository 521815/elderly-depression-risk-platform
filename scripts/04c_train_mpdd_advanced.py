"""Train label3 classification and PHQ-9 regression on MPDD utterances."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.data.mpdd import (
    add_mpdd_binary_target,
    build_utterance_feature_table,
)
from elderly_depression.models.multimodal import train_subject_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utterance-csv", default=str(PROJECT_ROOT / "data" / "mpdd_utterance_features.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    utterance_csv = Path(args.utterance_csv)
    if not utterance_csv.exists():
        build_utterance_feature_table(output_path=utterance_csv)
    df = add_mpdd_binary_target(pd.read_csv(utterance_csv))
    prefixes = ["face", "audio", "gait", "personality"]

    label3 = train_subject_target(
        df,
        prefixes=prefixes,
        target_col="label3",
        task="class",
        output_dir=args.output,
        n_splits=args.splits,
        n_repeats=args.repeats,
    )
    phq9 = train_subject_target(
        df,
        prefixes=prefixes,
        target_col="phq9_score",
        task="regression",
        output_dir=args.output,
        n_splits=args.splits,
        n_repeats=args.repeats,
    )
    print("label3:")
    print(label3.to_string(index=False))
    print("\nPHQ-9:")
    print(phq9.to_string(index=False))


if __name__ == "__main__":
    main()
