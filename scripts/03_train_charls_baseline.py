"""Train tabular depression-risk baselines on the standardized CHARLS table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.models.tabular import train_baseline

EXCLUDE = {
    "id",
    "wave",
    "cesd_score",
    "depression",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "charls_standardized.csv"))
    parser.add_argument("--resample", default="smote_tomek", choices=["none", "smote", "smote_enn", "smote_tomek", "adasyn"])
    parser.add_argument("--train", default="real", choices=["real", "synthetic"])
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output"))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"Missing {input_path}. Run scripts/01_build_charls_dataset.py first."
        )

    df = pd.read_csv(input_path)
    feature_cols = [c for c in df.columns if c not in EXCLUDE and c != "depression"]
    if args.train == "synthetic":
        from elderly_depression.models.tabular import train_tabular_from_synthetic

        synthetic_path = PROJECT_ROOT / "data" / "synthetic" / "charls_synthetic.csv"
        if not synthetic_path.exists():
            raise SystemExit(f"Missing {synthetic_path}. Run scripts/08_build_feature_cache.py or generate synthetic data first.")
        synthetic_df = pd.read_csv(synthetic_path)
        result = train_tabular_from_synthetic(
            df,
            synthetic_df,
            feature_cols=feature_cols,
            label_col="depression",
            output_dir=args.output,
        )
        print("\nSynthetic-train / real-eval report:")
        print(result["report"].to_string(index=False))
        print(f"\nSaved report: {Path(args.output) / 'tabular_synthetic_report.csv'}")
        return
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features.")
    print(f"Feature list: {feature_cols}")

    result = train_baseline(
        df,
        feature_cols=feature_cols,
        label_col="depression",
        resample_method=args.resample,
        output_dir=args.output,
    )
    print("\nTabular baseline report:")
    print(result["report"].to_string(index=False))
    print(f"\nSaved report: {Path(args.output) / 'tabular_model_report.csv'}")


if __name__ == "__main__":
    main()
