"""Train person-level longitudinal baselines with previous-wave lag features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.models.tabular import train_longitudinal_baseline

EXCLUDE = {
    "id",
    "wave",
    "cesd_score",
    "depression",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "charls_standardized.csv"))
    parser.add_argument("--resample", default="smote_tomek")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    feature_cols = [c for c in df.columns if c not in EXCLUDE and c != "depression"]
    result = train_longitudinal_baseline(
        df,
        feature_cols=feature_cols,
        label_col="depression",
        resample_method=args.resample,
        output_dir=args.output,
    )
    print("\nLongitudinal trajectory report:")
    print(result["report"].to_string(index=False))


if __name__ == "__main__":
    main()
