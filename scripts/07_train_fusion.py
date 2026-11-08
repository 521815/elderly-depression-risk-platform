"""Train the learned fusion layer from a CSV of signal probabilities and labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.services import FUSION_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with signal probabilities and label")
    parser.add_argument("--label", default="depression")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    from elderly_depression.models.fusion import train_fusion

    result = train_fusion(df, label_col=args.label, output_dir=FUSION_DIR)
    print(result["report"].to_string(index=False))
    print(f"\nSaved weights: {FUSION_DIR / 'fusion_weights.json'}")


if __name__ == "__main__":
    main()
