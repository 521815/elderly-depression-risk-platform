"""Scan MPDD feature files and merge them into one table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.data.mpdd import build_feature_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpdd-root", default=None, help="Path to MPDD feature directory")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    df = build_feature_table(mpdd_root=args.mpdd_root, output_path=args.output)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Label columns: {[c for c in df.columns if 'hamd' in c.lower()]}")
    print(f"Output: {args.output or PROJECT_ROOT / 'data' / 'mpdd_features.csv'}")


if __name__ == "__main__":
    main()
