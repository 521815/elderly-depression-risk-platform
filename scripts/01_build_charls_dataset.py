"""Build a standardized CHARLS elderly risk table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.config import CESD_THRESHOLD, MIN_AGE
from elderly_depression.data.charls import build_standardized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charls-root", default=None, help="Path to CHARLS folder")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--min-age", type=int, default=MIN_AGE)
    args = parser.parse_args()

    df = build_standardized(
        charls_root=args.charls_root,
        min_age=args.min_age,
        output_path=args.output,
    )
    pos = int(df["depression"].sum())
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Depression risk (CESD-10 >= {CESD_THRESHOLD}): {pos} ({pos / len(df):.2%})")
    print(f"Output: {args.output or PROJECT_ROOT / 'data' / 'charls_standardized.csv'}")
    print(f"Sample columns: {list(df.columns[:20])}")


if __name__ == "__main__":
    main()
