"""Build compact Parquet feature cache and MPDD file manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.data.feature_cache import build_feature_cache, cache_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild CSVs even if present")
    args = parser.parse_args()

    report = build_feature_cache(force=args.force)
    print("Before:", report["before"])
    print("After:", report["after"])
    print("Manifest fresh:", report["manifest_fresh"])
    print("Status:", cache_status())


if __name__ == "__main__":
    main()
