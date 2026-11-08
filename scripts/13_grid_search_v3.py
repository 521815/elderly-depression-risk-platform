"""Grid-search v3 synthetic parameters with marginal calibration and validation gates."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.multimodal_synthetic_v3 import grid_search_v3


def main() -> None:
    result = grid_search_v3()
    print("Grid:")
    print(result["grid"].to_string(index=False))
    print("\nBest:", result["best"])
    print("\nSubject-aware report (best):")
    print(result["report"].to_string(index=False))


if __name__ == "__main__":
    main()
