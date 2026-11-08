"""Train MPDD single-modality and late-fusion baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.models.multimodal import train_late_fusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "mpdd_features.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output"))
    parser.add_argument("--hamd-threshold", type=int, default=8)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"Missing {input_path}. Apply for MPDD features first and run "
            "scripts/02_build_mpdd_dataset.py."
        )

    df = pd.read_csv(input_path)
    if "hamd_score" not in df.columns:
        raise SystemExit("MPDD table must contain hamd_score.")
    df["hamd_binary"] = (df["hamd_score"] >= args.hamd_threshold).astype(int)

    prefixes = []
    for prefix in ["face", "audio", "gait", "personality"]:
        if any(c.startswith(prefix + "_") for c in df.columns):
            prefixes.append(prefix)
    print(f"Modalities found: {prefixes}")

    result = train_late_fusion(
        df,
        prefixes=prefixes,
        label_col="hamd_binary",
        output_dir=args.output,
    )
    print("\nMultimodal report:")
    print(result["report"].to_string(index=False))
    print(f"\nSaved report: {Path(args.output) / 'multimodal_report.csv'}")


if __name__ == "__main__":
    main()
