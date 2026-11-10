"""Train MPDD single-modality and late-fusion baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.data.mpdd import add_mpdd_binary_target
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

    df = add_mpdd_binary_target(
        pd.read_csv(input_path),
        threshold=args.hamd_threshold,
    )
    df = df.dropna(subset=["mpdd_binary"]).copy()
    if df.empty:
        raise SystemExit("MPDD table has no usable label2 or PHQ-9 score.")

    prefixes = []
    for prefix in ["face", "audio", "gait", "personality"]:
        if any(c.startswith(prefix + "_") for c in df.columns):
            prefixes.append(prefix)
    print(f"Modalities found: {prefixes}")

    result = train_late_fusion(
        df,
        prefixes=prefixes,
        label_col="mpdd_binary",
        output_dir=args.output,
    )
    print("\nMultimodal report:")
    print(result["report"].to_string(index=False))
    print(f"\nSaved report: {Path(args.output) / 'multimodal_report.csv'}")


if __name__ == "__main__":
    main()
