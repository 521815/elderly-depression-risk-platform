"""Experiment: class-balanced segment augmentation, validated on tuning subjects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.config import DATA_DIR
from elderly_depression.data.feature_cache import load_parquet_features
from elderly_depression.data.mpdd import add_mpdd_binary_target
from elderly_depression.multimodal_synthetic_v3 import (
    evaluate_final_holdout,
    generate_segment_augmented,
    split_subject_holdout,
    validate_vs_real,
)


def main() -> None:
    real = load_parquet_features("mpdd_utterance_features")
    if real is None:
        real = pd.read_csv(DATA_DIR / "mpdd_utterance_features.csv")
    real = add_mpdd_binary_target(real).dropna(subset=["mpdd_binary"]).copy()
    tune_df, test_df = split_subject_holdout(real, test_ratio=0.2, seed=42)

    best_path = DATA_DIR / "synthetic" / "augmentation_best_params.json"
    params = json.loads(best_path.read_text(encoding="utf-8"))
    scales = params["modality_noise_scales"]
    aug = generate_segment_augmented(
        tune_df,
        k=params.get("k", 3),
        noise_scale=params.get("noise_scale", 0.03),
        modality_noise_scales=scales,
        balance_classes=True,
    )
    validation = validate_vs_real(aug, tune_df)
    print("Validation feasible:", validation["feasible"])
    print(validation["gates"])

    report = evaluate_final_holdout(
        tune_df,
        test_df,
        k=params.get("k", 3),
        noise_scale=params.get("noise_scale", 0.03),
        modality_noise_scales=scales,
        balance_classes=True,
    )
    out = PROJECT_ROOT / "output" / "synthetic_augmentation_balanced_report.csv"
    report.to_csv(out, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
