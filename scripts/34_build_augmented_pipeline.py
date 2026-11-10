"""Build a deployable multimodal subject pipeline trained on augmented segments."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.data.feature_cache import load_parquet_features
from elderly_depression.data.mpdd import add_mpdd_binary_target
from elderly_depression.multimodal_synthetic_v3 import build_augmented_pipeline


def main() -> None:
    real = load_parquet_features("mpdd_utterance_features")
    if real is None:
        real = pd.read_csv(PROJECT_ROOT / "data" / "mpdd_utterance_features.csv")
    real = add_mpdd_binary_target(real).dropna(subset=["mpdd_binary"]).copy()
    out = PROJECT_ROOT / "output" / "multimodal_models" / "subject_fusion_pipeline_augmented.pkl"
    pipeline = build_augmented_pipeline(
        real,
        k=3,
        noise_scale=0.03,
        modality_noise_scales={
            "face_": 0.03,
            "audio_": 0.03,
            "gait_": 0.005,
            "personality_": 0.005,
        },
        output_path=out,
    )
    print("Prefixes:", pipeline["prefixes"])
    print("Fusion keys:", list(pipeline["fusion"].keys()))
    print("Saved:", out)
    print("Size:", out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
