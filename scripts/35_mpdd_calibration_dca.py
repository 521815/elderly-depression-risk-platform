"""Calibration, Brier and decision curve on held-out MPDD test subjects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.data.feature_cache import load_parquet_features
from elderly_depression.models.multimodal import predict_subject_pipeline
from elderly_depression.multimodal_synthetic_v3 import (
    build_augmented_pipeline,
    split_subject_holdout,
)


def _calibration_rows(y, prob):
    rows = []
    bins = np.linspace(0, 1, 11)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob < hi)
        if i == len(bins) - 2:
            mask = (prob >= lo) & (prob <= hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_low": round(float(lo), 3),
                "bin_high": round(float(hi), 3),
                "count": int(mask.sum()),
                "predicted_freq": round(float(prob[mask].mean()), 4),
                "observed_freq": round(float(y[mask].mean()), 4),
            }
        )
    return rows


def _dca_rows(y, prob):
    rows = []
    prevalence = float(y.mean())
    for t in np.linspace(0.05, 0.95, 19):
        pred = (prob >= t).astype(int)
        tp = float(((pred == 1) & (y == 1)).sum() / len(y))
        fp = float(((pred == 1) & (y == 0)).sum() / len(y))
        rows.append(
            {
                "threshold": round(float(t), 3),
                "net_benefit": round(float(tp - fp * (t / (1 - t))), 4),
                "treat_all_benefit": round(
                    float(prevalence - (1 - prevalence) * (t / (1 - t))), 4
                ),
            }
        )
    return rows


def main() -> None:
    real = load_parquet_features("mpdd_utterance_features")
    if real is None:
        real = pd.read_csv(PROJECT_ROOT / "data" / "mpdd_utterance_features.csv")
    real = real.dropna(subset=["hamd_score"]).copy()
    tune, test = split_subject_holdout(real, test_ratio=0.2, seed=42)
    y = test.groupby("subject_id")["hamd_score"].apply(lambda s: int((s >= 8).any()))

    result = {
        "tune_subjects": int(tune["subject_id"].nunique()),
        "test_subjects": int(len(y)),
        "pipelines": {},
    }
    pipelines = {
        "real": build_augmented_pipeline(tune, augment=False),
        "augmented": build_augmented_pipeline(
            tune,
            k=3,
            noise_scale=0.03,
            modality_noise_scales={
                "face_": 0.03,
                "audio_": 0.03,
                "gait_": 0.005,
                "personality_": 0.005,
            },
        ),
    }
    for name, pipeline in pipelines.items():
        pred = predict_subject_pipeline(pipeline, test).set_index("subject_id")
        prob = pred["calibrated_prob"].reindex(y.index)
        mask = y.notna() & prob.notna()
        yy, pp = y[mask].to_numpy(), prob[mask].to_numpy()
        result["pipelines"][name] = {
            "auc": round(float(roc_auc_score(yy, pp)), 4),
            "brier": round(float(brier_score_loss(yy, pp)), 4),
            "calibration": _calibration_rows(yy, pp),
            "decision_curve": _dca_rows(yy, pp),
        }
    out = PROJECT_ROOT / "output" / "mpdd_calibration_dca.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:3000])
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
