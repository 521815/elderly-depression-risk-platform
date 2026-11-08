"""Grid over multimodal augmentation copies k, compare subject-aware metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.multimodal_synthetic_v3 import run_augmentation_pipeline


def main() -> None:
    rows = []
    for k in [2, 3, 4, 5, 6]:
        result = run_augmentation_pipeline(noise_scale=0.03, k=k)
        report = result["report"]
        syn = report[report["source"] == "synthetic"]
        real = report[report["source"] == "real"]
        syn_by_model = syn.set_index("model")
        real_mean = real["auc_mean"].mean()
        rows.append(
            {
                "k": k,
                "validation_feasible": result["validation"]["feasible"],
                "mean_auc_synthetic": round(float(syn["auc_mean"].mean()), 4),
                "mean_f1_synthetic": round(float(syn["f1_mean"].mean()), 4),
                "mean_auc_delta": round(float(syn["auc_mean"].mean() - real_mean), 4),
                "fusion_auc": round(float(syn_by_model.loc["fusion", "auc_mean"]), 4),
                "face_auc": round(float(syn_by_model.loc["face", "auc_mean"]), 4),
                "audio_auc": round(float(syn_by_model.loc["audio", "auc_mean"]), 4),
                "gait_auc": round(float(syn_by_model.loc["gait", "auc_mean"]), 4),
                "personality_auc": round(float(syn_by_model.loc["personality", "auc_mean"]), 4),
            }
        )
        print(rows[-1])

    grid = pd.DataFrame(rows)
    grid.to_csv(
        PROJECT_ROOT / "output" / "multimodal_k_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feasible = grid[grid["validation_feasible"]]
    best = (
        feasible.loc[feasible["mean_auc_delta"].idxmax()]
        if not feasible.empty
        else grid.loc[grid["mean_auc_delta"].idxmax()]
    )
    print("\nBest k:", int(best["k"]), best.to_dict())

    # Restore official augmentation files with the best k.
    run_augmentation_pipeline(noise_scale=0.03, k=int(best["k"]))
    print("\nSaved:", PROJECT_ROOT / "output" / "multimodal_k_grid.csv")


if __name__ == "__main__":
    main()
