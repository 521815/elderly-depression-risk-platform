"""Grid over gait augmentation noise to close the gait AUC gap."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.multimodal_synthetic_v3 import run_augmentation_pipeline


def main() -> None:
    rows = []
    for gait_scale in [0.005, 0.01, 0.05, 0.08]:
        scales = {
            "face_": 0.03,
            "audio_": 0.03,
            "gait_": gait_scale,
            "personality_": 0.03,
        }
        result = run_augmentation_pipeline(
            noise_scale=0.03,
            k=3,
            modality_noise_scales=scales,
        )
        report = result["report"]
        syn = report[report["source"] == "synthetic"].set_index("model")
        real = report[report["source"] == "real"].set_index("model")
        rows.append(
            {
                "gait_noise_scale": gait_scale,
                "gait_auc": float(syn.loc["gait", "auc_mean"]),
                "gait_f1": float(syn.loc["gait", "f1_mean"]),
                "fusion_auc": float(syn.loc["fusion", "auc_mean"]),
                "face_auc": float(syn.loc["face", "auc_mean"]),
                "audio_auc": float(syn.loc["audio", "auc_mean"]),
                "personality_auc": float(syn.loc["personality", "auc_mean"]),
                "mean_auc_delta": float(syn["auc_mean"].mean() - real["auc_mean"].mean()),
                "validation_feasible": result["validation"]["feasible"],
            }
        )
        print(rows[-1])
    grid = pd.DataFrame(rows)
    out = PROJECT_ROOT / "output" / "gait_augmentation_grid.csv"
    grid.to_csv(out, index=False, encoding="utf-8-sig")
    print("\nGrid saved:", out)
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
