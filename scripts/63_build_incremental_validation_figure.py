"""Build the theory-baseline and modality-ablation figure."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "paper" / "figures" / "scripts"))

from common import COLORS, save, setup_chinese_font, style_axes


def main() -> None:
    root = PROJECT_ROOT / "output" / "incremental_validation"
    charls = pd.read_csv(root / "charls_theory_summary.csv")
    mpdd = pd.read_csv(root / "mpdd_modality_summary.csv")
    mpdd = (
        mpdd[mpdd["modalities"] != "prevalence"]
        .sort_values("auc_mean", ascending=False)
        .head(8)
    )

    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    labels = {
        "B0": "B0\n患病率",
        "B1": "B1\n人口学",
        "B2": "B2\n理论核心",
        "B3": "B3\n理论完整",
        "B3T": "B3T\n+共病轨迹",
        "B4": "B4\n随机森林",
        "B4T": "B4T\n森林+轨迹",
    }
    ax = axes[0]
    x = list(range(len(charls)))
    ax.bar(
        x,
        charls["auc"],
        yerr=[
            charls["auc"] - charls["auc_ci_low"],
            charls["auc_ci_high"] - charls["auc"],
        ],
        capsize=3,
        color=[COLORS[index % len(COLORS)] for index in x],
        alpha=0.88,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(value, value) for value in charls["baseline_id"]])
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("CHARLS 理论基线（按受试者分组 5 折）")
    for index, (_, row) in enumerate(charls.iterrows()):
        ax.text(
            index,
            min(0.985, float(row["auc"]) + 0.035),
            f"{float(row['auc']):.3f}",
            ha="center",
            fontsize=9,
        )
    style_axes(ax)

    ax = axes[1]
    x = list(range(len(mpdd)))
    ax.bar(
        x,
        mpdd["auc_mean"],
        yerr=mpdd["auc_std"],
        capsize=3,
        color=[COLORS[i % len(COLORS)] for i in x],
        alpha=0.88,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            value.replace("personality", "人格")
            .replace("audio", "语音")
            .replace("face", "面部")
            .replace("gait", "步态")
            .replace("+", "\n+")
            for value in mpdd["modalities"]
        ],
        fontsize=8,
    )
    ax.set_ylim(0.45, 0.8)
    ax.set_ylabel("AUC")
    ax.set_title("MPDD 同人级模态组合（5 折 × 5 次）")
    for index, (_, row) in enumerate(mpdd.iterrows()):
        ax.text(
            index,
            min(0.78, float(row["auc_mean"]) + 0.03),
            f"{float(row['auc_mean']):.3f}\nBrier {float(row['brier_mean']):.3f}",
            ha="center",
            fontsize=7.5,
        )
    style_axes(ax)

    fig.tight_layout(w_pad=2.0)
    save(fig, "fig8_incremental_validation.png")
    chapter_dir = PROJECT_ROOT / "paper" / "figures" / "chapters"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / "paper" / "figures" / "fig8_incremental_validation.png"
    (chapter_dir / source.name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
