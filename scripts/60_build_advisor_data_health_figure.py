"""Build a current data-health figure for the advisor report."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "paper" / "figures" / "scripts"))

from common import COLORS, setup_chinese_font, style_axes


OUTPUT = PROJECT_ROOT / "paper" / "figures" / "chapters" / "fig_data_health.png"


def rounded_box(ax, x, y, width, height, color, alpha=0.1):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=0.8,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)


def metric_card(ax, x, value, label, color):
    rounded_box(ax, x, 0.73, 0.21, 0.16, color, 0.09)
    ax.text(x + 0.105, 0.825, value, ha="center", va="center", fontsize=18, weight="bold", color=color)
    ax.text(x + 0.105, 0.77, label, ha="center", va="center", fontsize=9.5, color="#374151")


def main() -> None:
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.96, "数据健康与 MPDD 标签审计", fontsize=16, weight="bold", color="#0f2744")
    ax.text(0.02, 0.915, "2026-11-10 正式口径", fontsize=9.5, color="#667085")

    metric_card(ax, 0.02, "0", "阻断问题", COLORS[0])
    metric_card(ax, 0.265, "7", "风险提示", COLORS[1])
    metric_card(ax, 0.51, "87", "官方标签受试者", COLORS[2])
    metric_card(ax, 0.755, "11", "标签口径分歧", COLORS[3])

    rounded_box(ax, 0.02, 0.35, 0.46, 0.31, COLORS[0], 0.055)
    ax.text(0.05, 0.61, "CHARLS 老年表格数据", fontsize=12, weight="bold", color="#17324d")
    charls_rows = [
        ("样本量", "32,936 条"),
        ("整体缺失率", "5.5%"),
        ("抑郁标签阳性率", "36.8%"),
        ("共病均值", "2.07 种"),
    ]
    for index, (label, value) in enumerate(charls_rows):
        y = 0.55 - index * 0.056
        ax.text(0.05, y, label, fontsize=9.5, color="#52606d")
        ax.text(0.45, y, value, fontsize=10, weight="bold", color="#17324d", ha="right")

    rounded_box(ax, 0.51, 0.27, 0.47, 0.39, COLORS[2], 0.055)
    ax.text(0.54, 0.61, "MPDD 老年多模态数据", fontsize=12, weight="bold", color="#17324d")
    mpdd_rows = [
        ("受试者", "110 人"),
        ("训练 / 测试", "87 / 23 人"),
        ("整体缺失率", "1.4%"),
        ("官方 label2 阳性", "28 / 87（32.2%）"),
        ("PHQ-9 ≥8 阳性", "17 / 87"),
    ]
    for index, (label, value) in enumerate(mpdd_rows):
        y = 0.55 - index * 0.052
        ax.text(0.54, y, label, fontsize=9.5, color="#52606d")
        ax.text(0.94, y, value, fontsize=9.8, weight="bold", color="#17324d", ha="right")

    ax.text(
        0.02,
        0.18,
        "正式训练、验证与模拟组统一优先使用官方 label2；仅在缺失时回退 PHQ-9 阈值。",
        fontsize=9.4,
        color="#344054",
    )
    ax.text(
        0.02,
        0.125,
        "hamd_score 是 PHQ-9 兼容别名，不得解释为 HAMD-24。",
        fontsize=9.4,
        color="#9a3412",
    )
    ax.text(
        0.02,
        0.065,
        "数据健康页可按当前结果重新生成；本图仅用于科研数据质量检查，不替代临床判断。",
        fontsize=8.5,
        color="#667085",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
