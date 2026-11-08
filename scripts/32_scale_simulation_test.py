"""Generate larger simulated groups and re-run validation and A/B tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression import services
from elderly_depression.multimodal_synthetic_v3 import run_augmentation_pipeline


def main() -> None:
    generated = services.generate_synthetic(
        dialogue_n=500,
        tabular_n=1000,
        multimodal_n=300,
        seed=42,
    )
    dialogue_validation = generated["dialogue"]["validation"]
    tabular_report = services.train_tabular_synthetic()
    tabular_summary = (
        tabular_report.groupby("source")[["auc", "f1"]]
        .mean()
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )
    multimodal = run_augmentation_pipeline(noise_scale=0.03, k=5)
    mm_report = multimodal["report"]
    mm_summary = (
        mm_report.groupby("source")[["auc_mean", "f1_mean"]]
        .mean()
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )

    result = {
        "dialogue": {
            "rows": int(generated["dialogue"]["rows"]),
            "validation": dialogue_validation,
        },
        "tabular": {
            "rows": int(len(__import__("pandas").read_csv(PROJECT_ROOT / "data" / "synthetic" / "charls_synthetic.csv"))),
            "summary": tabular_summary,
        },
        "multimodal_augmentation": {
            "k": 5,
            "validation_feasible": multimodal["validation"]["feasible"],
            "summary": mm_summary,
        },
        "note": "更大规模模拟组测试：对话 500、表格 2000、多模态增强 k=5",
    }
    out = PROJECT_ROOT / "output" / "scale_simulation_test.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
