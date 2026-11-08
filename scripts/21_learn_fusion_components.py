"""Learn fusion component weights from real data without waiting for a full joint set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.models.llm_assessor import DialogueAssessor


def main() -> None:
    template_path = PROJECT_ROOT / "data" / "fusion_training_template.csv"
    assessor = DialogueAssessor()
    df = pd.read_csv(PROJECT_ROOT / "data" / "charls_standardized.csv").dropna(
        subset=["depression"]
    )
    df["scale_risk"] = df["cesd_score"] / 30.0 * 1.15
    df["context_risk"] = df.apply(
        lambda r: assessor._context_risk(
            {
                "age": r["age"],
                "gender": "female" if r["gender"] == 1 else "male",
                "married": r["marital_status"] != 7,
                "comorbidity_count": r["comorbidity_count"],
                "sleep_quality": (
                    r["sleep_quality"] if pd.notna(r["sleep_quality"]) else None
                ),
            }
        ),
        axis=1,
    )
    # The depression label is derived from CESD-10 itself, so scale_risk is a
    # deterministic transform of the label. Reporting AUC here would be circular.
    scale_label_corr = float(df[["scale_risk", "depression"]].corr().iloc[0, 1])

    mpdd_report = None
    mpdd_path = PROJECT_ROOT / "output" / "multimodal_subject_report.csv"
    if mpdd_path.exists():
        mpdd_report = pd.read_csv(mpdd_path)
        mpdd_fusion = mpdd_report[mpdd_report["modality"] == "fusion"]
        if not mpdd_fusion.empty:
            mpdd_summary = {
                "auc_mean": float(mpdd_fusion["auc_mean"].iloc[0]),
                "f1_mean": float(mpdd_fusion["f1_mean"].iloc[0]),
            }
        else:
            mpdd_summary = None
    else:
        mpdd_summary = None

    result = {
        "charls_scale_context_leakage": {
            "scale_risk_label_corr": round(scale_label_corr, 4),
            "usable": False,
            "reason": (
                "CHARLS 的 depression 标签由 CESD-10 得分阈值得到，"
                "scale_risk 与标签同源，不能用它学习融合权重，否则 AUC=1.0 是定义性假结果。"
            ),
        },
        "mpdd_signal": mpdd_summary,
        "fusion_training_template": str(template_path),
        "note": (
            "完整联合融合权重需要一份融合训练集：每条记录同时含 text_risk / scale_risk / context_risk / "
            "expression_risk / mpdd_risk 和独立标注的 depression 标签；模板已生成。"
            "MPDD 用现有真实 subject-aware 报告作为多模态信号质量。"
        ),
    }
    out = PROJECT_ROOT / "output" / "fusion_component_learning.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame(
        columns=[
            "text_risk",
            "scale_risk",
            "context_risk",
            "expression_risk",
            "mpdd_risk",
            "depression",
        ]
    ).to_csv(template_path, index=False, encoding="utf-8-sig")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
