"""A/B comparison: before vs after improvements on fixed evaluation sets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.features.text_features import summarize_dialogue
from elderly_depression.models.llm_assessor import DialogueAssessor


def _dialogue_ab() -> dict:
    path = PROJECT_ROOT / "data" / "synthetic" / "dialogue_samples.csv"
    df = pd.read_csv(path)
    assessor = DialogueAssessor()
    old_risks = []
    new_risks = []
    labels = []
    probs = []
    for _, row in df.iterrows():
        transcript = json.loads(row["transcript_json"])
        text_df = summarize_dialogue(transcript)
        cues = assessor._extract_language_cues(transcript)
        old_risks.append(assessor._text_risk(text_df, []))
        new_risks.append(assessor._text_risk(text_df, cues))
        labels.append(1 if row["risk_level"] == "high" else 0)
        probs.append(float(row["depression_probability"]))
    old_auc = float(roc_auc_score(labels, old_risks))
    new_auc = float(roc_auc_score(labels, new_risks))
    old_corr = float(spearmanr(old_risks, probs).statistic)
    new_corr = float(spearmanr(new_risks, probs).statistic)
    return {
        "old_text_auc": round(old_auc, 4),
        "new_text_auc": round(new_auc, 4),
        "old_prob_corr": round(old_corr, 4),
        "new_prob_corr": round(new_corr, 4),
    }


def main() -> None:
    tabular = pd.read_csv(PROJECT_ROOT / "output" / "synthetic_comparison_v2_report.csv")
    multimodal = pd.read_csv(PROJECT_ROOT / "output" / "synthetic_augmentation_report.csv")
    dialogue = _dialogue_ab()

    tabular = tabular[tabular["model"].isin(["logistic", "random_forest", "xgboost"])]
    tab_summary = (
        tabular.groupby("source")[["auc", "f1"]]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"auc": "mean_auc", "f1": "mean_f1"})
    )
    mm_summary = (
        multimodal.groupby("source")[["auc_mean", "f1_mean"]]
        .mean()
        .round(4)
        .reset_index()
    )

    summary = {
        "tabular": tab_summary.to_dict(orient="records"),
        "multimodal": mm_summary.to_dict(orient="records"),
        "dialogue": dialogue,
        "note": (
            "表格/多模态：同一真实评估协议下，真实训练 vs 模拟组训练；"
            "对话：固定模拟标签集下，纯规则 vs 规则+LLM 线索。"
        ),
    }
    out = PROJECT_ROOT / "output" / "ab_comparison_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
