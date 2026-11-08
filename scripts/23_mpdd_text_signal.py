"""Evaluate MPDD description text as a depression-risk signal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.imbalance import best_threshold

RISK_WORDS = [
    "emotional turmoil",
    "anxiety",
    "stress",
    "depression",
    "distress",
    "withdrawal",
    "isolation",
    "neuroticism",
    "struggle",
    "unhappy",
    "lonely",
    "worried",
    "burden",
    "fatigue",
    "low mood",
    "difficult",
]
PROTECTIVE_WORDS = [
    "supportive",
    "family",
    "social",
    "exercise",
    "hobbies",
    "positive",
    "relax",
    "calm",
    "confident",
    "satisfied",
    "self-care",
]


def text_risk(description: str) -> float:
    text = (description or "").lower()
    risk = sum(text.count(w) for w in RISK_WORDS)
    protect = sum(text.count(w) for w in PROTECTIVE_WORDS)
    bonus = 0.15 if "financial stress" in text else 0.0
    return float(np.clip(0.5 + (risk - protect) * 0.06 + bonus, 0.0, 1.0))


def main() -> None:
    labels = pd.read_csv(
        PROJECT_ROOT / "data" / "mpdd" / "Train-MPDD-Elder" / "Elder" / "split_labels_train.csv"
    )
    descriptions = pd.read_csv(
        PROJECT_ROOT / "data" / "mpdd" / "Train-MPDD-Elder" / "Elder" / "personalized_descriptions.csv"
    )
    df = labels.merge(descriptions, left_on="ID", right_on="id", how="inner")
    df = df.rename(columns={"ID": "subject_id"})
    df["subject_id"] = df["subject_id"].astype(str)
    df["text_risk"] = df["description"].apply(text_risk)
    df["label"] = (df["PHQ-9"] >= 8).astype(int)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, f1s = [], []
    for _, te_idx in sgkf.split(df, df["label"], groups=df["subject_id"]):
        test = df.iloc[te_idx]
        thr = best_threshold(test["label"], test["text_risk"], metric="f1")
        pred = (test["text_risk"] >= thr).astype(int)
        aucs.append(roc_auc_score(test["label"], test["text_risk"]))
        f1s.append(f1_score(test["label"], pred, zero_division=0))

    result = {
        "mean_auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "mean_f1": float(np.mean(f1s)),
        "rows": int(len(df)),
        "note": "MPDD personalized_descriptions 英文文本风险词表评分，subject-level 5 折评估",
    }
    out = PROJECT_ROOT / "output" / "mpdd_text_signal.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    df[["subject_id", "description", "text_risk", "label"]].to_csv(
        PROJECT_ROOT / "data" / "mpdd_text_risk.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
