"""Learn and calibrate the context/background risk component from real CHARLS."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.imbalance import best_threshold
from elderly_depression.models.llm_assessor import DialogueAssessor


def main() -> None:
    assessor = DialogueAssessor()
    df = pd.read_csv(PROJECT_ROOT / "data" / "charls_standardized.csv").dropna(
        subset=["depression"]
    )
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
                "self_rated_health": (
                    r["self_rated_health"] if pd.notna(r["self_rated_health"]) else None
                ),
            }
        ),
        axis=1,
    )
    y = df["depression"].astype(int)
    X = df[["context_risk"]].fillna(0.5)
    ids = df["id"].astype(str)

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(split.split(X, y, groups=ids))
    split_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr2_idx, va_idx = next(
        split_val.split(X.iloc[tr_idx], y.iloc[tr_idx], groups=ids.iloc[tr_idx])
    )
    X_tr, X_va, X_te = X.iloc[tr_idx].iloc[tr2_idx], X.iloc[tr_idx].iloc[va_idx], X.iloc[te_idx]
    y_tr, y_va, y_te = y.iloc[tr_idx].iloc[tr2_idx], y.iloc[tr_idx].iloc[va_idx], y.iloc[te_idx]

    model = LogisticRegression(max_iter=2000)
    model.fit(X_tr, y_tr)
    va_prob = model.predict_proba(X_va)[:, 1]
    te_prob = model.predict_proba(X_te)[:, 1]
    threshold = best_threshold(y_va, va_prob, metric="f1")
    pred = (te_prob >= threshold).astype(int)

    result = {
        "intercept": float(model.intercept_[0]),
        "coefficient": float(model.coef_[0][0]),
        "auc": float(roc_auc_score(y_te, te_prob)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_te, te_prob)),
        "threshold": float(threshold),
        "train_subjects": int(ids.iloc[tr_idx].nunique()),
        "test_subjects": int(ids.iloc[te_idx].nunique()),
        "note": "context_risk 启发式分数的真实校准；背景特征与 CESD 标签不同源，无循环问题。",
    }
    out = PROJECT_ROOT / "output" / "context_component_learning.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
