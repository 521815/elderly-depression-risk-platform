"""Fairness-style subgroup report for the CHARLS tabular model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.imbalance import best_threshold
from elderly_depression.models.tabular import prepare_X


def main() -> None:
    df = pd.read_csv(PROJECT_ROOT / "data" / "charls_standardized.csv").dropna(
        subset=["depression"]
    )
    exclude = {"id", "wave", "cesd_score", "depression"}
    feature_cols = [c for c in df.columns if c not in exclude and c != "depression"]
    y = df["depression"].astype(int)
    X = prepare_X(df, feature_cols).loc[y.index]
    ids = df["id"].astype(str)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(split.split(X, y, groups=ids))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    threshold = best_threshold(y_tr, model.predict_proba(X_tr)[:, 1], metric="f1")

    test = df.iloc[te_idx].copy()
    test["prob"] = prob
    test["pred"] = (prob >= threshold).astype(int)
    test["age_band"] = np.where(test["age"] < 70, "60-69", np.where(test["age"] < 80, "70-79", "80+"))
    test["gender"] = np.where(test["gender"] == 1, "女", "男")
    test["comorbidity_band"] = np.where(
        test["comorbidity_count"] == 0, "0", np.where(test["comorbidity_count"] <= 2, "1-2", "3+")
    )

    rows = []
    for group_col in ["age_band", "gender", "comorbidity_band"]:
        for value, sub in test.groupby(group_col):
            if sub["depression"].nunique() < 2 or len(sub) < 20:
                continue
            rows.append(
                {
                    "group": group_col,
                    "value": str(value),
                    "n": int(len(sub)),
                    "positive_rate": round(float(sub["depression"].mean()), 4),
                    "auc": round(float(roc_auc_score(sub["depression"], sub["prob"])), 4),
                    "f1": round(float(f1_score(sub["depression"], sub["pred"], zero_division=0)), 4),
                }
            )
    report = pd.DataFrame(rows)
    report.to_csv(
        PROJECT_ROOT / "output" / "charls_subgroup_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(report.to_string(index=False))
    print("\nSaved:", PROJECT_ROOT / "output" / "charls_subgroup_report.csv")


if __name__ == "__main__":
    main()
