"""Feature importance and local explanations for the CHARLS tabular model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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

    pipeline = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    pipeline.fit(X_tr, y_tr)
    coef = pipeline.named_steps["logisticregression"].coef_[0]
    importance = permutation_importance(
        pipeline, X_te, y_te, n_repeats=10, random_state=42, n_jobs=1
    )
    rows = pd.DataFrame(
        {
            "feature": X.columns,
            "logistic_coef": coef,
            "permutation_importance_mean": importance.importances_mean,
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    rows.to_csv(
        PROJECT_ROOT / "output" / "charls_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sample = X_te.iloc[[0]]
    scaled = pipeline.named_steps["standardscaler"].transform(sample)
    contributions = scaled[0] * coef
    local = pd.DataFrame({"feature": X.columns, "contribution": contributions}).sort_values(
        "contribution", key=lambda s: s.abs(), ascending=False
    ).head(10).to_dict(orient="records")
    result = {
        "top_features": rows.head(15).to_dict(orient="records"),
        "local_explanation_sample": local,
        "note": "置换重要性 + 逻辑回归系数；未安装 SHAP，故不称 SHAP。",
    }
    out = PROJECT_ROOT / "output" / "charls_interpretability.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:3000])
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
