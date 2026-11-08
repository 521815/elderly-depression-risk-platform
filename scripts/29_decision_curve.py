"""Decision curve analysis for the CHARLS tabular model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    prob = model.predict_proba(X.iloc[te_idx])[:, 1]
    y_te = y.iloc[te_idx].to_numpy()

    prevalence = float(y_te.mean())
    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = (prob >= t).astype(int)
        tp = float(((pred == 1) & (y_te == 1)).sum() / len(y_te))
        fp = float(((pred == 1) & (y_te == 0)).sum() / len(y_te))
        nb = tp - fp * (t / (1 - t))
        rows.append(
            {
                "threshold": round(float(t), 3),
                "net_benefit": round(float(nb), 4),
                "treat_all_benefit": round(prevalence - (1 - prevalence) * (t / (1 - t)), 4),
            }
        )
    curve = pd.DataFrame(rows)
    curve.to_csv(
        PROJECT_ROOT / "output" / "charls_decision_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(curve.head(20).to_string(index=False))
    print("\nSaved:", PROJECT_ROOT / "output" / "charls_decision_curve.csv")


if __name__ == "__main__":
    main()
