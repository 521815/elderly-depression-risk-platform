"""Brier score and reliability curve for the CHARLS tabular model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
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

    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    brier = float(brier_score_loss(y_te, prob))

    bins = np.linspace(0, 1, 11)
    rows = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob < hi)
        if i == len(bins) - 2:
            mask = (prob >= lo) & (prob <= hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_low": round(float(lo), 3),
                "bin_high": round(float(hi), 3),
                "count": int(mask.sum()),
                "predicted_freq": round(float(prob[mask].mean()), 4),
                "observed_freq": round(float(y_te.to_numpy()[mask].mean()), 4),
            }
        )
    curve = pd.DataFrame(rows)
    curve.to_csv(
        PROJECT_ROOT / "output" / "charls_calibration_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result = {
        "brier": round(brier, 4),
        "test_subjects": int(ids.iloc[te_idx].nunique()),
        "note": "逻辑回归概率的 Brier 与可靠性曲线，person-level holdout",
    }
    out = PROJECT_ROOT / "output" / "charls_calibration.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
