"""Bootstrap confidence intervals for CHARLS test AUC and Brier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
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
    subjects = ids.iloc[te_idx].astype(str).to_numpy()

    rng = np.random.default_rng(42)
    aucs, briers = [], []
    for _ in range(200):
        sample = set(rng.choice(np.unique(subjects), size=len(np.unique(subjects)), replace=True))
        mask = np.isin(subjects, list(sample))
        yy, pp = y_te[mask], prob[mask]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, pp))
        briers.append(brier_score_loss(yy, pp))

    result = {
        "auc_point": float(roc_auc_score(y_te, prob)),
        "auc_ci": [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)],
        "brier_point": float(brier_score_loss(y_te, prob)),
        "brier_ci": [round(float(np.percentile(briers, 2.5)), 4), round(float(np.percentile(briers, 97.5)), 4)],
        "test_subjects": int(len(np.unique(subjects))),
        "n_boot": 200,
    }
    out = PROJECT_ROOT / "output" / "charls_bootstrap_ci.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
