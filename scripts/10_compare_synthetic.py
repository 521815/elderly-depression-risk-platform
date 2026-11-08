"""Compare real-data training against synthetic-only training on real holdout."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.imbalance import best_threshold
from elderly_depression.models.tabular import prepare_X
from elderly_depression.synthetic import TABULAR_CSV, MULTIMODAL_PARQUET
from elderly_depression import services

EXCLUDE = {"id", "wave", "cesd_score", "depression"}
MODALITIES = ["face", "audio", "gait", "personality"]


def _evaluate(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "positive_rate": float(pred.mean()),
    }


def _tabular_models(sw: float):
    return {
        "logistic": make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "xgboost": _xgb(sw),
    }


def _xgb(sw: float):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=sw,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def compare_tabular() -> list[dict]:
    real = services.load_charls()
    feature_cols = [c for c in real.columns if c not in EXCLUDE and c != "depression"]
    y = real["depression"].astype(int)
    X = prepare_X(real, feature_cols).loc[y.index]
    ids = real["id"].astype(str)

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(X, y, groups=ids))
    split_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, va_idx = next(
        split_val.split(X.iloc[train_idx], y.iloc[train_idx], groups=ids.iloc[train_idx])
    )
    X_tr, X_va, X_te = X.iloc[train_idx].iloc[tr_idx], X.iloc[train_idx].iloc[va_idx], X.iloc[test_idx]
    y_tr, y_va, y_te = y.iloc[train_idx].iloc[tr_idx], y.iloc[train_idx].iloc[va_idx], y.iloc[test_idx]

    syn = pd.read_csv(TABULAR_CSV)
    syn_cols = [c for c in syn.columns if c in X.columns]
    X_syn = syn[syn_cols].reindex(columns=X.columns, fill_value=0)
    y_syn = syn["depression"].astype(int)
    sw = (y_syn == 0).sum() / max(1, (y_syn == 1).sum())

    results = []
    for source, X_train, y_train in [("real", X_tr, y_tr), ("synthetic", X_syn, y_syn)]:
        for name, model in _tabular_models(sw).items():
            model.fit(X_train, y_train)
            va_prob = model.predict_proba(X_va)[:, 1]
            te_prob = model.predict_proba(X_te)[:, 1]
            threshold = best_threshold(y_va, va_prob, metric="f1")
            row = {"source": source, "model": name, "threshold": round(threshold, 3)}
            row.update(_evaluate(y_te, te_prob, threshold))
            results.append(row)
    return results


def _modality_model(prefix: str, X_syn, y_syn, X_real_train, y_real_train):
    """Train scaler+PCA+RF on synthetic features, return fitted model."""
    scaler = StandardScaler()
    pca = PCA(n_components=min(64, X_syn.shape[1], X_syn.shape[0] - 1), random_state=42)
    model = RandomForestClassifier(n_estimators=150, class_weight="balanced", n_jobs=-1, random_state=42)
    x = scaler.fit_transform(X_syn)
    x = pca.fit_transform(x)
    model.fit(x, y_syn)
    return scaler, pca, model


def compare_multimodal() -> list[dict]:
    real = services.load_mpdd_utterance_features().dropna(subset=["hamd_score"])
    real["hamd_binary"] = (real["hamd_score"] >= 8).astype(int)
    ids = real["subject_id"].astype(str)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(real, real["hamd_binary"], groups=ids))
    train = real.iloc[train_idx]
    test = real.iloc[test_idx]
    syn = pd.read_parquet(MULTIMODAL_PARQUET)

    results = []
    for prefix in MODALITIES:
        cols = [c for c in train.columns if c.startswith(prefix + "_")]
        if not cols:
            continue
        X_te = test[cols].fillna(0).astype("float32").to_numpy()
        y_te = test["hamd_binary"].to_numpy()

        # real baseline
        X_tr = train[cols].fillna(0).astype("float32").to_numpy()
        y_tr = train["hamd_binary"].to_numpy()
        scaler_r, pca_r, model_r = _modality_model(prefix, X_tr, y_tr, X_tr, y_tr)
        prob_r = model_r.predict_proba(pca_r.transform(scaler_r.transform(X_te)))[:, 1]
        thr_r = best_threshold(y_tr, model_r.predict_proba(pca_r.transform(scaler_r.transform(X_tr)))[:, 1], metric="f1")

        # synthetic-only
        X_syn = syn[[c for c in syn.columns if c.startswith(prefix + "_")]].fillna(0).astype("float32").to_numpy()
        y_syn = syn["hamd_binary"].to_numpy()
        scaler_s, pca_s, model_s = _modality_model(prefix, X_syn, y_syn, X_tr, y_tr)
        prob_s = model_s.predict_proba(pca_s.transform(scaler_s.transform(X_te)))[:, 1]
        thr_s = best_threshold(y_tr, model_s.predict_proba(pca_s.transform(scaler_s.transform(X_tr)))[:, 1], metric="f1")

        for source, prob, thr in [("real", prob_r, thr_r), ("synthetic", prob_s, thr_s)]:
            row = {"source": source, "model": f"rf_{prefix}", "threshold": round(float(thr), 3)}
            row.update(_evaluate(y_te, prob, thr))
            results.append(row)
    return results


def main() -> None:
    tabular_rows = compare_tabular()
    multimodal_rows = compare_multimodal()
    report = pd.DataFrame(tabular_rows + multimodal_rows)
    out = PROJECT_ROOT / "output" / "synthetic_comparison_report.csv"
    report.to_csv(out, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
