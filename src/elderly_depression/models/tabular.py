"""Tabular baseline models with imbalance handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE, TEST_SIZE
from ..evaluation import classification_report_frame
from ..imbalance import best_threshold, resample_train, scale_pos_weight


def prepare_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.columns.difference(num_cols)
    X[num_cols] = X[num_cols].fillna(X[num_cols].median(numeric_only=True))
    for c in cat_cols:
        X[c] = X[c].fillna("UNK")
    if len(cat_cols):
        X = pd.get_dummies(X, columns=list(cat_cols), drop_first=True)
    return X.fillna(0)


def _select_features(X_train: pd.DataFrame, y_train: pd.Series) -> list[str]:
    if X_train.shape[1] <= 60:
        return X_train.columns.tolist()
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_train)
    selector = LogisticRegression(
        penalty="l1", solver="saga", max_iter=2000, random_state=RANDOM_STATE
    )
    selector.fit(X_scaled, y_train)
    coef = np.abs(selector.coef_[0])
    top = min(200, len(coef))
    idx = np.argsort(coef)[::-1][:top]
    return X_train.columns[idx].tolist()


def _fit_eval(
    model,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    name: str,
) -> tuple[pd.DataFrame, dict]:
    model.fit(X_tr, y_tr)
    va_prob = model.predict_proba(X_va)[:, 1]
    te_prob = model.predict_proba(X_te)[:, 1]
    threshold = best_threshold(y_va, va_prob, metric="f1")
    y_pred = (te_prob >= threshold).astype(int)
    report = classification_report_frame(y_te, y_pred, te_prob)
    report.insert(0, "model", name)
    report["threshold"] = threshold
    return report, {"model": model, "threshold": threshold}


def train_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "depression",
    resample_method: str = "smote",
    output_dir: Path | str | None = None,
) -> dict:
    df = df.dropna(subset=[label_col]).copy()
    y = df[label_col].astype(int)
    X = prepare_X(df, feature_cols)
    X = X.loc[y.index]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr
    )
    features = _select_features(X_tr, y_tr)
    X_tr, X_va, X_te = X_tr[features], X_va[features], X_te[features]
    X_tr, y_tr = resample_train(X_tr, y_tr, method=resample_method, random_state=RANDOM_STATE)

    sw = scale_pos_weight(y_tr)
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "xgboost": _xgb(sw),
        "logistic": make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
    }

    fitted = {}
    results = []
    for name, model in models.items():
        report, info = _fit_eval(model, X_tr, y_tr, X_va, y_va, X_te, y_te, name)
        results.append(report)
        fitted[name] = {**info, "features": features}

    result_df = pd.concat(results, ignore_index=True)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out / "tabular_model_report.csv", index=False, encoding="utf-8-sig")
    return {"report": result_df, "models": fitted}


def train_temporal_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "depression",
    resample_method: str = "smote_tomek",
    output_dir: Path | str | None = None,
) -> dict:
    """Train on earlier CHARLS waves and validate on a later wave."""
    df = df.dropna(subset=[label_col, "wave"]).copy()
    df[label_col] = df[label_col].astype(int)
    waves = sorted(df["wave"].unique())

    all_reports = []
    for test_wave in waves[1:]:
        train_df = df[df["wave"] < test_wave]
        test_df = df[df["wave"] == test_wave]
        if len(train_df) < 60 or len(test_df) < 30:
            continue

        y_train = train_df[label_col]
        X_train = prepare_X(train_df, feature_cols).loc[y_train.index]
        y_test = test_df[label_col]
        X_test = prepare_X(test_df, feature_cols).loc[y_test.index]
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        X_tr, X_va, y_tr, y_va = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
        )
        features = _select_features(X_tr, y_tr)
        X_tr, X_va, X_te = X_tr[features], X_va[features], X_test[features]
        X_tr, y_tr = resample_train(X_tr, y_tr, method=resample_method, random_state=RANDOM_STATE)

        sw = scale_pos_weight(y_tr)
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
            ),
            "xgboost": _xgb(sw),
            "logistic": make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ),
        }
        for name, model in models.items():
            report, _ = _fit_eval(model, X_tr, y_tr, X_va, y_va, X_te, y_test, name)
            report.insert(0, "test_wave", test_wave)
            report["train_rows"] = len(train_df)
            report["test_rows"] = len(test_df)
            all_reports.append(report)

    if not all_reports:
        raise ValueError("没有足够的前后波次数据用于时间外验证")
    result_df = pd.concat(all_reports, ignore_index=True)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out / "tabular_temporal_report.csv", index=False, encoding="utf-8-sig")
    return {"report": result_df}


def build_longitudinal_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "depression",
) -> pd.DataFrame:
    """Create per-person lag features from the previous CHARLS wave."""
    out = df.copy()
    out = out.sort_values(["id", "wave"]).reset_index(drop=True)
    lag_base = [c for c in feature_cols if c in out.columns]
    if "cesd_score" in out.columns:
        lag_base = lag_base + ["cesd_score"]
    for col in lag_base:
        out[f"prev_{col}"] = out.groupby("id")[col].shift(1)
    if "prev_age" not in out.columns:
        raise ValueError("纵向特征需要上一波次年龄，请确认数据包含 id 和 wave")
    out = out[out["prev_age"].notna()].copy()
    return out


def train_longitudinal_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "depression",
    resample_method: str = "smote_tomek",
    output_dir: Path | str | None = None,
) -> dict:
    """Person-level split with previous-wave lag features."""
    long_df = build_longitudinal_features(df, feature_cols, label_col)
    y = long_df[label_col].astype(int)
    ids = long_df["id"].astype(str)
    prev_cols = [
        c for c in long_df.columns if c.startswith("prev_") and c != label_col
    ]
    X = prepare_X(long_df, prev_cols).loc[y.index]

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(split.split(X, y, groups=ids))
    X_tr0, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr0, y_te = y.iloc[train_idx], y.iloc[test_idx]

    split_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(
        split_val.split(X_tr0, y_tr0, groups=ids.iloc[train_idx])
    )
    X_tr, X_va = X_tr0.iloc[tr_idx], X_tr0.iloc[va_idx]
    y_tr, y_va = y_tr0.iloc[tr_idx], y_tr0.iloc[va_idx]

    features = _select_features(X_tr, y_tr)
    X_tr, X_va, X_te = X_tr[features], X_va[features], X_te[features]
    X_tr, y_tr = resample_train(X_tr, y_tr, method=resample_method, random_state=RANDOM_STATE)

    sw = scale_pos_weight(y_tr)
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "xgboost": _xgb(sw),
        "logistic": make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
    }
    results = []
    for name, model in models.items():
        report, _ = _fit_eval(model, X_tr, y_tr, X_va, y_va, X_te, y_te, name)
        report.insert(0, "split", "person-level")
        report["train_ids"] = ids.iloc[train_idx].nunique()
        report["test_ids"] = ids.iloc[test_idx].nunique()
        report["train_rows"] = len(train_idx)
        report["test_rows"] = len(test_idx)
        results.append(report)

    result_df = pd.concat(results, ignore_index=True)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out / "tabular_longitudinal_report.csv", index=False, encoding="utf-8-sig")
    return {"report": result_df, "features": features}


def train_tabular_from_synthetic(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "depression",
    output_dir: Path | str | None = None,
) -> dict:
    """Train on synthetic rows, evaluate on a real person-level holdout."""
    real_df = real_df.dropna(subset=[label_col]).copy()
    y_real = real_df[label_col].astype(int)
    X_real = prepare_X(real_df, feature_cols).loc[y_real.index]
    ids = real_df["id"].astype(str)

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(split.split(X_real, y_real, groups=ids))
    split_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(
        split_val.split(X_real.iloc[train_idx], y_real.iloc[train_idx], groups=ids.iloc[train_idx])
    )
    X_va = X_real.iloc[train_idx].iloc[va_idx]
    X_te = X_real.iloc[test_idx]
    y_va = y_real.iloc[train_idx].iloc[va_idx]
    y_te = y_real.iloc[test_idx]

    syn_cols = [c for c in synthetic_df.columns if c in X_real.columns]
    X_syn = synthetic_df[syn_cols].reindex(columns=X_real.columns, fill_value=0)
    y_syn = synthetic_df[label_col].astype(int)
    sw = scale_pos_weight(y_syn)
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "xgboost": _xgb(sw),
        "logistic": make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
    }
    results = []
    for name, model in models.items():
        report, _ = _fit_eval(model, X_syn, y_syn, X_va, y_va, X_te, y_te, name)
        report.insert(0, "source", "synthetic_train_real_eval")
        report["train_ids"] = int(ids.iloc[train_idx].nunique())
        report["test_ids"] = int(ids.iloc[test_idx].nunique())
        results.append(report)
    result_df = pd.concat(results, ignore_index=True)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out / "tabular_synthetic_report.csv", index=False, encoding="utf-8-sig")
    return {"report": result_df}


def _xgb(sw: float):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=sw,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
