"""Late-fusion multimodal baseline for MPDD-style features."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE, TEST_SIZE
from ..evaluation import imbalanced_metrics_frame
from ..imbalance import best_threshold, scale_pos_weight


def split_modalities(df: pd.DataFrame, prefixes: list[str], label_col: str) -> dict[str, pd.DataFrame]:
    out = {}
    for prefix in prefixes:
        cols = [c for c in df.columns if c.startswith(prefix + "_")]
        if cols:
            out[prefix] = df[cols]
    return out


def train_late_fusion(
    df: pd.DataFrame,
    prefixes: list[str],
    label_col: str = "hamd_binary",
    output_dir: Path | str | None = None,
) -> dict:
    df = df.dropna(subset=[label_col]).copy()
    y = df[label_col].astype(int)
    modality_dfs = split_modalities(df, prefixes, label_col)
    if not modality_dfs:
        raise ValueError("No modality feature columns found")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr
    )

    single_reports = []
    prob_train = []
    prob_val = []
    prob_test = []
    for prefix, mdf in modality_dfs.items():
        cols = mdf.columns
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(X_tr[cols].fillna(0).select_dtypes(include=[np.number]).values)
        x_va = scaler.transform(X_va[cols].fillna(0).select_dtypes(include=[np.number]).values)
        x_te = scaler.transform(X_te[cols].fillna(0).select_dtypes(include=[np.number]).values)
        sw = scale_pos_weight(y_tr)
        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        )
        model.fit(x_tr, y_tr)
        p_tr = model.predict_proba(x_tr)[:, 1]
        p_va = model.predict_proba(x_va)[:, 1]
        p_te = model.predict_proba(x_te)[:, 1]
        threshold = best_threshold(y_va, p_va, metric="youden")
        prob_train.append(p_tr.reshape(-1, 1))
        prob_val.append(p_va.reshape(-1, 1))
        prob_test.append(p_te.reshape(-1, 1))
        pred = (p_te >= threshold).astype(int)
        report = imbalanced_metrics_frame(y_te, pred, p_te)
        report.insert(0, "positive_rate", float(pred.mean()))
        report.insert(0, "threshold", threshold)
        report.insert(0, "modality", prefix)
        single_reports.append(report)

    fusion_X_tr = np.hstack(prob_train)
    fusion_X_va = np.hstack(prob_val)
    fusion_X_te = np.hstack(prob_test)
    fusion = LogisticRegression(class_weight="balanced", max_iter=2000)
    fusion.fit(fusion_X_tr, y_tr)
    f_va = fusion.predict_proba(fusion_X_va)[:, 1]
    f_prob = fusion.predict_proba(fusion_X_te)[:, 1]
    threshold = best_threshold(y_va, f_va, metric="youden")
    pred = (f_prob >= threshold).astype(int)
    f_report = imbalanced_metrics_frame(y_te, pred, f_prob)
    f_report.insert(0, "positive_rate", float(pred.mean()))
    f_report.insert(0, "threshold", threshold)
    f_report.insert(0, "modality", "fusion")
    reports = pd.concat(single_reports + [f_report], ignore_index=True)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        reports.to_csv(out / "multimodal_report.csv", index=False, encoding="utf-8-sig")
    return {"report": reports, "fusion_model": fusion, "features_per_modality": {
        k: len(v.columns) for k, v in modality_dfs.items()
    }}


def _fit_modality(
    x_tr: np.ndarray,
    y_tr: pd.Series,
    pca_dims: int,
    n_estimators: int = 150,
) -> tuple[StandardScaler, PCA, RandomForestClassifier]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_tr)
    pca = PCA(
        n_components=min(pca_dims, x_scaled.shape[1], x_scaled.shape[0] - 1),
        random_state=RANDOM_STATE,
    )
    x_pca = pca.fit_transform(x_scaled)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_pca, y_tr)
    return scaler, pca, model


def _fit_regression(
    x_tr: np.ndarray,
    y_tr: pd.Series,
    pca_dims: int,
    n_estimators: int = 150,
) -> tuple[StandardScaler, PCA, RandomForestRegressor]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_tr)
    pca = PCA(
        n_components=min(pca_dims, x_scaled.shape[1], x_scaled.shape[0] - 1),
        random_state=RANDOM_STATE,
    )
    x_pca = pca.fit_transform(x_scaled)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_pca, y_tr)
    return scaler, pca, model


def train_subject_late_fusion(
    df: pd.DataFrame,
    prefixes: list[str],
    label_col: str = "hamd_binary",
    subject_col: str = "subject_id",
    output_dir: Path | str | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    pca_dims: int = 64,
) -> dict:
    """Repeated subject-grouped CV with PCA and subject-level probability aggregation."""
    df = df.dropna(subset=[label_col]).copy()
    y = df[label_col].astype(int).values
    groups = df[subject_col].astype(str)
    subjects = groups.copy()

    fold_reports = []
    for repeat in range(n_repeats):
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + repeat
        )
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, y, groups)):
            y_tr = pd.Series(y[train_idx])
            y_va = pd.Series(y[val_idx])
            val_subjects = groups.iloc[val_idx]
            train_probs: dict[str, np.ndarray] = {}
            val_probs: dict[str, np.ndarray] = {}

            for prefix in prefixes:
                cols = [c for c in df.columns if c.startswith(prefix + "_")]
                if not cols:
                    continue
                x_all = df[cols].fillna(0).select_dtypes(include=[np.number]).values
                scaler, pca, model = _fit_modality(x_all[train_idx], y_tr, pca_dims)
                p_tr = model.predict_proba(pca.transform(scaler.transform(x_all[train_idx])))[:, 1]
                p_va = model.predict_proba(pca.transform(scaler.transform(x_all[val_idx])))[:, 1]
                train_probs[prefix] = p_tr
                val_probs[prefix] = p_va

                prob = pd.Series(p_va, index=groups.iloc[val_idx].index)
                subj_prob = prob.groupby(groups.iloc[val_idx], sort=False).mean()
                subj_y = pd.Series(y[val_idx], index=groups.iloc[val_idx].index).groupby(
                    groups.iloc[val_idx], sort=False
                ).first()
                threshold = best_threshold(subj_y, subj_prob.values, metric="youden")
                pred = (subj_prob.values >= threshold).astype(int)
                metrics = imbalanced_metrics_frame(subj_y, pred, subj_prob.values)
                metrics.insert(0, "positive_rate", float(pred.mean()))
                metrics.insert(0, "threshold", threshold)
                metrics.insert(0, "modality", prefix)
                metrics.insert(0, "fold", fold)
                metrics.insert(0, "repeat", repeat)
                fold_reports.append(metrics)

            if train_probs:
                fusion = LogisticRegression(class_weight="balanced", max_iter=2000)
                fusion.fit(np.column_stack(list(train_probs.values())), y_tr)
                f_va = fusion.predict_proba(np.column_stack(list(val_probs.values())))[:, 1]
                prob = pd.Series(f_va, index=groups.iloc[val_idx].index)
                subj_prob = prob.groupby(groups.iloc[val_idx], sort=False).mean()
                subj_y = pd.Series(y[val_idx], index=groups.iloc[val_idx].index).groupby(
                    groups.iloc[val_idx], sort=False
                ).first()
                threshold = best_threshold(subj_y, subj_prob.values, metric="youden")
                pred = (subj_prob.values >= threshold).astype(int)
                metrics = imbalanced_metrics_frame(subj_y, pred, subj_prob.values)
                metrics.insert(0, "positive_rate", float(pred.mean()))
                metrics.insert(0, "threshold", threshold)
                metrics.insert(0, "modality", "fusion")
                metrics.insert(0, "fold", fold)
                metrics.insert(0, "repeat", repeat)
                fold_reports.append(metrics)

    reports = pd.concat(fold_reports, ignore_index=True)
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
        "auc",
        "pr_auc",
        "brier",
        "positive_rate",
        "threshold",
    ]
    summary_rows = []
    for modality, group in reports.groupby("modality", sort=False):
        row = {"modality": modality}
        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    pipeline = fit_subject_pipeline(df, prefixes, label_col=label_col, subject_col=subject_col, pca_dims=pca_dims)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        reports.to_csv(out / "multimodal_subject_fold_report.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(out / "multimodal_subject_report.csv", index=False, encoding="utf-8-sig")
        joblib.dump(pipeline, out / "subject_fusion_pipeline.pkl")
    return {"report": reports, "summary": summary, "pipeline": pipeline}


def fit_subject_pipeline(
    df: pd.DataFrame,
    prefixes: list[str],
    label_col: str = "hamd_binary",
    subject_col: str = "subject_id",
    pca_dims: int = 64,
    n_splits: int = 5,
) -> dict:
    """Fit per-modality models plus an OOF-trained fusion model for new-subject inference."""
    df = df.dropna(subset=[label_col]).copy()
    y = df[label_col].astype(int)
    groups = df[subject_col].astype(str)
    feature_cols = {
        prefix: [c for c in df.columns if c.startswith(prefix + "_")]
        for prefix in prefixes
    }

    oof = {prefix: np.full(len(df), np.nan) for prefix in feature_cols if feature_cols[prefix]}
    if oof:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, val_idx in sgkf.split(df, y, groups):
            for prefix, cols in feature_cols.items():
                if not cols:
                    continue
                x_all = df[cols].fillna(0).select_dtypes(include=[np.number]).values
                scaler, pca, model = _fit_modality(x_all[train_idx], y.iloc[train_idx], pca_dims)
                oof[prefix][val_idx] = model.predict_proba(
                    pca.transform(scaler.transform(x_all[val_idx]))
                )[:, 1]

    modalities = {}
    for prefix, cols in feature_cols.items():
        if not cols:
            continue
        x_all = df[cols].fillna(0).select_dtypes(include=[np.number]).values
        scaler, pca, model = _fit_modality(x_all, y, pca_dims)
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        top_features = []
        for idx in top_idx:
            component = pca.components_[idx]
            top_features.append(cols[int(np.argmax(np.abs(component)))])
        modalities[prefix] = {
            "columns": cols,
            "scaler": scaler,
            "pca": pca,
            "model": model,
            "missing_prob": 0.5,
            "top_features": top_features,
            "pca_importance": {f"pc{idx}": float(importances[idx]) for idx in top_idx},
        }

    prefixes_used = list(modalities.keys())
    fusion_options = {}
    calib_pool = []
    for size in range(1, len(prefixes_used) + 1):
        for subset in combinations(prefixes_used, size):
            subset = list(subset)
            matrix = np.column_stack([oof[p] for p in subset])
            model = LogisticRegression(class_weight="balanced", max_iter=2000)
            model.fit(matrix, y)
            prob = model.predict_proba(matrix)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(prob, y)
            calibrated = calibrator.predict(prob)
            calib_pool.extend(calibrated.tolist())
            fusion_options["|".join(subset)] = {
                "prefixes": subset,
                "model": model,
                "threshold": float(best_threshold(y, calibrated, metric="youden")),
                "calibrator": calibrator,
            }
    risk_bins = {"low": float(np.quantile(calib_pool, 0.4)), "medium": float(np.quantile(calib_pool, 0.8))}
    return {
        "version": 2,
        "subject_col": subject_col,
        "label_col": label_col,
        "prefixes": prefixes_used,
        "modalities": modalities,
        "fusion": fusion_options,
        "risk_bins": risk_bins,
    }


def predict_subject_pipeline(pipeline: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Predict subject-level probabilities for one or more subjects."""
    subject_col = pipeline["subject_col"]
    rows = []
    for subject, sub in df.groupby(subject_col, sort=False):
        row = {"subject_id": subject}
        modality_probs = {}
        available = []
        for prefix in pipeline["prefixes"]:
            spec = pipeline["modalities"][prefix]
            cols = [c for c in spec["columns"] if c in sub.columns]
            if not cols or sub[cols].notna().sum().sum() == 0:
                modality_probs[prefix] = spec.get("missing_prob", 0.5)
            else:
                available.append(prefix)
                x = sub[cols].fillna(0).select_dtypes(include=[np.number]).values
                prob = spec["model"].predict_proba(
                    spec["pca"].transform(spec["scaler"].transform(x))
                )[:, 1]
                modality_probs[prefix] = float(np.mean(prob))
            row[f"{prefix}_prob"] = modality_probs[prefix]
            row[f"{prefix}_top_features"] = "、".join(spec.get("top_features", [])[:3])
        fusion_prefixes = available if available else [pipeline["prefixes"][0]]
        fusion_key = "|".join(fusion_prefixes)
        if fusion_key not in pipeline["fusion"]:
            fusion_key = "|".join(pipeline["prefixes"])
        fusion_spec = pipeline["fusion"][fusion_key]
        vector = np.array([modality_probs[p] for p in fusion_prefixes]).reshape(1, -1)
        fusion_prob = float(fusion_spec["model"].predict_proba(vector)[:, 1][0])
        calibrator = fusion_spec.get("calibrator")
        calibrated_prob = float(calibrator.predict(np.array([fusion_prob]))[0]) if calibrator else fusion_prob
        threshold = fusion_spec["threshold"]
        row["fusion_prob"] = fusion_prob
        row["calibrated_prob"] = calibrated_prob
        row["prediction"] = int(calibrated_prob >= threshold)
        row["threshold"] = threshold
        bins = pipeline.get("risk_bins", {"low": 0.4, "medium": 0.7})
        if calibrated_prob >= bins.get("medium", 0.7):
            row["risk_level"] = "high"
        elif calibrated_prob >= bins.get("low", 0.4):
            row["risk_level"] = "medium"
        else:
            row["risk_level"] = "low"
        row["explanation"] = "；".join(
            f"{p}: {row[f'{p}_prob']:.0%}" for p in pipeline["prefixes"]
        ) + f"，融合: {fusion_prob:.0%}"
        rows.append(row)
    return pd.DataFrame(rows)


def train_subject_target(
    df: pd.DataFrame,
    prefixes: list[str],
    target_col: str,
    task: str = "class",
    subject_col: str = "subject_id",
    output_dir: Path | str | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    pca_dims: int = 64,
) -> pd.DataFrame:
    """Subject-grouped CV for label3 classification or PHQ-9 regression."""
    df = df.dropna(subset=[target_col]).copy()
    groups = df[subject_col].astype(str)
    y = df[target_col]
    if task == "class":
        y = y.astype(int)
    else:
        y = y.astype(float)

    fold_rows = []
    for repeat in range(n_repeats):
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + 1000 + repeat
        )
        strat_y = y if task == "class" else df.get("hamd_binary", y).fillna(0).astype(int)
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, strat_y, groups)):
            y_tr = y.iloc[train_idx]
            y_va = y.iloc[val_idx]
            subject_preds: dict[str, list] = defaultdict(list)
            subject_y: dict[str, float] = {}

            for prefix in prefixes:
                cols = [c for c in df.columns if c.startswith(prefix + "_")]
                if not cols:
                    continue
                x_all = df[cols].fillna(0).select_dtypes(include=[np.number]).values
                if task == "class":
                    scaler, pca, model = _fit_modality(x_all[train_idx], y_tr, pca_dims)
                    pred = model.predict(pca.transform(scaler.transform(x_all[val_idx])))
                else:
                    scaler, pca, model = _fit_regression(x_all[train_idx], y_tr, pca_dims)
                    pred = model.predict(pca.transform(scaler.transform(x_all[val_idx])))
                for subject, value in zip(groups.iloc[val_idx], pred):
                    subject_preds[str(subject)].append(float(value))
            for subject in subject_preds:
                subject_y[subject] = float(y_va.iloc[groups.iloc[val_idx].values == subject].iloc[0])

            agg_y = []
            agg_pred = []
            for subject, values in subject_preds.items():
                agg_y.append(subject_y[subject])
                if task == "class":
                    agg_pred.append(Counter(values).most_common(1)[0][0])
                else:
                    agg_pred.append(float(np.mean(values)))

            if task == "class":
                metrics = {
                    "accuracy": accuracy_score(agg_y, agg_pred),
                    "balanced_accuracy": balanced_accuracy_score(agg_y, agg_pred),
                    "f1_macro": f1_score(agg_y, agg_pred, average="macro", zero_division=0),
                }
            else:
                mae = mean_absolute_error(agg_y, agg_pred)
                rmse = float(np.sqrt(mean_squared_error(agg_y, agg_pred)))
                metrics = {
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2_score(agg_y, agg_pred),
                    "phq_binary_accuracy": accuracy_score(
                        np.asarray(agg_y) >= 8, np.asarray(agg_pred) >= 8
                    ),
                }
            metrics["repeat"] = repeat
            metrics["fold"] = fold
            fold_rows.append(metrics)

    reports = pd.DataFrame(fold_rows)
    summary = reports.drop(columns=["repeat", "fold"]).mean().to_frame().T
    summary.insert(0, "target", target_col)
    summary.insert(1, "task", task)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = target_col.replace("-", "_").lower()
        reports.to_csv(out / f"multimodal_subject_{stem}_fold_report.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(out / f"multimodal_subject_{stem}_report.csv", index=False, encoding="utf-8-sig")
    return summary
