"""Subject-aware CMDC benchmark and external-validation models."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CMDC_ALL_MODALITIES = (
    "text",
    "visual",
    "audio_deep",
    "acoustic",
)

CMDC_VARIANTS = {
    "text": ("text",),
    "acoustic": ("acoustic",),
    "audio": ("audio_deep", "acoustic"),
    "visual": ("visual",),
    "text_audio": ("text", "audio_deep", "acoustic"),
    "complete_multimodal": CMDC_ALL_MODALITIES,
}

CMDC_PRIMARY_VARIANT = "text_audio"
CMDC_SCHEMA_VERSION = 1
CMDC_EVALUATION_PROTOCOL = "subject_aware_repeated_stratified_cv_v1"
CMDC_RANDOM_STATE = 42


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Return a stable fingerprint for schema and subject-level feature values."""
    frame = df.reindex(sorted(df.columns), axis=1)
    digest = hashlib.sha256()
    digest.update("\x1f".join(str(column) for column in frame.columns).encode("utf-8"))
    digest.update(
        pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    )
    return digest.hexdigest()


def _modality_features(df: pd.DataFrame, modality: str) -> list[str]:
    return [
        column
        for column in df.columns
        if column.startswith(f"{modality}_")
        and not column.endswith("_questions")
    ]


def _eligible_rows(
    df: pd.DataFrame,
    modalities: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    columns = [column for modality in modalities for column in _modality_features(df, modality)]
    if not columns:
        return df.iloc[0:0].copy(), []
    available = df[columns].notna().any(axis=1)
    if len(modalities) > 1:
        for modality in modalities:
            modality_columns = _modality_features(df, modality)
            if not modality_columns:
                return df.iloc[0:0].copy(), []
            available &= df[modality_columns].notna().any(axis=1)
    return df.loc[available].copy(), columns


def _model() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=False),
        StandardScaler(),
        LogisticRegression(
            C=0.01,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
        ),
    )


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def _evaluate_variant(
    df: pd.DataFrame,
    modalities: tuple[str, ...],
    repeats: int,
) -> tuple[dict, object | None, list[str]]:
    subset, columns = _eligible_rows(df, modalities)
    if subset.empty or "label" not in subset.columns:
        return {}, None, []
    labels = pd.to_numeric(subset["label"], errors="coerce")
    valid = labels.notna()
    subset = subset.loc[valid].copy()
    labels = labels.loc[valid].astype(int)
    if labels.nunique() < 2 or min(labels.value_counts()) < 4:
        return {}, None, columns

    X = subset[columns].to_numpy(dtype=np.float64)
    y = labels.to_numpy(dtype=int)
    folds = min(5, int(labels.value_counts().min()))
    cv = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=max(1, repeats),
        random_state=CMDC_RANDOM_STATE,
    )
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    for train_index, test_index in cv.split(X, y):
        model = _model()
        model.fit(X[train_index], y[train_index])
        probability = model.predict_proba(X[test_index])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        tn, fp, _, _ = confusion_matrix(
            y[test_index],
            prediction,
            labels=[0, 1],
        ).ravel()
        metrics["auc"].append(roc_auc_score(y[test_index], probability))
        metrics["pr_auc"].append(
            average_precision_score(y[test_index], probability)
        )
        metrics["balanced_accuracy"].append(
            balanced_accuracy_score(y[test_index], prediction)
        )
        metrics["sensitivity"].append(
            recall_score(
                y[test_index],
                prediction,
                pos_label=1,
                zero_division=0,
            )
        )
        metrics["specificity"].append(tn / (tn + fp) if tn + fp else 0.0)
        metrics["f1"].append(
            f1_score(y[test_index], prediction, pos_label=1, zero_division=0)
        )
        metrics["brier"].append(
            brier_score_loss(y[test_index], probability)
        )

    final_model = _model()
    final_model.fit(X, y)
    row = {
        "variant": "+".join(modalities),
        "subjects": int(len(subset)),
        "hc_subjects": int((y == 0).sum()),
        "mdd_subjects": int((y == 1).sum()),
        "features": int(len(columns)),
        "folds": int(folds),
        "repeats": int(max(1, repeats)),
    }
    for metric, values in metrics.items():
        mean, std = _mean_std(values)
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
    return row, final_model, columns


def train_cmdc_models(
    df: pd.DataFrame,
    report_path: Path | str,
    summary_path: Path | str,
    model_dir: Path | str,
    repeats: int = 10,
) -> dict:
    """Evaluate CMDC with subject-level repeated cross-validation."""

    if df.empty or "label" not in df.columns:
        raise ValueError("CMDC 特征表为空或缺少 label")

    fingerprint = dataset_fingerprint(df)
    report_rows = []
    trained_models: dict[str, dict] = {}
    for name, modalities in CMDC_VARIANTS.items():
        row, model, columns = _evaluate_variant(df, modalities, repeats)
        if not row:
            continue
        row["variant_name"] = name
        report_rows.append(row)
        if name in {CMDC_PRIMARY_VARIANT, "complete_multimodal"} and model is not None:
            trained_models[name] = {
                "model": model,
                "features": columns,
                "variant": name,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "schema_version": CMDC_SCHEMA_VERSION,
                "dataset_fingerprint": fingerprint,
                "random_state": CMDC_RANDOM_STATE,
                "evaluation_protocol": CMDC_EVALUATION_PROTOCOL,
                "note": "仅用于 CMDC 外部验证，不进入老年主风险评估。",
            }

    if not report_rows:
        raise ValueError("没有足够的 CMDC 受试者完成建模")

    report = pd.DataFrame(report_rows)
    report_order = {name: index for index, name in enumerate(CMDC_VARIANTS)}
    report["_order"] = report["variant_name"].map(report_order)
    report = report.sort_values("_order").drop(columns="_order")
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_output, index=False, encoding="utf-8-sig")
    report_output.chmod(0o600)

    models_output = Path(model_dir)
    models_output.mkdir(parents=True, exist_ok=True)
    model_artifacts = []
    for name, payload in trained_models.items():
        path = models_output / f"cmdc_{name}.pkl"
        joblib.dump(payload, path)
        path.chmod(0o600)
        model_artifacts.append(str(path))

    label_counts = pd.to_numeric(df["label"], errors="coerce").value_counts()
    primary = report.loc[
        report["variant_name"] == CMDC_PRIMARY_VARIANT
    ]
    summary = {
        "ready": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subjects": int(len(df)),
        "hc_subjects": int(label_counts.get(0, 0)),
        "mdd_subjects": int(label_counts.get(1, 0)),
        "primary_variant": CMDC_PRIMARY_VARIANT,
        "schema_version": CMDC_SCHEMA_VERSION,
        "dataset_fingerprint": fingerprint,
        "random_state": CMDC_RANDOM_STATE,
        "evaluation_protocol_id": CMDC_EVALUATION_PROTOCOL,
        "primary_result": (
            primary.iloc[0].to_dict() if not primary.empty else None
        ),
        "report_path": str(report_output),
        "model_artifacts": model_artifacts,
        "evaluation_protocol": (
            "按 subject_id 聚合；重复分层 K 折；无问句级样本泄漏；"
            "固定正则化逻辑回归和 0.5 阈值。"
        ),
        "evidence_boundary": (
            "探索性单中心小样本结果，不代表跨中心、老年人群或临床部署性能。"
        ),
        "age_scope_warning": "CMDC 为混合成人样本，不是老年专属队列。",
    }
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_output.chmod(0o600)
    return summary
