"""Locked subject-level holdout evaluation for CMDC."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.model_selection import train_test_split

from .cmdc_advanced import (
    _bootstrap_ci,
    _logistic_model,
    _mean_oof,
    _metrics,
    _prepare_subset,
    _repeated_oof,
)
from .cmdc import (
    CMDC_SCHEMA_VERSION,
    dataset_fingerprint,
)


HOLDOUT_SEED = 20261109
MIN_SCREENING_SENSITIVITY = 0.85


def _clopper_pearson(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> dict:
    """Return an exact binomial interval without assuming normal errors."""
    if trials <= 0:
        return {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    alpha = 1.0 - confidence
    low = (
        0.0
        if successes <= 0
        else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    )
    high = (
        1.0
        if successes >= trials
        else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return {
        "estimate": float(successes / trials),
        "successes": int(successes),
        "trials": int(trials),
        "ci_low": low,
        "ci_high": high,
    }


def _select_screening_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum_sensitivity: float = MIN_SCREENING_SENSITIVITY,
) -> dict:
    candidates = []
    for threshold in np.unique(np.round(probabilities, 4)):
        prediction = (probabilities >= threshold).astype(int)
        tp = int(((prediction == 1) & (labels == 1)).sum())
        tn = int(((prediction == 0) & (labels == 0)).sum())
        fp = int(((prediction == 1) & (labels == 0)).sum())
        fn = int(((prediction == 0) & (labels == 1)).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        balanced = (sensitivity + specificity) / 2
        candidates.append(
            {
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "balanced_accuracy": float(balanced),
            }
        )
    eligible = [
        item
        for item in candidates
        if item["sensitivity"] >= minimum_sensitivity
    ]
    pool = eligible or candidates
    return max(
        pool,
        key=lambda item: (
            item["specificity"],
            item["balanced_accuracy"],
            -abs(item["threshold"] - 0.5),
        ),
    )


def run_cmdc_locked_holdout(
    df: pd.DataFrame,
    output_dir: Path | str,
    *,
    repeats: int = 10,
    bootstrap_iterations: int = 1000,
    test_size: float = 0.2,
) -> dict:
    fingerprint = dataset_fingerprint(df)
    subset, labels, columns = _prepare_subset(
        df,
        ("text", "audio_deep", "acoustic"),
    )
    if len(subset) < 30 or len(np.unique(labels)) < 2:
        raise ValueError("CMDC 可用样本不足，无法进行留出评估")

    train_index, test_index = train_test_split(
        np.arange(len(subset)),
        test_size=test_size,
        random_state=HOLDOUT_SEED,
        stratify=labels,
    )
    train = subset.iloc[train_index].reset_index(drop=True)
    test = subset.iloc[test_index].reset_index(drop=True)
    y_train = labels[train_index]
    y_test = labels[test_index]

    train_oof, folds = _repeated_oof(
        train,
        y_train,
        columns,
        repeats,
    )
    train_probability = _mean_oof(train_oof)
    threshold_selection = _select_screening_threshold(
        y_train,
        train_probability,
    )
    threshold = threshold_selection["threshold"]
    train_cv_metrics = _metrics(y_train, train_probability)
    train_cv_metrics.update(threshold_selection)

    model = _logistic_model()
    X_train = train[columns].to_numpy(dtype=np.float64)
    X_test = test[columns].to_numpy(dtype=np.float64)
    model.fit(X_train, y_train)
    test_probability = model.predict_proba(X_test)[:, 1]

    locked_prediction = (test_probability >= threshold).astype(int)
    default_prediction = (test_probability >= 0.5).astype(int)

    def metrics_at(prediction: np.ndarray) -> dict:
        tp = int(((prediction == 1) & (y_test == 1)).sum())
        tn = int(((prediction == 0) & (y_test == 0)).sum())
        fp = int(((prediction == 1) & (y_test == 0)).sum())
        fn = int(((prediction == 0) & (y_test == 1)).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        output = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "balanced_accuracy": float((sensitivity + specificity) / 2),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
        output["sensitivity_exact_ci"] = _clopper_pearson(tp, tp + fn)
        output["specificity_exact_ci"] = _clopper_pearson(tn, tn + fp)
        return output

    holdout_metrics = _metrics(y_test, test_probability)
    holdout_metrics["locked_threshold"] = metrics_at(locked_prediction)
    holdout_metrics["default_threshold"] = metrics_at(default_prediction)
    holdout_bootstrap = _bootstrap_ci(
        y_test,
        test_probability,
        bootstrap_iterations,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "subject_id": test["subject_id"].astype(str),
            "label": y_test,
            "probability": test_probability,
            "prediction_locked": locked_prediction,
            "prediction_default": default_prediction,
        }
    )
    prediction_path = output / "cmdc_locked_holdout_predictions.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    prediction_path.chmod(0o600)

    model_path = output / "cmdc_locked_holdout_model.pkl"
    joblib.dump(
        {
            "model": model,
            "features": columns,
            "threshold": threshold,
            "minimum_sensitivity": MIN_SCREENING_SENSITIVITY,
            "holdout_seed": HOLDOUT_SEED,
            "schema_version": CMDC_SCHEMA_VERSION,
            "dataset_fingerprint": fingerprint,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "note": "锁定留出评估模型，不进入老年主风险评估。",
        },
        model_path,
    )
    model_path.chmod(0o600)

    summary = {
        "ready": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "holdout_seed": HOLDOUT_SEED,
        "schema_version": CMDC_SCHEMA_VERSION,
        "dataset_fingerprint": fingerprint,
        "variant": "text_audio",
        "subjects": int(len(subset)),
        "train_subjects": int(len(train)),
        "test_subjects": int(len(test)),
        "train_hc": int((y_train == 0).sum()),
        "train_mdd": int((y_train == 1).sum()),
        "test_hc": int((y_test == 0).sum()),
        "test_mdd": int((y_test == 1).sum()),
        "folds": int(folds),
        "repeats": int(repeats),
        "minimum_screening_sensitivity": MIN_SCREENING_SENSITIVITY,
        "locked_threshold": float(threshold),
        "train_cv_metrics": train_cv_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_bootstrap_ci": holdout_bootstrap,
        "prediction_path": str(prediction_path),
        "model_path": str(model_path),
        "evidence_boundary": (
            "单次本地留出集仅有约五分之一受试者，结果用于检验流程和阈值锁定，"
            "不能替代多中心或老年独立外部测试。"
        ),
        "small_sample_warning": (
            "测试集样本量很小；完美 AUC 或宽精确区间反映可评估信息有限，"
            "不能据此宣称模型已达到稳定临床性能。"
        ),
    }
    summary_path = output / "cmdc_locked_holdout_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.chmod(0o600)
    return summary
