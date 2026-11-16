"""MODMA audio pretraining and MPDD target-domain transfer experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE
from ..evaluation import imbalanced_metrics_frame
from ..features.modma_audio import (
    aggregate_modma_subject_features,
    extract_modma_wav_features,
)
from .fusion import expected_calibration_error


METADATA_COLUMNS = {
    "source",
    "subject_id",
    "group",
    "label",
    "age",
    "gender",
    "readable_segments",
    "task_types",
}


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in METADATA_COLUMNS
    ]


def build_mpdd_raw_audio_features(
    raw_audio_root: Path | str,
    labels_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Extract MODMA-compatible acoustic features from raw MPDD WAV files."""
    raw_audio_root = Path(raw_audio_root)
    labels = pd.read_csv(labels_path).rename(
        columns={"ID": "subject_id", "label2": "mpdd_binary"}
    )
    labels["subject_id"] = labels["subject_id"].astype(str)
    labels = labels.dropna(subset=["mpdd_binary"]).copy()
    label_map = labels.set_index("subject_id")["mpdd_binary"].astype(int).to_dict()

    rows = []
    errors = []
    for subject_dir in sorted(
        path
        for path in raw_audio_root.iterdir()
        if path.is_dir() and path.name in label_map
    ):
        subject_id = subject_dir.name
        label = int(label_map[subject_id])
        for path in sorted(subject_dir.glob("*.WAV")):
            try:
                features = extract_modma_wav_features(path)
            except Exception as exc:
                errors.append(
                    {
                        "subject_id": subject_id,
                        "filename": path.name,
                        "error": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "source": "MPDD",
                    "subject_id": subject_id,
                    "group": "MDD" if label == 1 else "HC",
                    "label": label,
                    "age": np.nan,
                    "gender": "",
                    "segment": int(path.stem.split("_")[-1])
                    if path.stem.split("_")[-1].isdigit()
                    else -1,
                    "task_type": "mpdd_interview",
                    **features,
                }
            )
    if not rows:
        raise ValueError("No readable MPDD raw WAV files were found")
    utterance = pd.DataFrame(rows)
    subject = aggregate_modma_subject_features(
        utterance,
        source_name="MPDD",
    )
    subject["mpdd_binary"] = subject["label"].astype(int)
    return utterance, subject, errors


def _metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    row = imbalanced_metrics_frame(y_true, prediction, probability).iloc[0]
    return {
        **{key: float(value) for key, value in row.to_dict().items()},
        "ece": float(expected_calibration_error(y_true, probability)),
    }


def _bootstrap_delta_ci(
    deltas: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = [
        float(np.mean(rng.choice(deltas, size=len(deltas), replace=True)))
        for _ in range(n_bootstrap)
    ]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def run_modma_to_mpdd_transfer(
    source_features: pd.DataFrame,
    target_features: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    source_weight: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compare target-only, source pretraining, stacking, and joint training."""
    common_features = sorted(
        set(_feature_columns(source_features))
        & set(_feature_columns(target_features))
    )
    if not common_features:
        raise ValueError("MODMA and MPDD have no shared acoustic features")
    source = source_features.dropna(subset=["label"]).copy()
    target = target_features.dropna(subset=["mpdd_binary"]).copy()
    source_x = source[common_features].replace([np.inf, -np.inf], np.nan)
    source_y = source["label"].astype(int).to_numpy()
    target_x = target[common_features].replace([np.inf, -np.inf], np.nan)
    target_y = target["mpdd_binary"].astype(int).to_numpy()
    target_groups = target["subject_id"].astype(str)

    source_scaler = StandardScaler().fit(source_x)
    source_model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    ).fit(source_scaler.transform(source_x), source_y)
    source_probability = source_model.predict_proba(
        source_scaler.transform(target_x)
    )[:, 1]

    fold_rows = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )
        for fold, (train_idx, validation_idx) in enumerate(
            splitter.split(target, target_y, target_groups)
        ):
            target_scaler = StandardScaler().fit(target_x.iloc[train_idx])
            target_train = target_scaler.transform(target_x.iloc[train_idx])
            target_validation = target_scaler.transform(target_x.iloc[validation_idx])
            y_train = target_y[train_idx]
            y_validation = target_y[validation_idx]

            baseline = LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ).fit(target_train, y_train)
            baseline_probability = baseline.predict_proba(target_validation)[:, 1]

            zero_shot = source_probability[validation_idx]

            source_train_probability = source_probability[train_idx]
            stacking_x = np.column_stack(
                [target_train, source_train_probability]
            )
            stacking_validation = np.column_stack(
                [target_validation, zero_shot]
            )
            stacking = LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ).fit(stacking_x, y_train)
            stacking_probability = stacking.predict_proba(stacking_validation)[:, 1]

            joint_x = np.vstack(
                [target_train, source_scaler.transform(source_x)]
            )
            joint_y = np.concatenate([y_train, source_y])
            weights = np.concatenate(
                [
                    np.ones(len(target_train), dtype=float),
                    np.full(len(source_x), source_weight, dtype=float),
                ]
            )
            joint = LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ).fit(joint_x, joint_y, sample_weight=weights)
            joint_probability = joint.predict_proba(target_validation)[:, 1]

            for condition, probability in [
                ("target_only", baseline_probability),
                ("modma_zero_shot", zero_shot),
                ("modma_probability_stacking", stacking_probability),
                ("joint_source_target", joint_probability),
            ]:
                fold_rows.append(
                    {
                        "source": "MODMA_TO_MPDD",
                        "condition": condition,
                        "repeat": repeat,
                        "fold": fold,
                        **_metrics(y_validation, probability),
                        "n_train_subjects": int(
                            target_groups.iloc[train_idx].nunique()
                        ),
                        "n_validation_subjects": int(
                            target_groups.iloc[validation_idx].nunique()
                        ),
                    }
                )
    fold_report = pd.DataFrame(fold_rows)
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "auc",
        "pr_auc",
        "brier",
        "ece",
    ]
    summary = (
        fold_report.groupby("condition")[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    comparisons = []
    baseline = fold_report[fold_report["condition"] == "target_only"].set_index(
        ["repeat", "fold"]
    )
    for condition in [
        "modma_zero_shot",
        "modma_probability_stacking",
        "joint_source_target",
    ]:
        current = fold_report[fold_report["condition"] == condition].set_index(
            ["repeat", "fold"]
        )
        paired = current[["auc", "brier"]].join(
            baseline[["auc", "brier"]],
            lsuffix="_candidate",
            rsuffix="_baseline",
        ).dropna()
        auc_delta = (
            paired["auc_candidate"] - paired["auc_baseline"]
        ).to_numpy()
        brier_delta = (
            paired["brier_candidate"] - paired["brier_baseline"]
        ).to_numpy()
        auc_ci = _bootstrap_delta_ci(auc_delta)
        brier_ci = _bootstrap_delta_ci(brier_delta)
        comparisons.append(
            {
                "condition": condition,
                "reference": "target_only",
                "n_paired_folds": len(paired),
                "auc_delta_mean": float(auc_delta.mean()),
                "auc_delta_ci_low": auc_ci[0],
                "auc_delta_ci_high": auc_ci[1],
                "brier_delta_mean": float(brier_delta.mean()),
                "brier_delta_ci_low": brier_ci[0],
                "brier_delta_ci_high": brier_ci[1],
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    best_transfer_auc = (
        float(
            comparison_frame.loc[
                comparison_frame["auc_delta_mean"].idxmax(),
                "auc_delta_mean",
            ]
        )
        if not comparison_frame.empty
        else 0.0
    )
    meta = {
        "source": "MODMA",
        "target": "MPDD",
        "source_subjects": int(len(source)),
        "target_subjects": int(len(target)),
        "shared_features": len(common_features),
        "source_weight": source_weight,
        "splits": n_splits,
        "repeats": n_repeats,
        "same_target_splits": True,
        "best_transfer_auc_delta": best_transfer_auc,
        "recommended_for_main_model": bool(best_transfer_auc > 0.01),
        "note": (
            "仅当目标域配对切分上的增量稳定且不是单一折偶然结果时，"
            "才考虑接入主模型。"
        ),
    }
    return fold_report, comparison_frame, meta
