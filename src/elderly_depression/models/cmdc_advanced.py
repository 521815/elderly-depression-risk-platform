"""Advanced, subject-aware CMDC validation analyses."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


AUDIO_MODALITIES = ("audio_deep", "acoustic")
SINGLE_MODALITIES = {
    "text": ("text",),
    "visual": ("visual",),
    "audio": AUDIO_MODALITIES,
}


def _modality_columns(df: pd.DataFrame, modalities: tuple[str, ...]) -> list[str]:
    return [
        column
        for column in df.columns
        if any(
            column.startswith(f"{modality}_")
            for modality in modalities
        )
        and not column.endswith("_questions")
    ]


def _prepare_subset(
    df: pd.DataFrame,
    modalities: tuple[str, ...],
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    columns = _modality_columns(df, modalities)
    if not columns:
        return df.iloc[0:0].copy(), np.asarray([], dtype=int), []
    available = df[columns].notna().any(axis=1)
    for modality in modalities:
        modality_group = _modality_columns(df, (modality,))
        if not modality_group:
            return df.iloc[0:0].copy(), np.asarray([], dtype=int), []
        available &= df[modality_group].notna().any(axis=1)
    subset = df.loc[available].copy().reset_index(drop=True)
    labels = pd.to_numeric(subset.get("label"), errors="coerce")
    valid = labels.notna()
    subset = subset.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].astype(int)
    return subset, labels.to_numpy(dtype=int), columns


def _logistic_model() -> object:
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


def _repeated_oof(
    subset: pd.DataFrame,
    labels: np.ndarray,
    columns: list[str],
    repeats: int,
) -> tuple[np.ndarray, int]:
    """Return one out-of-fold probability per subject and repeat."""

    folds = min(5, int(pd.Series(labels).value_counts().min()))
    cv = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=max(1, repeats),
        random_state=42,
    )
    matrix = np.full((max(1, repeats), len(subset)), np.nan)
    X = subset[columns].to_numpy(dtype=np.float64)
    for split_index, (train_index, test_index) in enumerate(cv.split(X, labels)):
        repeat_index = split_index // folds
        if len(np.unique(labels[train_index])) < 2:
            continue
        model = _logistic_model()
        model.fit(X[train_index], labels[train_index])
        matrix[repeat_index, test_index] = model.predict_proba(
            X[test_index]
        )[:, 1]
    return matrix, folds


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = (probabilities >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, prediction)
        ),
        "sensitivity": float(
            recall_score(
                labels,
                prediction,
                pos_label=1,
                zero_division=0,
            )
        ),
        "specificity": float(
            recall_score(
                labels,
                prediction,
                pos_label=0,
                zero_division=0,
            )
        ),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def _bootstrap_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    iterations: int,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    values: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(max(50, iterations)):
        sample = rng.integers(0, len(labels), len(labels))
        y_sample = labels[sample]
        if len(np.unique(y_sample)) < 2:
            continue
        result = _metrics(y_sample, probabilities[sample])
        for name, value in result.items():
            values[name].append(value)
    output = {}
    for name, items in values.items():
        output[name] = {
            "mean": float(np.mean(items)),
            "ci_low": float(np.percentile(items, 2.5)),
            "ci_high": float(np.percentile(items, 97.5)),
        }
    return output


def _paired_bootstrap_difference(
    labels: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    iterations: int,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(max(50, iterations)):
        sample = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[sample])) < 2:
            continue
        auc_a = roc_auc_score(labels[sample], probabilities_a[sample])
        auc_b = roc_auc_score(labels[sample], probabilities_b[sample])
        differences.append(float(auc_a - auc_b))
    return {
        "auc_difference": float(
            roc_auc_score(labels, probabilities_a)
            - roc_auc_score(labels, probabilities_b)
        ),
        "ci_low": float(np.percentile(differences, 2.5)),
        "ci_high": float(np.percentile(differences, 97.5)),
        "positive_bootstrap_rate": float(np.mean(np.asarray(differences) > 0)),
    }


def _paired_repeat_comparison(
    labels: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
) -> dict:
    auc_a = [
        roc_auc_score(labels, probabilities_a[index])
        for index in range(len(probabilities_a))
    ]
    auc_b = [
        roc_auc_score(labels, probabilities_b[index])
        for index in range(len(probabilities_b))
    ]
    differences = np.asarray(auc_a) - np.asarray(auc_b)
    p_value = float(ttest_rel(auc_a, auc_b).pvalue)
    return {
        "paired_repeat_mean_auc_difference": float(np.mean(differences)),
        "paired_repeat_sd": float(np.std(differences, ddof=1))
        if len(differences) > 1
        else 0.0,
        "positive_repeats": int((differences > 0).sum()),
        "total_repeats": int(len(differences)),
        "paired_t_p_value": p_value,
    }


def _paired_experiments(
    df: pd.DataFrame,
    repeats: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    matrices = {}

    common77_modalities = ("text", "audio_deep", "acoustic")
    subset77, labels77, _ = _prepare_subset(df, common77_modalities)
    group77_variants = {
        "text": ("text",),
        "acoustic": ("acoustic",),
        "audio": AUDIO_MODALITIES,
        "text_audio": common77_modalities,
    }
    for name, modalities in group77_variants.items():
        columns = _modality_columns(subset77, modalities)
        matrices[("77", name)] = _repeated_oof(
            subset77,
            labels77,
            columns,
            repeats,
        )[0]
    for comparator in ("text", "acoustic", "audio"):
        a = matrices[("77", "text_audio")]
        b = matrices[("77", comparator)]
        mean_a = np.nanmean(a, axis=0)
        mean_b = np.nanmean(b, axis=0)
        row = {
            "group": "text_audio_subset",
            "subjects": int(len(subset77)),
            "model_a": "text_audio",
            "model_b": comparator,
            "auc_a": float(roc_auc_score(labels77, mean_a)),
            "auc_b": float(roc_auc_score(labels77, mean_b)),
        }
        row.update(
            _paired_bootstrap_difference(
                labels77,
                mean_a,
                mean_b,
                bootstrap_iterations,
            )
        )
        row.update(_paired_repeat_comparison(labels77, a, b))
        rows.append(row)

    subset44, labels44, _ = _prepare_subset(
        df,
        ("text", "visual", "audio_deep", "acoustic"),
    )
    group44_variants = {
        "text": ("text",),
        "visual": ("visual",),
        "audio": AUDIO_MODALITIES,
        "text_audio": common77_modalities,
        "complete_multimodal": (
            "text",
            "visual",
            "audio_deep",
            "acoustic",
        ),
    }
    for name, modalities in group44_variants.items():
        columns = _modality_columns(subset44, modalities)
        matrices[("44", name)] = _repeated_oof(
            subset44,
            labels44,
            columns,
            repeats,
        )[0]
    for comparator in ("text_audio", "visual"):
        a = matrices[("44", "complete_multimodal")]
        b = matrices[("44", comparator)]
        mean_a = np.nanmean(a, axis=0)
        mean_b = np.nanmean(b, axis=0)
        row = {
            "group": "complete_multimodal_subset",
            "subjects": int(len(subset44)),
            "model_a": "complete_multimodal",
            "model_b": comparator,
            "auc_a": float(roc_auc_score(labels44, mean_a)),
            "auc_b": float(roc_auc_score(labels44, mean_b)),
        }
        row.update(
            _paired_bootstrap_difference(
                labels44,
                mean_a,
                mean_b,
                bootstrap_iterations,
            )
        )
        row.update(_paired_repeat_comparison(labels44, a, b))
        rows.append(row)

    return pd.DataFrame(rows), {
        "text_audio_group_subjects": int(len(subset77)),
        "complete_multimodal_group_subjects": int(len(subset44)),
        "probability_matrices": {
            f"{group}:{model_name}": matrix.shape
            for (group, model_name), matrix in matrices.items()
        },
    }


def _dynamic_fusion_predictions(
    df: pd.DataFrame,
    repeats: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    valid = pd.to_numeric(df.get("label"), errors="coerce").notna()
    subset = df.loc[valid].reset_index(drop=True)
    labels = pd.to_numeric(subset["label"], errors="coerce").astype(int).to_numpy()
    subject_ids = subset["subject_id"].astype(str).tolist()
    folds = min(5, int(pd.Series(labels).value_counts().min()))
    cv = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=max(1, repeats),
        random_state=42,
    )
    modality_probabilities = {
        name: np.full((max(1, repeats), len(subset)), np.nan)
        for name in SINGLE_MODALITIES
    }
    for split_index, (train_index, test_index) in enumerate(cv.split(subset, labels)):
        repeat_index = split_index // folds
        train = subset.iloc[train_index]
        test = subset.iloc[test_index]
        train_labels = labels[train_index]
        for name, modalities in SINGLE_MODALITIES.items():
            columns = _modality_columns(subset, modalities)
            train_available = train[columns].notna().any(axis=1)
            test_available = test[columns].notna().any(axis=1)
            if train_available.sum() < 8 or not test_available.any():
                continue
            y_train = train_labels[train_available.to_numpy()]
            if len(np.unique(y_train)) < 2:
                continue
            model = _logistic_model()
            X_train = train.loc[train_available, columns].to_numpy(dtype=np.float64)
            X_test = test.loc[test_available, columns].to_numpy(dtype=np.float64)
            model.fit(X_train, y_train)
            positions = np.flatnonzero(test_available.to_numpy())
            modality_probabilities[name][
                repeat_index,
                test_index[positions],
            ] = model.predict_proba(X_test)[:, 1]

    dynamic = np.full_like(modality_probabilities["text"], np.nan)
    for index in range(dynamic.shape[1]):
        values = [
            matrix[:, index]
            for matrix in modality_probabilities.values()
        ]
        dynamic[:, index] = np.nanmean(np.vstack(values), axis=0)
    return labels, dynamic, modality_probabilities, subject_ids


def _mean_oof(matrix: np.ndarray) -> np.ndarray:
    return np.nanmean(matrix, axis=0)


def _missing_modality_analysis(
    df: pd.DataFrame,
    repeats: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, dict]:
    labels, dynamic_all_matrix, modality_matrices, subject_ids = (
        _dynamic_fusion_predictions(df, repeats)
    )
    complete_subset, complete_labels, complete_columns = _prepare_subset(
        df,
        ("text", "visual", "audio_deep", "acoustic"),
    )
    fixed_matrix = _repeated_oof(
        complete_subset,
        complete_labels,
        complete_columns,
        repeats,
    )[0]
    complete_ids = set(complete_subset["subject_id"].astype(str))
    complete_positions = [
        index
        for index, subject_id in enumerate(subject_ids)
        if subject_id in complete_ids
    ]
    dynamic_complete_matrix = dynamic_all_matrix[:, complete_positions]

    rows = []
    configurations = [
        ("available_modality_dynamic_fusion", labels, _mean_oof(dynamic_all_matrix)),
        (
            "dynamic_fusion_complete_subjects",
            labels[complete_positions],
            _mean_oof(dynamic_complete_matrix),
        ),
        (
            "fixed_complete_multimodal",
            complete_labels,
            _mean_oof(fixed_matrix),
        ),
    ]
    for name, y, probability in configurations:
        row = {
            "strategy": name,
            "subjects": int(len(y)),
            "features_or_modalities": (
                "text+visual+audio available"
                if "dynamic" in name
                else "text+visual+audio_deep+acoustic"
            ),
        }
        row.update(_metrics(y, probability))
        bootstrap = _bootstrap_ci(
            y,
            probability,
            bootstrap_iterations,
        )
        for metric, values in bootstrap.items():
            row[f"{metric}_ci_low"] = values["ci_low"]
            row[f"{metric}_ci_high"] = values["ci_high"]
        rows.append(row)

    y_common = labels[complete_positions]
    dynamic_common = _mean_oof(dynamic_complete_matrix)
    fixed_common = _mean_oof(fixed_matrix)
    comparison = _paired_bootstrap_difference(
        y_common,
        dynamic_common,
        fixed_common,
        bootstrap_iterations,
    )
    comparison.update(
        _paired_repeat_comparison(
            y_common,
            dynamic_complete_matrix,
            fixed_matrix,
        )
    )
    return pd.DataFrame(rows), comparison


def _nested_model_specs() -> dict[str, tuple[object, dict]]:
    return {
        "logistic_l2": (
            make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=False),
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                ),
            ),
            {"logisticregression__C": [0.003, 0.01, 0.03]},
        ),
        "random_forest": (
            make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=False),
                RandomForestClassifier(
                    n_estimators=180,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
            {
                "randomforestclassifier__max_depth": [3, 5, None],
                "randomforestclassifier__min_samples_leaf": [1, 3],
            },
        ),
        "extra_trees": (
            make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=False),
                ExtraTreesClassifier(
                    n_estimators=220,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
            {
                "extratreesclassifier__max_depth": [4, 8, None],
                "extratreesclassifier__min_samples_leaf": [1, 3],
            },
        ),
    }


def _model_comparison(
    df: pd.DataFrame,
    repeats: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, object | None, str, list[str]]:
    subset, labels, columns = _prepare_subset(
        df,
        ("text", "audio_deep", "acoustic"),
    )
    X = subset[columns].to_numpy(dtype=np.float64)
    folds = min(5, int(pd.Series(labels).value_counts().min()))
    outer = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=max(1, repeats),
        random_state=42,
    )
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rows = []
    best_model = None
    best_name = ""
    best_auc = -1.0
    for name, (estimator, params) in _nested_model_specs().items():
        probabilities = np.full((max(1, repeats), len(subset)), np.nan)
        for split_index, (train_index, test_index) in enumerate(outer.split(X, labels)):
            repeat_index = split_index // folds
            search = GridSearchCV(
                estimator,
                params,
                scoring="roc_auc",
                cv=inner,
                n_jobs=1,
                refit=True,
            )
            search.fit(X[train_index], labels[train_index])
            probabilities[repeat_index, test_index] = search.predict_proba(
                X[test_index]
            )[:, 1]
        mean_probability = _mean_oof(probabilities)
        metrics = _metrics(labels, mean_probability)
        row = {
            "model": name,
            "subjects": int(len(labels)),
            "features": int(len(columns)),
        }
        row.update(metrics)
        bootstrap = _bootstrap_ci(
            labels,
            mean_probability,
            bootstrap_iterations,
        )
        for metric, values in bootstrap.items():
            row[f"{metric}_ci_low"] = values["ci_low"]
            row[f"{metric}_ci_high"] = values["ci_high"]
        rows.append(row)
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_name = name
            best_model = clone(estimator)
            best_model.fit(X, labels)
    return pd.DataFrame(rows), best_model, best_name, columns


def _calibration_and_decision_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed, predicted = calibration_curve(
        labels,
        probabilities,
        n_bins=5,
        strategy="quantile",
    )
    calibration = pd.DataFrame(
        {
            "predicted_frequency": predicted,
            "observed_frequency": observed,
        }
    )
    prevalence = float(np.mean(labels))
    rows = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        prediction = probabilities >= threshold
        tp = int(((prediction == 1) & (labels == 1)).sum())
        fp = int(((prediction == 1) & (labels == 0)).sum())
        n = len(labels)
        net_benefit = tp / n - fp / n * (threshold / (1 - threshold))
        treat_all = (
            prevalence
            - (1 - prevalence) * (threshold / (1 - threshold))
        )
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit": float(net_benefit),
                "treat_all_net_benefit": float(treat_all),
                "treat_none_net_benefit": 0.0,
            }
        )
    return calibration, pd.DataFrame(rows)


def run_cmdc_advanced_analysis(
    df: pd.DataFrame,
    output_dir: Path | str,
    *,
    repeats: int = 10,
    model_repeats: int = 5,
    bootstrap_iterations: int = 1000,
) -> dict:
    if df.empty or "label" not in df.columns:
        raise ValueError("CMDC 特征表为空或缺少 label")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paired_report, paired_meta = _paired_experiments(
        df,
        repeats,
        bootstrap_iterations,
    )
    missing_report, missing_comparison = _missing_modality_analysis(
        df,
        repeats,
        bootstrap_iterations,
    )
    model_report, best_model, best_model_name, model_columns = _model_comparison(
        df,
        model_repeats,
        bootstrap_iterations,
    )

    primary_subset, primary_labels, primary_columns = _prepare_subset(
        df,
        ("text", "audio_deep", "acoustic"),
    )
    primary_matrix = _repeated_oof(
        primary_subset,
        primary_labels,
        primary_columns,
        repeats,
    )[0]
    primary_probability = _mean_oof(primary_matrix)
    bootstrap = _bootstrap_ci(
        primary_labels,
        primary_probability,
        bootstrap_iterations,
    )
    calibration, decision_curve = _calibration_and_decision_curves(
        primary_labels,
        primary_probability,
    )

    paired_path = output / "cmdc_paired_comparison.csv"
    missing_path = output / "cmdc_missing_modality_report.csv"
    model_path = output / "cmdc_model_comparison.csv"
    calibration_path = output / "cmdc_calibration_curve.csv"
    decision_path = output / "cmdc_decision_curve.csv"
    summary_path = output / "cmdc_advanced_summary.json"
    for frame, path in (
        (paired_report, paired_path),
        (missing_report, missing_path),
        (model_report, model_path),
        (calibration, calibration_path),
        (decision_curve, decision_path),
    ):
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        path.chmod(0o600)

    model_artifact = None
    if best_model is not None:
        model_artifact = output / "cmdc_nested_best_model.pkl"
        joblib.dump(
            {
                "model": best_model,
                "model_name": best_model_name,
                "features": model_columns,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "note": "仅用于 CMDC 外部验证，不进入老年主风险评估。",
            },
            model_artifact,
        )
        model_artifact.chmod(0o600)

    summary = {
        "ready": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subjects": int(len(df)),
        "primary_subjects": int(len(primary_subset)),
        "bootstrap_iterations": int(max(50, bootstrap_iterations)),
        "primary_bootstrap_ci": bootstrap,
        "paired_comparison": paired_meta,
        "missing_modality_comparison": missing_comparison,
        "best_nested_model": best_model_name,
        "best_nested_model_artifact": str(model_artifact) if model_artifact else "",
        "reports": {
            "paired": str(paired_path),
            "missing_modality": str(missing_path),
            "model_comparison": str(model_path),
            "calibration": str(calibration_path),
            "decision_curve": str(decision_path),
        },
        "implemented_measures": [
            "同一受试者上的配对 Bootstrap AUC 差异",
            "重复折级配对检验",
            "Bootstrap 95% 置信区间",
            "可用模态动态融合与缺失模态分析",
            "嵌套调参的 Logistic、Random Forest、Extra Trees 比较",
            "校准曲线与决策曲线",
        ],
        "not_yet_feasible": [
            "原始文本/语音编码器跨域预训练：当前只有派生特征，缺少可复现的编码器训练输入",
            "老年外部验证：CMDC 为混合成人且缺少年龄和共病元数据",
            "严重程度回归：缺少结构化 PHQ-9/HAMD 分数",
        ],
        "evidence_boundary": (
            "探索性小样本分析；重复交叉验证和 Bootstrap 不能替代独立外部测试集。"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.chmod(0o600)
    return summary
