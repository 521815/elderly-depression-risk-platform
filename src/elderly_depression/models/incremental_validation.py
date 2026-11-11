"""Theory baselines, modality ablations, and cross-source evidence audits."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE
from ..data.mpdd import add_mpdd_binary_target
from ..evaluation import imbalanced_metrics_frame
from .fusion import expected_calibration_error
from .multimodal import _fit_modality


CHARLS_DEMOGRAPHIC_FEATURES = [
    "age",
    "gender",
    "education",
    "marital_status",
]
CHARLS_SOCIOECONOMIC_FEATURES = [
    "income_earn",
    "income_pension",
    "social_work",
]
CHARLS_HEALTH_FEATURES = [
    "self_rated_health",
    "adl_score",
    "mobility_score",
    "sleep_quality",
    "comorbidity_count",
]
CHARLS_LIFESTYLE_FEATURES = [
    "vigorous_activity",
    "moderate_activity",
    "light_activity",
    "smoker",
    "drinker",
]
CHARLS_COGNITIVE_FEATURES = [
    "self_memory",
    "immediate_recall",
    "delayed_recall",
    "serial7",
]
CHARLS_CHRONIC_FEATURES = [
    "chronic_hibpe",
    "chronic_diabe",
    "chronic_cancre",
    "chronic_lunge",
    "chronic_hearte",
    "chronic_stroke",
    "chronic_psyche",
    "chronic_arthre",
    "chronic_dyslipe",
    "chronic_livere",
    "chronic_kidneye",
    "chronic_digeste",
    "chronic_asthmae",
    "chronic_memrye",
]
MPDD_PREFIXES = ["face", "audio", "gait", "personality"]


def _available(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _pipeline(model) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        model,
    )


def _tree_pipeline(model) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        model,
    )


def _metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    row = imbalanced_metrics_frame(y_true, predictions, probabilities).iloc[0]
    return {
        **{key: float(value) for key, value in row.to_dict().items()},
        "ece": float(expected_calibration_error(y_true, probabilities)),
    }


def _bootstrap_group_metrics(
    groups: pd.Series,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bootstrap: int = 300,
    seed: int = RANDOM_STATE,
) -> dict:
    rng = np.random.default_rng(seed)
    group_values = groups.astype(str).to_numpy()
    unique_groups = np.unique(group_values)
    positions = {
        group: np.flatnonzero(group_values == group) for group in unique_groups
    }
    values: dict[str, list[float]] = {
        "auc": [],
        "pr_auc": [],
        "brier": [],
        "f1": [],
    }
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        y = y_true[indices]
        if np.unique(y).size < 2:
            continue
        prob = probabilities[indices]
        pred = (prob >= 0.5).astype(int)
        values["auc"].append(float(roc_auc_score(y, prob)))
        values["pr_auc"].append(float(average_precision_score(y, prob)))
        values["brier"].append(float(brier_score_loss(y, prob)))
        values["f1"].append(float(f1_score(y, pred, zero_division=0)))
    return {
        f"{metric}_ci_low": float(np.percentile(samples, 2.5))
        for metric, samples in values.items()
    } | {
        f"{metric}_ci_high": float(np.percentile(samples, 97.5))
        for metric, samples in values.items()
    }


def _bootstrap_group_delta(
    groups: pd.Series,
    y_true: np.ndarray,
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    n_bootstrap: int = 300,
    seed: int = RANDOM_STATE,
) -> dict:
    rng = np.random.default_rng(seed + 17)
    group_values = groups.astype(str).to_numpy()
    unique_groups = np.unique(group_values)
    positions = {
        group: np.flatnonzero(group_values == group) for group in unique_groups
    }
    values: dict[str, list[float]] = {
        "auc_delta": [],
        "pr_auc_delta": [],
        "brier_delta": [],
    }
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        y = y_true[indices]
        if np.unique(y).size < 2:
            continue
        candidate = candidate_probabilities[indices]
        reference = reference_probabilities[indices]
        values["auc_delta"].append(
            float(roc_auc_score(y, candidate) - roc_auc_score(y, reference))
        )
        values["pr_auc_delta"].append(
            float(
                average_precision_score(y, candidate)
                - average_precision_score(y, reference)
            )
        )
        values["brier_delta"].append(
            float(
                brier_score_loss(y, candidate)
                - brier_score_loss(y, reference)
            )
        )
    result = {}
    for metric, samples in values.items():
        result[f"{metric}_ci_low"] = float(np.percentile(samples, 2.5))
        result[f"{metric}_ci_high"] = float(np.percentile(samples, 97.5))
    return result


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.ones(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, float(p_values[index]) * (count - rank))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _paired_wilcoxon_p(differences: np.ndarray) -> float:
    """Return a defined p value for paired differences.

    SciPy emits a runtime warning when every paired difference is zero. A
    constant zero difference provides no evidence of a shift, so its two-sided
    p value is one by definition.
    """
    values = np.asarray(differences, dtype=float)
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    from scipy.stats import wilcoxon

    try:
        return float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def run_charls_theory_baselines(
    df: pd.DataFrame,
    n_splits: int = 5,
    n_bootstrap: int = 300,
) -> dict:
    """Evaluate pre-specified theory baselines with subject-grouped CV."""
    data = df.dropna(subset=["depression", "id"]).copy()
    data["id"] = data["id"].astype(str)
    y = data["depression"].astype(int)
    groups = data["id"]

    demographic = _available(data, CHARLS_DEMOGRAPHIC_FEATURES)
    theory_core = (
        demographic
        + _available(data, CHARLS_SOCIOECONOMIC_FEATURES)
        + _available(data, CHARLS_HEALTH_FEATURES)
        + _available(data, CHARLS_LIFESTYLE_FEATURES)
    )
    theory_full = (
        theory_core
        + _available(data, CHARLS_CHRONIC_FEATURES)
        + _available(data, CHARLS_COGNITIVE_FEATURES)
    )

    specs = [
        (
            "B0",
            "prevalence",
            "population_prevalence",
            [],
            None,
        ),
        (
            "B1",
            "logistic",
            "demographic",
            demographic,
            "logistic",
        ),
        (
            "B2",
            "logistic",
            "theory_core",
            theory_core,
            "logistic",
        ),
        (
            "B3",
            "logistic",
            "theory_full",
            theory_full,
            "logistic",
        ),
        (
            "B4",
            "random_forest",
            "theory_full",
            theory_full,
            "forest",
        ),
    ]

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_rows = []
    oof_by_model: dict[str, np.ndarray] = {}
    for baseline_id, model_name, feature_set, features, family in specs:
        oof_by_model[baseline_id] = np.full(len(data), np.nan, dtype=float)

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(data, y, groups)
    ):
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        for baseline_id, model_name, feature_set, features, family in specs:
            if baseline_id == "B0":
                probabilities = np.full(len(val_idx), float(y_train.mean()))
            elif family == "logistic":
                model = _pipeline(
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    )
                )
                model.fit(data.iloc[train_idx][features], y_train)
                probabilities = model.predict_proba(
                    data.iloc[val_idx][features]
                )[:, 1]
            elif family == "forest":
                model = _tree_pipeline(
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    )
                )
                model.fit(data.iloc[train_idx][features], y_train)
                probabilities = model.predict_proba(
                    data.iloc[val_idx][features]
                )[:, 1]
            else:
                raise ValueError(f"Unknown model family: {family}")

            probabilities = np.asarray(probabilities, dtype=float)
            oof_by_model[baseline_id][val_idx] = probabilities
            metrics = _metrics(y_val.to_numpy(), probabilities)
            fold_rows.append(
                {
                    "dataset": "CHARLS",
                    "baseline_id": baseline_id,
                    "model": model_name,
                    "feature_set": feature_set,
                    "n_features": len(features),
                    "fold": fold,
                    **metrics,
                    "threshold": 0.5,
                    "n_train_rows": len(train_idx),
                    "n_validation_rows": len(val_idx),
                    "n_train_subjects": groups.iloc[train_idx].nunique(),
                    "n_validation_subjects": groups.iloc[val_idx].nunique(),
                }
            )

    fold_report = pd.DataFrame(fold_rows)
    metric_cols = [
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
    summary_rows = []
    for baseline_id, group in fold_report.groupby("baseline_id", sort=False):
        first = group.iloc[0]
        probabilities = oof_by_model[baseline_id]
        oof = _metrics(y.to_numpy(), probabilities)
        row = {
            "dataset": "CHARLS",
            "baseline_id": baseline_id,
            "model": first["model"],
            "feature_set": first["feature_set"],
            "n_features": int(first["n_features"]),
            "n_rows": len(data),
            "n_subjects": int(groups.nunique()),
            "positive_rate": float(y.mean()),
            "threshold": 0.5,
        }
        row.update(oof)
        for metric in metric_cols:
            row[f"{metric}_fold_mean"] = float(group[metric].mean())
            row[f"{metric}_fold_std"] = float(group[metric].std())
        row.update(
            _bootstrap_group_metrics(
                groups,
                y.to_numpy(),
                probabilities,
                n_bootstrap=n_bootstrap,
            )
        )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    comparisons = []
    for reference_id, candidate_id in [
        ("B0", "B1"),
        ("B1", "B2"),
        ("B2", "B3"),
        ("B3", "B4"),
    ]:
        reference = oof_by_model[reference_id]
        candidate = oof_by_model[candidate_id]
        row = {
            "dataset": "CHARLS",
            "reference": reference_id,
            "candidate": candidate_id,
            "auc_delta": float(
                roc_auc_score(y, candidate) - roc_auc_score(y, reference)
            ),
            "pr_auc_delta": float(
                average_precision_score(y, candidate)
                - average_precision_score(y, reference)
            ),
            "brier_delta": float(
                brier_score_loss(y, candidate)
                - brier_score_loss(y, reference)
            ),
        }
        row.update(
            _bootstrap_group_delta(
                groups,
                y.to_numpy(),
                candidate,
                reference,
                n_bootstrap=n_bootstrap,
            )
        )
        comparisons.append(row)
    return {
        "fold_report": fold_report,
        "summary": summary,
        "comparisons": pd.DataFrame(comparisons),
        "oof_predictions": pd.DataFrame(
            {
                "id": groups,
                "depression": y,
                **{
                    f"{baseline_id}_probability": probabilities
                    for baseline_id, probabilities in oof_by_model.items()
                },
            }
        ),
        "feature_sets": {
            "demographic": demographic,
            "theory_core": theory_core,
            "theory_full": theory_full,
            "excluded": [
                "cesd_score",
                "depression",
                "wave",
                "id",
            ],
        },
    }


def run_mpdd_modality_ablation(
    subject_df: pd.DataFrame,
    prefixes: list[str] | None = None,
    pca_dims: int = 64,
    n_splits: int = 5,
    n_repeats: int = 5,
) -> dict:
    """Evaluate all non-empty modality subsets with subject-grouped CV."""
    from itertools import combinations

    prefixes = prefixes or MPDD_PREFIXES
    data = add_mpdd_binary_target(subject_df)
    data = data.dropna(subset=["mpdd_binary"]).copy()
    data["subject_id"] = data["subject_id"].astype(str)
    y = data["mpdd_binary"].astype(int).to_numpy()
    groups = data["subject_id"]
    subsets = [
        subset
        for size in range(1, len(prefixes) + 1)
        for subset in combinations(prefixes, size)
    ]
    fold_rows = []

    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )
        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(data, y, groups)
        ):
            y_train = y[train_idx]
            y_val = y[val_idx]
            prevalence = float(y_train.mean())
            fold_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "modalities": "prevalence",
                    "n_modalities": 0,
                    **_metrics(y_val, np.full(len(val_idx), prevalence)),
                    "threshold": 0.5,
                    "n_train_subjects": int(
                        groups.iloc[train_idx].nunique()
                    ),
                    "n_validation_subjects": int(
                        groups.iloc[val_idx].nunique()
                    ),
                }
            )

            modality_probabilities: dict[str, np.ndarray] = {}
            for prefix in prefixes:
                columns = [
                    column
                    for column in data.columns
                    if column.startswith(prefix + "_")
                ]
                if not columns:
                    continue
                x_all = data[columns].fillna(0.0).to_numpy(dtype=float)
                scaler, pca, model = _fit_modality(
                    x_all[train_idx],
                    pd.Series(y_train),
                    pca_dims,
                )
                modality_probabilities[prefix] = model.predict_proba(
                    pca.transform(scaler.transform(x_all[val_idx]))
                )[:, 1]

            for subset in subsets:
                if not all(prefix in modality_probabilities for prefix in subset):
                    continue
                probabilities = np.mean(
                    np.column_stack(
                        [modality_probabilities[prefix] for prefix in subset]
                    ),
                    axis=1,
                )
                fold_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "modalities": "+".join(subset),
                        "n_modalities": len(subset),
                        **_metrics(y_val, probabilities),
                        "threshold": 0.5,
                        "n_train_subjects": int(
                            groups.iloc[train_idx].nunique()
                        ),
                        "n_validation_subjects": int(
                            groups.iloc[val_idx].nunique()
                        ),
                    }
                )

    fold_report = pd.DataFrame(fold_rows)
    metric_cols = [
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
        fold_report.groupby(["modalities", "n_modalities"], sort=False)[
            metric_cols
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.insert(0, "dataset", "MPDD")
    summary["threshold"] = 0.5
    comparison_rows = []
    full_label = "+".join(prefixes)
    full_fold = fold_report[
        fold_report["modalities"] == full_label
    ].set_index(["repeat", "fold"])
    for modality_set, group in fold_report.groupby("modalities", sort=False):
        if modality_set == full_label:
            continue
        current = group.set_index(["repeat", "fold"])
        paired = current[["auc", "brier"]].join(
            full_fold[["auc", "brier"]],
            lsuffix="_candidate",
            rsuffix="_full",
        ).dropna()
        if paired.empty:
            continue
        auc_delta = (
            paired["auc_candidate"] - paired["auc_full"]
        ).to_numpy()
        brier_delta = (
            paired["brier_candidate"] - paired["brier_full"]
        ).to_numpy()
        auc_p = _paired_wilcoxon_p(auc_delta)
        brier_p = _paired_wilcoxon_p(brier_delta)
        comparison_rows.append(
            {
                "dataset": "MPDD",
                "candidate": modality_set,
                "reference": full_label,
                "n_paired_folds": len(paired),
                "auc_delta_candidate_minus_full": float(auc_delta.mean()),
                "brier_delta_candidate_minus_full": float(brier_delta.mean()),
                "auc_wilcoxon_p_raw": auc_p,
                "brier_wilcoxon_p_raw": brier_p,
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons["auc_wilcoxon_p_holm"] = _holm_adjust(
            comparisons["auc_wilcoxon_p_raw"].tolist()
        )
        comparisons["brier_wilcoxon_p_holm"] = _holm_adjust(
            comparisons["brier_wilcoxon_p_raw"].tolist()
        )
    return {
        "fold_report": fold_report,
        "summary": summary,
        "comparisons": comparisons,
        "fine_tuning_boundary": {
            "clinical_end_to_end_finetuning": "not_estimable",
            "reason": (
                "当前仅有 87 名有官方标签的老年受试者，缺少独立锁定测试集；"
                "全参数微调或把普通分类器称为大模型微调会造成证据过度声明。"
            ),
            "available_adaptation": (
                "本批 MPDD 模态消融使用 PCA+L2 线性分类头，CHARLS 另比较了"
                "随机森林；真正的冻结编码器、LoRA 和全参数微调必须等"
                "同人配对数据和独立锁定测试集到位后再运行。"
            ),
        },
    }


def cross_source_audit(
    charls: pd.DataFrame,
    mpdd: pd.DataFrame,
    cmdc: pd.DataFrame | None = None,
) -> dict:
    """Document which theory constructs exist in each source."""
    def present(frame: pd.DataFrame | None, columns: list[str]) -> dict:
        if frame is None:
            return {column: False for column in columns}
        return {column: column in frame.columns for column in columns}

    sources = {
        "CHARLS": {
            "subjects": (
                int(charls["id"].nunique())
                if "id" in charls.columns
                else len(charls)
            ),
            "outcome": "CESD-10 >= 10",
            "paired_multimodal": False,
            "fields": present(
                charls,
                [
                    "age",
                    "gender",
                    "education",
                    "comorbidity_count",
                    "sleep_quality",
                    "adl_score",
                    "mobility_score",
                    "social_work",
                    "cesd_score",
                ],
            ),
        },
        "MPDD": {
            "subjects": (
                int(mpdd["subject_id"].nunique())
                if "subject_id" in mpdd.columns
                else len(mpdd)
            ),
            "outcome": "official label2",
            "paired_multimodal": True,
            "fields": {
                "label2": "label2" in mpdd.columns,
                "label3": "label3" in mpdd.columns,
                "phq9_score": "phq9_score" in mpdd.columns,
                "age": "age" in mpdd.columns,
                "sex": "sex" in mpdd.columns or "gender" in mpdd.columns,
                "education": "education" in mpdd.columns,
                "comorbidity_count": "comorbidity_count" in mpdd.columns,
                "sleep_quality": "sleep_quality" in mpdd.columns,
                "social_support": (
                    "social_support" in mpdd.columns
                    or "social_work" in mpdd.columns
                ),
                **{
                    prefix: any(
                        column.startswith(prefix + "_")
                        for column in mpdd.columns
                    )
                    for prefix in MPDD_PREFIXES
                },
            },
        },
    }
    if cmdc is not None:
        sources["CMDC"] = {
            "subjects": (
                int(cmdc["subject_id"].nunique())
                if "subject_id" in cmdc.columns
                else len(cmdc)
            ),
            "outcome": "HC / MDD",
            "paired_multimodal": True,
            "fields": present(
                cmdc,
                [
                    "label",
                    "subject_id",
                    "age",
                    "sex",
                    "education",
                    "comorbidity_count",
                    "sleep_quality",
                    "social_support",
                ],
            )
            | {
                prefix: any(
                    column.startswith(prefix + "_")
                    for column in cmdc.columns
                )
                for prefix in ["text", "visual", "audio_deep", "acoustic"]
            },
        }

    return {
        "sources": sources,
        "pairing": {
            "CHARLS__MPDD": False,
            "CHARLS__CMDC": False,
            "reason": "不同研究、不同受试者，且标签与采集协议不一致。",
        },
        "allowed": [
            "各数据源内部训练和受试者级验证",
            "同源配对数据内部进行模态消融",
            "不同队列之间只做概率校准、领域适配和迁移验证",
        ],
        "forbidden": [
            "将不同受试者的表格、语音和面部特征按行拼接",
            "使用随机配对或模拟配对证明同人级多模态融合有效",
            "把 CESD-10、PHQ-9 和临床诊断当作完全等价的标签",
        ],
    }


def run_incremental_validation(
    charls: pd.DataFrame,
    mpdd_subject: pd.DataFrame,
    cmdc: pd.DataFrame | None,
    output_dir: Path | str,
    charls_splits: int = 5,
    mpdd_splits: int = 5,
    mpdd_repeats: int = 5,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    charls_result = run_charls_theory_baselines(
        charls,
        n_splits=charls_splits,
    )
    charls_result["fold_report"].to_csv(
        output / "charls_theory_fold_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    charls_result["summary"].to_csv(
        output / "charls_theory_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    charls_result["comparisons"].to_csv(
        output / "charls_theory_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    charls_result["oof_predictions"].to_csv(
        output / "charls_theory_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mpdd_result = run_mpdd_modality_ablation(
        mpdd_subject,
        n_splits=mpdd_splits,
        n_repeats=mpdd_repeats,
    )
    mpdd_result["fold_report"].to_csv(
        output / "mpdd_modality_fold_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mpdd_result["summary"].to_csv(
        output / "mpdd_modality_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mpdd_result["comparisons"].to_csv(
        output / "mpdd_modality_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit = cross_source_audit(charls, mpdd_subject, cmdc)
    (output / "cross_source_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    charls_summary = charls_result["summary"].set_index("baseline_id")
    mpdd_summary = mpdd_result["summary"].set_index("modalities")
    best_mpdd = (
        mpdd_result["summary"]
        .loc[lambda frame: frame["modalities"] != "prevalence"]
        .sort_values("auc_mean", ascending=False)
        .iloc[0]
    )
    full_mpdd = mpdd_summary.loc["face+audio+gait+personality"]
    audio_removed = mpdd_summary.loc["face+gait+personality"]
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "charls_splits": charls_splits,
        "mpdd_splits": mpdd_splits,
        "mpdd_repeats": mpdd_repeats,
        "theory_feature_sets": charls_result["feature_sets"],
        "fine_tuning_boundary": mpdd_result["fine_tuning_boundary"],
        "headline": {
            "charls_demographic_auc": float(charls_summary.loc["B1", "auc"]),
            "charls_theory_auc": float(charls_summary.loc["B2", "auc"]),
            "charls_theory_delta_auc": float(
                charls_summary.loc["B2", "auc"]
                - charls_summary.loc["B1", "auc"]
            ),
            "mpdd_full_auc": float(full_mpdd["auc_mean"]),
            "mpdd_full_pr_auc": float(full_mpdd["pr_auc_mean"]),
            "mpdd_full_brier": float(full_mpdd["brier_mean"]),
            "mpdd_best_subset": str(best_mpdd["modalities"]),
            "mpdd_best_subset_auc": float(best_mpdd["auc_mean"]),
            "mpdd_audio_removed_auc": float(audio_removed["auc_mean"]),
            "mpdd_audio_removed_delta_auc": float(
                audio_removed["auc_mean"] - full_mpdd["auc_mean"]
            ),
        },
    }
    (output / "incremental_validation_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "charls": charls_result["summary"],
        "charls_folds": charls_result["fold_report"],
        "charls_comparisons": charls_result["comparisons"],
        "mpdd": mpdd_result["summary"],
        "mpdd_folds": mpdd_result["fold_report"],
        "mpdd_comparisons": mpdd_result["comparisons"],
        "audit": audit,
        "meta": metadata,
    }
