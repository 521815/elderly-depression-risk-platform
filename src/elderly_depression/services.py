"""Shared service layer used by both the Flask API and the Streamlit frontend."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from sklearn.model_selection import train_test_split

from .config import CMDC_ROOT, MPDD_ROOT, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE
from .evaluation import classification_report_frame, imbalanced_metrics_frame
from .imbalance import best_threshold, resample_train, scale_pos_weight
from .models.llm_assessor import DialogueAssessor
from .models.tabular import prepare_X
from .data.mpdd import add_mpdd_binary_target, mpdd_label_audit
from .data.feature_cache import load_parquet_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHARLS_CSV = DATA_DIR / "charls_standardized.csv"
MPDD_CSV = DATA_DIR / "mpdd_features.csv"
MPDD_UPLOAD_CSV = DATA_DIR / "mpdd_features_uploaded.csv"
MPDD_UTTERANCE_CSV = DATA_DIR / "mpdd_utterance_features.csv"
CMDC_CSV = DATA_DIR / "cmdc_features.csv"
CMDC_AUDIT_JSON = OUTPUT_DIR / "cmdc_data_audit.json"
CMDC_REPORT = OUTPUT_DIR / "cmdc_benchmark_report.csv"
CMDC_MODEL_SUMMARY = OUTPUT_DIR / "cmdc_benchmark_summary.json"
CMDC_MODEL_DIR = OUTPUT_DIR / "cmdc_models"
CMDC_PAIRED_REPORT = OUTPUT_DIR / "cmdc_paired_comparison.csv"
CMDC_MISSING_MODALITY_REPORT = OUTPUT_DIR / "cmdc_missing_modality_report.csv"
CMDC_MODEL_COMPARISON_REPORT = OUTPUT_DIR / "cmdc_model_comparison.csv"
CMDC_CALIBRATION_CURVE = OUTPUT_DIR / "cmdc_calibration_curve.csv"
CMDC_DECISION_CURVE = OUTPUT_DIR / "cmdc_decision_curve.csv"
CMDC_ADVANCED_SUMMARY = OUTPUT_DIR / "cmdc_advanced_summary.json"
CMDC_NESTED_MODEL = OUTPUT_DIR / "cmdc_nested_best_model.pkl"
CMDC_HOLDOUT_SUMMARY = OUTPUT_DIR / "cmdc_locked_holdout_summary.json"
CMDC_HOLDOUT_PREDICTIONS = OUTPUT_DIR / "cmdc_locked_holdout_predictions.csv"
CMDC_HOLDOUT_MODEL = OUTPUT_DIR / "cmdc_locked_holdout_model.pkl"
MODMA_ROOT = DATA_DIR / "modma"
MODMA_METADATA = (
    MODMA_ROOT / "metadata" / "subjects_information_audio_lanzhou_2015.xlsx"
)
MODMA_RUN_DIR = OUTPUT_DIR / "modma"
MODMA_AUDIT_JSON = MODMA_RUN_DIR / "modma_audio_audit.json"
MODMA_SUBJECT_INDEX = MODMA_RUN_DIR / "modma_subject_index.csv"
MODMA_AUDIO_SUMMARY = MODMA_RUN_DIR / "modma_audio_summary.csv"
MODMA_AUDIO_FOLD_REPORT = MODMA_RUN_DIR / "modma_audio_fold_report.csv"
MODMA_AUDIO_SUBJECT_FEATURES = (
    MODMA_RUN_DIR / "modma_audio_subject_features.csv"
)
MODMA_AUDIO_TRAINING_META = MODMA_RUN_DIR / "modma_audio_training_meta.json"
MODMA_TRANSFER_DIR = OUTPUT_DIR / "modma_transfer"
MODMA_TRANSFER_SUMMARY = (
    MODMA_TRANSFER_DIR / "modma_to_mpdd_summary.csv"
)
MODMA_TRANSFER_COMPARISONS = (
    MODMA_TRANSFER_DIR / "modma_to_mpdd_comparisons.csv"
)
MODMA_TRANSFER_META = MODMA_TRANSFER_DIR / "modma_to_mpdd_meta.json"
MODEL_REPORT = OUTPUT_DIR / "tabular_model_report.csv"
IMBALANCE_REPORT = OUTPUT_DIR / "imbalance_comparison.csv"
IMBALANCE_PAPER_REPORT = OUTPUT_DIR / "imbalance_paper_report.csv"
MULTIMODAL_REPORT = OUTPUT_DIR / "multimodal_report.csv"
MULTIMODAL_SUBJECT_REPORT = OUTPUT_DIR / "multimodal_subject_report.csv"
MULTIMODAL_SUBJECT_FOLD_REPORT = OUTPUT_DIR / "multimodal_subject_fold_report.csv"
MULTIMODAL_SUBJECT_LABEL3_REPORT = OUTPUT_DIR / "multimodal_subject_label3_report.csv"
MULTIMODAL_SUBJECT_PHQ9_REPORT = OUTPUT_DIR / "multimodal_subject_phq9_score_report.csv"
MPDD_TEST_PREDICTIONS = OUTPUT_DIR / "mpdd_test_predictions.csv"
MPDD_LABEL_COMPARISON_REPORT = OUTPUT_DIR / "mpdd_label_protocol_comparison.csv"
INCREMENTAL_VALIDATION_DIR = OUTPUT_DIR / "incremental_validation"
INCREMENTAL_CHARLS_SUMMARY = (
    INCREMENTAL_VALIDATION_DIR / "charls_theory_summary.csv"
)
INCREMENTAL_MPDD_SUMMARY = (
    INCREMENTAL_VALIDATION_DIR / "mpdd_modality_summary.csv"
)
INCREMENTAL_CHARLS_COMPARISONS = (
    INCREMENTAL_VALIDATION_DIR / "charls_theory_comparisons.csv"
)
INCREMENTAL_MPDD_COMPARISONS = (
    INCREMENTAL_VALIDATION_DIR / "mpdd_modality_comparisons.csv"
)
INCREMENTAL_CROSS_SOURCE_AUDIT = (
    INCREMENTAL_VALIDATION_DIR / "cross_source_audit.json"
)
INCREMENTAL_VALIDATION_META = (
    INCREMENTAL_VALIDATION_DIR / "incremental_validation_meta.json"
)
TABULAR_MODEL_DIR = OUTPUT_DIR / "tabular_models"
MULTIMODAL_MODEL_DIR = OUTPUT_DIR / "multimodal_models"
FUSION_DIR = OUTPUT_DIR / "fusion_models"
FUSION_REPORT = FUSION_DIR / "fusion_report.csv"
FUSION_CALIBRATION_CURVE = FUSION_DIR / "calibration_curve.csv"
FUSION_WEIGHTS_JSON = FUSION_DIR / "fusion_weights.json"
CHARLS_TEMPORAL_REPORT = OUTPUT_DIR / "tabular_temporal_report.csv"
CHARLS_LONGITUDINAL_REPORT = OUTPUT_DIR / "tabular_longitudinal_report.csv"
TABULAR_SYNTHETIC_REPORT = OUTPUT_DIR / "tabular_synthetic_report.csv"
SYNTHETIC_AUGMENTATION_REPORT = OUTPUT_DIR / "synthetic_augmentation_report.csv"
MODEL_REGISTRY = OUTPUT_DIR / "model_registry.json"
DIALOGUE_INTENT_MODEL = OUTPUT_DIR / "dialogue_intent_model.pkl"
DIALOGUE_INTENT_REPORT = OUTPUT_DIR / "dialogue_intent_report.json"
DIALOGUE_RESPONSE_POLICY_MODEL = OUTPUT_DIR / "dialogue_response_policy.pkl"
DIALOGUE_RESPONSE_POLICY_REPORT = OUTPUT_DIR / "dialogue_response_policy_report.json"
ETHICS_AUDIT_REPORT = OUTPUT_DIR / "ethics_security_audit.json"
MEDICAL_KB_PATH = DATA_DIR / "medical_kb" / "medlineplus_topics.json"
_REPORTS_CACHE: dict = {"sig": None, "data": None}
_ARTIFACT_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def charls_exists() -> bool:
    return CHARLS_CSV.exists()


def mpdd_exists() -> bool:
    return MPDD_CSV.exists() or MPDD_UPLOAD_CSV.exists()


def cmdc_exists() -> bool:
    return CMDC_CSV.exists()


def modma_exists() -> bool:
    return (
        MODMA_METADATA.exists()
        and MODMA_ROOT.exists()
        and any(
            path.is_dir() and path.name.isdigit()
            for path in MODMA_ROOT.iterdir()
        )
    )


def load_charls() -> pd.DataFrame:
    return pd.read_csv(CHARLS_CSV)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    return pd.read_csv(path)


def read_json_file(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def artifact_metadata(path: Path | str) -> dict:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return {}
    stat = target.stat()
    cache_key = (str(target.resolve()), stat.st_mtime_ns, stat.st_size)
    digest = _ARTIFACT_HASH_CACHE.get(cache_key)
    if digest is None:
        hasher = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        _ARTIFACT_HASH_CACHE[cache_key] = digest
        if len(_ARTIFACT_HASH_CACHE) > 256:
            _ARTIFACT_HASH_CACHE.pop(next(iter(_ARTIFACT_HASH_CACHE)))
    return {
        "path": str(target),
        "name": target.name,
        "size_bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(
            timespec="seconds"
        ),
        "sha256": digest,
    }


def model_registry() -> list[dict]:
    entries = []

    def add_report(path: Path, model_type: str) -> None:
        df = load_csv(path)
        if df is None:
            return
        for _, row in df.iterrows():
            model = (
                row.get("model")
                or row.get("modality")
                or row.get("variant_name")
                or row.get("resample_method")
                or "unknown"
            )
            metrics = {}
            for col in ["auc", "auc_mean", "f1", "f1_mean", "brier", "accuracy"]:
                if col in df.columns and pd.notna(row.get(col)):
                    metrics[col] = round(float(row[col]), 4)
            entries.append(
                {
                    "model_type": model_type,
                    "model": str(model),
                    "source": str(row.get("source") or row.get("resample_method") or ""),
                    "metrics": metrics,
                    "report": str(path),
                    "artifact": artifact_metadata(path),
                }
            )

    add_report(MODEL_REPORT, "tabular")
    add_report(TABULAR_SYNTHETIC_REPORT, "tabular_synthetic")
    add_report(CHARLS_TEMPORAL_REPORT, "tabular_temporal")
    add_report(CHARLS_LONGITUDINAL_REPORT, "tabular_longitudinal")
    add_report(MULTIMODAL_SUBJECT_REPORT, "multimodal_subject")
    add_report(SYNTHETIC_AUGMENTATION_REPORT, "multimodal_augmentation")
    add_report(CMDC_REPORT, "cmdc_external_validation")
    add_report(CMDC_MODEL_COMPARISON_REPORT, "cmdc_model_comparison")
    return entries


def model_cards() -> list[dict]:
    cards = []

    def rounded_metrics(row: pd.Series, mapping: dict[str, str]) -> dict:
        metrics = {}
        for output_name, source_name in mapping.items():
            value = row.get(source_name)
            if value is not None and pd.notna(value):
                metrics[output_name] = round(float(value), 4)
        return metrics

    tabular = load_csv(MODEL_REPORT)
    if tabular is not None and not tabular.empty and "auc" in tabular.columns:
        auc_values = pd.to_numeric(tabular["auc"], errors="coerce")
    else:
        auc_values = pd.Series(dtype=float)
    if not auc_values.dropna().empty:
        best = tabular.loc[auc_values.idxmax()]
        cards.append(
            {
                "id": "charls_tabular",
                "name": "CHARLS 老年表格风险模型",
                "dataset": "CHARLS",
                "population": "中国老年人",
                "role": "老年主风险评估的表格基线",
                "label": "CHARLS 抑郁风险标签",
                "status": "available",
                "metrics": rounded_metrics(
                    best,
                    {
                        "auc": "auc",
                        "accuracy": "accuracy",
                        "f1": "f1",
                        "sensitivity": "recall",
                    },
                ),
                "artifact": artifact_metadata(MODEL_REPORT),
                "intended_use": "老年人群科研风险筛查与因素解释",
                "not_intended_for": "个体临床诊断、自动治疗决策",
            }
        )

    subject_report = load_csv(MULTIMODAL_SUBJECT_REPORT)
    if subject_report is not None and not subject_report.empty:
        fusion = subject_report.loc[
            subject_report.get("modality", pd.Series(dtype=str)).eq("fusion")
        ]
        row = fusion.iloc[0] if not fusion.empty else subject_report.iloc[0]
        cards.append(
            {
                "id": "mpdd_multimodal",
                "name": "MPDD 老年多模态模型",
                "dataset": "MPDD",
                "population": "中国老年慢病人群",
                "role": "语音、面部、步态和人格信号的多模态基线",
                "label": "MPDD 官方 label2 二分类标签；PHQ-9 仅作为缺失时回退",
                "status": "available" if mpdd_exists() else "artifact_only",
                "metrics": rounded_metrics(
                    row,
                    {
                        "auc": "auc_mean",
                        "pr_auc": "pr_auc_mean",
                        "f1": "f1_mean",
                        "sensitivity": "sensitivity_mean",
                        "specificity": "specificity_mean",
                    },
                ),
                "artifact": artifact_metadata(MULTIMODAL_SUBJECT_REPORT),
                "intended_use": "老年多模态科研评估与缺失模态敏感性分析",
                "not_intended_for": "替代量表、临床诊断或跨中心直接部署",
            }
        )

    cmdc_benchmark = read_json_file(CMDC_MODEL_SUMMARY) or {}
    cmdc_holdout = read_json_file(CMDC_HOLDOUT_SUMMARY) or {}
    if cmdc_benchmark:
        primary = cmdc_benchmark.get("primary_result") or {}
        cards.append(
            {
                "id": "cmdc_external",
                "name": "CMDC 中文多模态外部验证模型",
                "dataset": "CMDC",
                "population": "中文混合成人样本",
                "role": "独立外部验证，不进入老年主风险融合",
                "label": "HC / MDD 二分类",
                "status": "available",
                "metrics": {
                    key: round(float(primary[key]), 4)
                    for key in (
                        "auc_mean",
                        "pr_auc_mean",
                        "sensitivity_mean",
                        "specificity_mean",
                        "brier_mean",
                    )
                    if primary.get(key) is not None
                },
                "holdout_metrics": (
                    cmdc_holdout.get("holdout_metrics") or {}
                ),
                "dataset_fingerprint": cmdc_benchmark.get(
                    "dataset_fingerprint", ""
                ),
                "schema_version": cmdc_benchmark.get("schema_version"),
                "artifact": artifact_metadata(CMDC_MODEL_SUMMARY),
                "intended_use": "中文访谈多模态外部验证和领域适配",
                "not_intended_for": "代表中国老年人、替代老年主模型或临床部署",
            }
        )

    modma_summary_report = load_csv(MODMA_AUDIO_SUMMARY)
    if modma_summary_report is not None and not modma_summary_report.empty:
        row = modma_summary_report.iloc[0]
        transfer_meta = read_json_file(MODMA_TRANSFER_META) or {}
        cards.append(
            {
                "id": "modma_audio_auxiliary",
                "name": "MODMA 中文跨年龄语音辅助模型",
                "dataset": "MODMA",
                "population": "18 至 55 岁中文临床 MDD 与健康对照",
                "role": "跨年龄段语音预训练与目标域迁移研究，不进入老年主评分",
                "label": "精神科 MINI/DSM-IV 临床二分类",
                "status": "available",
                "metrics": rounded_metrics(
                    row,
                    {
                        "auc": "auc_mean",
                        "pr_auc": "pr_auc_mean",
                        "f1": "f1_mean",
                        "brier": "brier_mean",
                        "ece": "ece_mean",
                    },
                ),
                "transfer_to_mpdd": {
                    "recommended_for_main_model": bool(
                        transfer_meta.get("recommended_for_main_model", False)
                    ),
                    "best_transfer_auc_delta": transfer_meta.get(
                        "best_transfer_auc_delta"
                    ),
                },
                "artifact": artifact_metadata(MODMA_AUDIO_SUMMARY),
                "intended_use": "跨年龄段语音可行性、预训练与域迁移审计",
                "not_intended_for": "老年外部验证、临床诊断或进入当前主风险评分",
            }
        )

    fusion_meta = read_json_file(FUSION_WEIGHTS_JSON) or {}
    if fusion_meta:
        cards.append(
            {
                "id": "assessment_fusion",
                "name": "风险评估融合层",
                "dataset": "CHARLS + MPDD + 对话 + 量表",
                "population": "老年科研风险筛查",
                "role": "将文本、量表、背景、表情和可选多模态信号融合为总风险",
                "label": "综合抑郁风险概率",
                "status": "available",
                "metrics": {},
                "artifact": artifact_metadata(FUSION_WEIGHTS_JSON),
                "intended_use": "科研风险筛查、解释和人工复核",
                "not_intended_for": "自动诊断、自动处方或无人监督决策",
            }
        )

    if SYNTHETIC_AUGMENTATION_REPORT.exists():
        cards.append(
            {
                "id": "synthetic_training_only",
                "name": "模拟数据增强组",
                "dataset": "由真实数据元素重组的模拟数据",
                "population": "仅用于训练侧增强",
                "role": "补充训练样本，不作为验证或测试",
                "label": "继承真实数据标签结构",
                "status": "training_only",
                "metrics": {},
                "artifact": artifact_metadata(SYNTHETIC_AUGMENTATION_REPORT),
                "intended_use": "训练侧增强与鲁棒性实验",
                "not_intended_for": "验证集、测试集、外部验证或临床性能声明",
            }
        )
    return cards


def code_signature() -> str:
    """Return a timestamp signature of the newest backend source file."""
    roots = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[2] / "backend_api",
        PROJECT_ROOT / "prompts",
    ]
    newest = 0.0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            newest = max(newest, path.stat().st_mtime)
    if newest <= 0:
        return "unknown"
    return datetime.fromtimestamp(newest).strftime("%Y%m%d%H%M%S")


def data_signature() -> str:
    """Return a timestamp signature of key data/model artifacts."""
    paths = [
        CHARLS_CSV,
        MPDD_CSV,
        MPDD_UPLOAD_CSV,
        MPDD_UTTERANCE_CSV,
        CMDC_CSV,
        CMDC_AUDIT_JSON,
        CMDC_REPORT,
        CMDC_MODEL_SUMMARY,
        CMDC_PAIRED_REPORT,
        CMDC_MISSING_MODALITY_REPORT,
        CMDC_MODEL_COMPARISON_REPORT,
        CMDC_CALIBRATION_CURVE,
        CMDC_DECISION_CURVE,
        CMDC_ADVANCED_SUMMARY,
        CMDC_NESTED_MODEL,
        CMDC_HOLDOUT_SUMMARY,
        CMDC_HOLDOUT_PREDICTIONS,
        CMDC_HOLDOUT_MODEL,
        MODMA_METADATA,
        MODMA_AUDIT_JSON,
        MODMA_SUBJECT_INDEX,
        MODMA_AUDIO_SUMMARY,
        MODMA_AUDIO_FOLD_REPORT,
        MODMA_AUDIO_SUBJECT_FEATURES,
        MODMA_AUDIO_TRAINING_META,
        MODMA_TRANSFER_SUMMARY,
        MODMA_TRANSFER_COMPARISONS,
        MODMA_TRANSFER_META,
        MODEL_REPORT,
        IMBALANCE_REPORT,
        MULTIMODAL_REPORT,
        MULTIMODAL_SUBJECT_REPORT,
        OUTPUT_DIR / "mpdd_label_audit.json",
        MPDD_LABEL_COMPARISON_REPORT,
        MPDD_TEST_PREDICTIONS,
        INCREMENTAL_CHARLS_SUMMARY,
        INCREMENTAL_MPDD_SUMMARY,
        INCREMENTAL_CHARLS_COMPARISONS,
        INCREMENTAL_MPDD_COMPARISONS,
        INCREMENTAL_CROSS_SOURCE_AUDIT,
        INCREMENTAL_VALIDATION_META,
        MODMA_AUDIT_JSON,
        MODMA_AUDIO_SUMMARY,
        MODMA_TRANSFER_SUMMARY,
        MODMA_TRANSFER_COMPARISONS,
        MODMA_TRANSFER_META,
        DIALOGUE_INTENT_MODEL,
        DIALOGUE_INTENT_REPORT,
        DIALOGUE_RESPONSE_POLICY_MODEL,
        DIALOGUE_RESPONSE_POLICY_REPORT,
        ETHICS_AUDIT_REPORT,
        MEDICAL_KB_PATH,
        OUTPUT_DIR / "dialogue_intent_optimization_report.json",
        OUTPUT_DIR / "dialogue_self_evaluation.json",
        OUTPUT_DIR / "dialogue_three_model_report.csv",
        OUTPUT_DIR / "dialogue_three_model_report.json",
        OUTPUT_DIR / "charls_subgroup_shrunk.csv",
        OUTPUT_DIR / "robust_aggregation_report.csv",
        SYNTHETIC_AUGMENTATION_REPORT,
        FUSION_REPORT,
        FUSION_WEIGHTS_JSON,
        CHARLS_TEMPORAL_REPORT,
        CHARLS_LONGITUDINAL_REPORT,
        *(
            sorted(TABULAR_MODEL_DIR.glob("*.pkl"))
            if TABULAR_MODEL_DIR.exists()
            else []
        ),
        *(
            sorted(MULTIMODAL_MODEL_DIR.glob("*.pkl"))
            if MULTIMODAL_MODEL_DIR.exists()
            else []
        ),
        *(
            sorted(CMDC_MODEL_DIR.glob("*.pkl"))
            if CMDC_MODEL_DIR.exists()
            else []
        ),
    ]
    newest = 0.0
    for path in paths:
        try:
            if path.exists():
                newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    if newest <= 0:
        return "unknown"
    return datetime.fromtimestamp(newest).strftime("%Y%m%d%H%M%S")


def build_charls_dataset() -> pd.DataFrame:
    from .data.charls import build_standardized

    return build_standardized(output_path=CHARLS_CSV)


def build_mpdd_dataset() -> pd.DataFrame:
    from .data.mpdd import build_feature_table

    return build_feature_table(mpdd_root=MPDD_ROOT, output_path=MPDD_CSV)


def build_mpdd_utterance_dataset() -> pd.DataFrame:
    from .data.mpdd import build_utterance_feature_table

    return build_utterance_feature_table(mpdd_root=MPDD_ROOT, output_path=MPDD_UTTERANCE_CSV)


def build_cmdc_dataset(password: str | None = None) -> dict:
    from .data.cmdc import build_cmdc_feature_table

    build_cmdc_feature_table(
        cmdc_root=CMDC_ROOT,
        output_path=CMDC_CSV,
        audit_path=CMDC_AUDIT_JSON,
        password=password,
    )
    return cmdc_summary()


def train_cmdc_external_models(repeats: int = 10) -> dict:
    from .models.cmdc import train_cmdc_models

    if not CMDC_CSV.exists():
        raise FileNotFoundError("CMDC 特征表尚未构建")
    df = pd.read_csv(CMDC_CSV)
    return train_cmdc_models(
        df=df,
        report_path=CMDC_REPORT,
        summary_path=CMDC_MODEL_SUMMARY,
        model_dir=CMDC_MODEL_DIR,
        repeats=int(repeats),
    )


def run_cmdc_advanced_analysis(
    repeats: int = 10,
    model_repeats: int = 5,
    bootstrap_iterations: int = 1000,
) -> dict:
    from .models.cmdc_advanced import run_cmdc_advanced_analysis as _run

    if not CMDC_CSV.exists():
        raise FileNotFoundError("CMDC 特征表尚未构建")
    df = pd.read_csv(CMDC_CSV)
    return _run(
        df,
        OUTPUT_DIR,
        repeats=int(repeats),
        model_repeats=int(model_repeats),
        bootstrap_iterations=int(bootstrap_iterations),
    )


def run_cmdc_locked_holdout(
    repeats: int = 10,
    bootstrap_iterations: int = 1000,
) -> dict:
    from .models.cmdc_holdout import run_cmdc_locked_holdout as _run

    if not CMDC_CSV.exists():
        raise FileNotFoundError("CMDC 特征表尚未构建")
    df = pd.read_csv(CMDC_CSV)
    return _run(
        df,
        OUTPUT_DIR,
        repeats=int(repeats),
        bootstrap_iterations=int(bootstrap_iterations),
    )


def build_mpdd_feature_cache(force: bool = False) -> dict:
    from .data.feature_cache import build_feature_cache

    return build_feature_cache(force=force)


def mpdd_cache_status() -> dict:
    from .data.feature_cache import cache_status

    return cache_status()


def generate_synthetic(
    dialogue_n: int = 80,
    tabular_n: int = 400,
    multimodal_n: int = 300,
    seed: int = 42,
) -> dict:
    from .synthetic import generate_all

    return generate_all(
        dialogue_n=dialogue_n,
        tabular_n=tabular_n,
        multimodal_n=multimodal_n,
        seed=seed,
    )


def synthetic_status() -> dict:
    from .synthetic import synthetic_status

    return synthetic_status()


def delete_synthetic() -> bool:
    from .synthetic import delete_synthetic

    return delete_synthetic()


def validate_dialogue_samples() -> dict:
    from .synthetic import validate_dialogue_samples

    return validate_dialogue_samples()


def generate_real_profile() -> dict:
    from .real_profile import generate_profile

    return generate_profile()


def real_profile_status() -> dict:
    from .real_profile import profile_status

    return profile_status()


def run_synthetic_v3(n_subjects: int = 120, segments_per_subject: int = 5) -> dict:
    from .multimodal_synthetic_v3 import run_v3_pipeline

    return run_v3_pipeline(
        n_subjects=n_subjects,
        segments_per_subject=segments_per_subject,
    )


def run_synthetic_v3_grid() -> dict:
    from .multimodal_synthetic_v3 import grid_search_v3

    return grid_search_v3()


def run_segment_augmentation(noise_scale: float = 0.03, k: int = 3) -> dict:
    from .multimodal_synthetic_v3 import run_augmentation_pipeline

    return run_augmentation_pipeline(noise_scale=noise_scale, k=k)


def run_augmentation_optimization() -> dict:
    from .multimodal_synthetic_v3 import run_augmentation_optimization

    return run_augmentation_optimization()


def run_repeated_holdout(n_repeats: int = 5) -> dict:
    from .multimodal_synthetic_v3 import run_repeated_holdout

    return run_repeated_holdout(n_repeats=n_repeats)


def synthetic_v3_status() -> dict:
    from .multimodal_synthetic_v3 import (
        BEST_PARAMS_JSON,
        GRID_RESULTS_CSV,
        AUGMENTED_PARQUET,
        SYNTHETIC_V3_PARQUET,
        VALIDATION_JSON,
    )

    ready = SYNTHETIC_V3_PARQUET.exists()
    validation = None
    if VALIDATION_JSON.exists():
        import json

        try:
            validation = json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    grid = load_csv(GRID_RESULTS_CSV)
    best_params = None
    if BEST_PARAMS_JSON.exists():
        try:
            best_params = json.loads(BEST_PARAMS_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    augmentation_ready = AUGMENTED_PARQUET.exists()
    augmentation_report = load_csv(OUTPUT_DIR / "synthetic_augmentation_report.csv")
    gait_grid = load_csv(OUTPUT_DIR / "gait_augmentation_grid.csv")
    tuning_grid = load_csv(OUTPUT_DIR / "augmentation_tuning_grid.csv")
    final_report = load_csv(OUTPUT_DIR / "synthetic_augmentation_final_report.csv")
    best_params_path = DATA_DIR / "synthetic" / "augmentation_best_params.json"
    best_params = None
    if best_params_path.exists():
        try:
            best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    repeated_report = load_csv(OUTPUT_DIR / "synthetic_repeated_holdout_report.csv")
    repeated_summary = load_csv(OUTPUT_DIR / "synthetic_repeated_holdout_summary.csv")
    return {
        "ready": ready,
        "validation": validation,
        "grid": grid.to_dict(orient="records") if grid is not None else None,
        "best_params": best_params,
        "augmentation_ready": augmentation_ready,
        "augmentation_report": (
            augmentation_report.to_dict(orient="records")
            if augmentation_report is not None
            else None
        ),
        "gait_augmentation_grid": (
            gait_grid.to_dict(orient="records") if gait_grid is not None else None
        ),
        "augmentation_tuning_grid": (
            tuning_grid.to_dict(orient="records") if tuning_grid is not None else None
        ),
        "augmentation_final_report": (
            final_report.to_dict(orient="records") if final_report is not None else None
        ),
        "augmentation_best_params": best_params,
        "repeated_holdout_report": (
            repeated_report.to_dict(orient="records") if repeated_report is not None else None
        ),
        "repeated_holdout_summary": (
            repeated_summary.to_dict(orient="records") if repeated_summary is not None else None
        ),
    }


def load_mpdd_utterance_features() -> pd.DataFrame:
    cached = load_parquet_features("mpdd_utterance_features")
    if cached is not None:
        return cached
    if not MPDD_UTTERANCE_CSV.exists():
        build_mpdd_utterance_dataset()
    return pd.read_csv(MPDD_UTTERANCE_CSV)


def load_mpdd_subject_features() -> pd.DataFrame:
    cached = load_parquet_features("mpdd_features")
    if cached is not None:
        return cached
    if not MPDD_CSV.exists():
        build_mpdd_dataset()
    return pd.read_csv(MPDD_CSV)


def train_mpdd_subject_baseline() -> pd.DataFrame:
    from .models.multimodal import train_subject_late_fusion
    from .smoothing import aggregate_mpdd_subjects

    raw = load_mpdd_utterance_features()
    audit = mpdd_label_audit(raw)
    if not MPDD_UTTERANCE_CSV.exists():
        build_mpdd_utterance_dataset()
    df = aggregate_mpdd_subjects(MPDD_UTTERANCE_CSV, method="mean")
    prefixes = []
    for prefix in ["face", "audio", "gait", "personality"]:
        if any(c.startswith(prefix + "_") for c in df.columns):
            prefixes.append(prefix)
    result = train_subject_late_fusion(
        df,
        prefixes=prefixes,
        label_col="_label",
        output_dir=OUTPUT_DIR,
    )
    (OUTPUT_DIR / "mpdd_label_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    MULTIMODAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_src = OUTPUT_DIR / "subject_fusion_pipeline.pkl"
    pipeline_dst = MULTIMODAL_MODEL_DIR / "subject_fusion_pipeline.pkl"
    if pipeline_src.exists():
        pipeline_dst.write_bytes(pipeline_src.read_bytes())
    return result["summary"]


def train_mpdd_subject_advanced() -> dict:
    from .models.multimodal import train_subject_target
    from .smoothing import aggregate_mpdd_subjects

    if not MPDD_UTTERANCE_CSV.exists():
        build_mpdd_utterance_dataset()
    df = aggregate_mpdd_subjects(MPDD_UTTERANCE_CSV, method="mean")
    df["mpdd_binary"] = df["_label"].astype(int)
    prefixes = []
    for prefix in ["face", "audio", "gait", "personality"]:
        if any(c.startswith(prefix + "_") for c in df.columns):
            prefixes.append(prefix)

    label3 = train_subject_target(
        df,
        prefixes=prefixes,
        target_col="label3",
        task="class",
        output_dir=OUTPUT_DIR,
        n_repeats=3,
    )
    phq9 = train_subject_target(
        df,
        prefixes=prefixes,
        target_col="phq9_score",
        task="regression",
        output_dir=OUTPUT_DIR,
        n_repeats=3,
    )
    return {"label3": label3, "phq9": phq9}


def export_mpdd_test_predictions() -> pd.DataFrame:
    import joblib

    from .models.multimodal import predict_subject_pipeline

    pipeline_path = MULTIMODAL_MODEL_DIR / "subject_fusion_pipeline.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError("请先训练被试级多模态基线，再导出测试集预测。")
    if not MPDD_UTTERANCE_CSV.exists():
        build_mpdd_utterance_dataset()
    pipeline = joblib.load(pipeline_path)
    df = load_mpdd_utterance_features()
    test = df[df["fold"] == "test"]
    pred = predict_subject_pipeline(pipeline, test)
    pred.to_csv(MPDD_TEST_PREDICTIONS, index=False, encoding="utf-8-sig")
    return pred


def train_tabular_baseline(method: str = "smote_tomek") -> pd.DataFrame:
    from .models.tabular import train_baseline

    df = load_charls()
    exclude = {"id", "wave", "cesd_score", "depression"}
    feature_cols = [c for c in df.columns if c not in exclude and c != "depression"]
    result = train_baseline(
        df,
        feature_cols=feature_cols,
        label_col="depression",
        resample_method=method,
        output_dir=OUTPUT_DIR,
    )
    _save_tabular_models(result)
    return result["report"]


def train_charls_temporal(method: str = "smote_tomek") -> pd.DataFrame:
    from .models.tabular import train_temporal_baseline

    df = load_charls()
    exclude = {"id", "wave", "cesd_score", "depression"}
    feature_cols = [c for c in df.columns if c not in exclude and c != "depression"]
    result = train_temporal_baseline(
        df,
        feature_cols=feature_cols,
        label_col="depression",
        resample_method=method,
        output_dir=OUTPUT_DIR,
    )
    return result["report"]


def train_charls_longitudinal(method: str = "smote_tomek") -> pd.DataFrame:
    from .models.tabular import train_longitudinal_baseline

    df = load_charls()
    exclude = {"id", "wave", "cesd_score", "depression"}
    feature_cols = [c for c in df.columns if c not in exclude and c != "depression"]
    result = train_longitudinal_baseline(
        df,
        feature_cols=feature_cols,
        label_col="depression",
        resample_method=method,
        output_dir=OUTPUT_DIR,
    )
    return result["report"]


def train_tabular_synthetic() -> pd.DataFrame:
    from .models.tabular import train_tabular_from_synthetic

    df = load_charls()
    exclude = {"id", "wave", "cesd_score", "depression"}
    feature_cols = [c for c in df.columns if c not in exclude and c != "depression"]
    synthetic_path = DATA_DIR / "synthetic" / "charls_synthetic.csv"
    if not synthetic_path.exists():
        raise FileNotFoundError("请先生成表格模拟组")
    synthetic_df = pd.read_csv(synthetic_path)
    result = train_tabular_from_synthetic(
        df,
        synthetic_df,
        feature_cols=feature_cols,
        label_col="depression",
        output_dir=OUTPUT_DIR,
    )
    return result["report"]


def train_fusion_model(df: pd.DataFrame, label_col: str = "depression") -> dict:
    from .models.fusion import train_fusion

    return train_fusion(df, label_col=label_col, output_dir=FUSION_DIR)


def fusion_status() -> dict:
    from .models.fusion import load_weights

    weights = load_weights(FUSION_WEIGHTS_JSON)
    if weights is None:
        return {
            "ready": False,
            "mode": "literature_prior",
            "note": "尚未训练融合模型，风险评估使用文献先验权重。",
        }
    return {
        "ready": True,
        "mode": "learned",
        "trained_at": weights.get("trained_at"),
        "calibrated": True,
        "variants": {
            name: {
                "features": v["features"],
                "coefficients": v["coefficients"],
                "threshold": v["threshold"],
            }
            for name, v in weights.get("variants", {}).items()
        },
    }


def fusion_components() -> dict:
    components = {}
    for key, path in [
        ("context", OUTPUT_DIR / "context_component_learning.json"),
        ("mpdd_text", OUTPUT_DIR / "mpdd_text_signal.json"),
        ("mpdd_fusion", OUTPUT_DIR / "fusion_component_learning.json"),
    ]:
        if path.exists():
            try:
                components[key] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    return components


def run_subgroup_shrinkage() -> dict:
    from .smoothing import run_subgroup_shrinkage as _run

    return _run()


def run_robust_aggregation_ablation(
    repeats: int = 2,
    splits: int = 5,
    pca_dims: int = 32,
) -> dict:
    from .smoothing import run_robust_aggregation_ablation as _run

    return _run(repeats=repeats, splits=splits, pca_dims=pca_dims)


def incremental_validation_reports() -> dict:
    return {
        "ready": (
            INCREMENTAL_CHARLS_SUMMARY.exists()
            and INCREMENTAL_MPDD_SUMMARY.exists()
        ),
        "charls": (
            load_csv(INCREMENTAL_CHARLS_SUMMARY).to_dict(orient="records")
            if INCREMENTAL_CHARLS_SUMMARY.exists()
            else None
        ),
        "mpdd": (
            load_csv(INCREMENTAL_MPDD_SUMMARY).to_dict(orient="records")
            if INCREMENTAL_MPDD_SUMMARY.exists()
            else None
        ),
        "charls_comparisons": (
            load_csv(INCREMENTAL_CHARLS_COMPARISONS).to_dict(orient="records")
            if INCREMENTAL_CHARLS_COMPARISONS.exists()
            else None
        ),
        "mpdd_comparisons": (
            load_csv(INCREMENTAL_MPDD_COMPARISONS).to_dict(orient="records")
            if INCREMENTAL_MPDD_COMPARISONS.exists()
            else None
        ),
        "cross_source_audit": read_json_file(INCREMENTAL_CROSS_SOURCE_AUDIT),
        "meta": read_json_file(INCREMENTAL_VALIDATION_META),
    }


def run_incremental_validation() -> dict:
    from .models.incremental_validation import (
        run_incremental_validation as _run_incremental_validation,
    )
    from .smoothing import aggregate_mpdd_subjects

    if not MPDD_UTTERANCE_CSV.exists():
        build_mpdd_utterance_dataset()
    mpdd_subject = aggregate_mpdd_subjects(MPDD_UTTERANCE_CSV, method="mean")
    cmdc = pd.read_csv(CMDC_CSV) if CMDC_CSV.exists() else None
    _run_incremental_validation(
        charls=load_charls(),
        mpdd_subject=mpdd_subject,
        cmdc=cmdc,
        output_dir=INCREMENTAL_VALIDATION_DIR,
    )
    return incremental_validation_reports()


def research_reports() -> dict:
    def read_json(path: Path):
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def read_csv(path: Path):
        df = load_csv(path)
        return df.to_dict(orient="records") if df is not None else None

    paths = [
        OUTPUT_DIR / "charls_interpretability.json",
        OUTPUT_DIR / "charls_feature_importance.csv",
        OUTPUT_DIR / "charls_calibration.json",
        OUTPUT_DIR / "charls_calibration_curve.csv",
        OUTPUT_DIR / "charls_decision_curve.csv",
        OUTPUT_DIR / "charls_bootstrap_ci.json",
        OUTPUT_DIR / "charls_subgroup_report.csv",
        OUTPUT_DIR / "charls_subgroup_shrunk.csv",
        OUTPUT_DIR / "charls_subgroup_shrunk.json",
        OUTPUT_DIR / "robust_aggregation_report.csv",
        OUTPUT_DIR / "robust_aggregation_report.json",
        MPDD_LABEL_COMPARISON_REPORT,
        INCREMENTAL_CHARLS_SUMMARY,
        INCREMENTAL_MPDD_SUMMARY,
        INCREMENTAL_CHARLS_COMPARISONS,
        INCREMENTAL_MPDD_COMPARISONS,
        INCREMENTAL_CROSS_SOURCE_AUDIT,
        INCREMENTAL_VALIDATION_META,
        DIALOGUE_INTENT_REPORT,
        OUTPUT_DIR / "dialogue_intent_optimization_report.json",
        OUTPUT_DIR / "dialogue_self_evaluation.json",
        OUTPUT_DIR / "dialogue_three_model_report.csv",
        OUTPUT_DIR / "dialogue_three_model_report.json",
        OUTPUT_DIR / "mpdd_structured_text.json",
        OUTPUT_DIR / "mpdd_calibration_dca.json",
        OUTPUT_DIR / "ab_check_history.csv",
        DATA_DIR / "fusion_training_data.csv",
    ]
    sig = tuple(
        (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in paths
    )
    if _REPORTS_CACHE["sig"] == sig and _REPORTS_CACHE["data"] is not None:
        return _REPORTS_CACHE["data"]

    data = {
        "charls_interpretability": read_json(OUTPUT_DIR / "charls_interpretability.json"),
        "charls_feature_importance": read_csv(OUTPUT_DIR / "charls_feature_importance.csv"),
        "charls_calibration": read_json(OUTPUT_DIR / "charls_calibration.json"),
        "charls_calibration_curve": read_csv(OUTPUT_DIR / "charls_calibration_curve.csv"),
        "charls_decision_curve": read_csv(OUTPUT_DIR / "charls_decision_curve.csv"),
        "charls_bootstrap_ci": read_json(OUTPUT_DIR / "charls_bootstrap_ci.json"),
        "charls_subgroup": read_csv(OUTPUT_DIR / "charls_subgroup_report.csv"),
        "charls_subgroup_shrunk": read_csv(OUTPUT_DIR / "charls_subgroup_shrunk.csv"),
        "charls_subgroup_shrunk_meta": read_json(
            OUTPUT_DIR / "charls_subgroup_shrunk.json"
        ),
        "robust_aggregation": read_csv(OUTPUT_DIR / "robust_aggregation_report.csv"),
        "robust_aggregation_meta": read_json(
            OUTPUT_DIR / "robust_aggregation_report.json"
        ),
        "mpdd_label_protocol_comparison": read_csv(
            MPDD_LABEL_COMPARISON_REPORT
        ),
        "mpdd_label_protocol_comparison_meta": read_json(
            OUTPUT_DIR / "mpdd_label_protocol_comparison.json"
        ),
        "incremental_validation_charls": read_csv(
            INCREMENTAL_CHARLS_SUMMARY
        ),
        "incremental_validation_mpdd": read_csv(
            INCREMENTAL_MPDD_SUMMARY
        ),
        "incremental_validation_charls_comparisons": read_csv(
            INCREMENTAL_CHARLS_COMPARISONS
        ),
        "incremental_validation_mpdd_comparisons": read_csv(
            INCREMENTAL_MPDD_COMPARISONS
        ),
        "incremental_validation_audit": read_json(
            INCREMENTAL_CROSS_SOURCE_AUDIT
        ),
        "incremental_validation_meta": read_json(
            INCREMENTAL_VALIDATION_META
        ),
        "modma_audit": read_json(MODMA_AUDIT_JSON),
        "modma_audio_summary": read_csv(MODMA_AUDIO_SUMMARY),
        "modma_transfer_summary": read_csv(MODMA_TRANSFER_SUMMARY),
        "modma_transfer_comparisons": read_csv(
            MODMA_TRANSFER_COMPARISONS
        ),
        "modma_transfer_meta": read_json(MODMA_TRANSFER_META),
        "dialogue_intent": read_json(DIALOGUE_INTENT_REPORT),
        "dialogue_intent_optimization": read_json(
            OUTPUT_DIR / "dialogue_intent_optimization_report.json"
        ),
        "dialogue_self_evaluation": read_json(
            OUTPUT_DIR / "dialogue_self_evaluation.json"
        ),
        "dialogue_three_model": read_csv(
            OUTPUT_DIR / "dialogue_three_model_report.csv"
        ),
        "dialogue_three_model_meta": read_json(
            OUTPUT_DIR / "dialogue_three_model_report.json"
        ),
        "mpdd_structured_text": read_json(OUTPUT_DIR / "mpdd_structured_text.json"),
        "mpdd_calibration_dca": read_json(OUTPUT_DIR / "mpdd_calibration_dca.json"),
        "ab_history": read_csv(OUTPUT_DIR / "ab_check_history.csv"),
        "fusion_training_rows": (
            int(len(pd.read_csv(DATA_DIR / "fusion_training_data.csv")))
            if (DATA_DIR / "fusion_training_data.csv").exists()
            else 0
        ),
    }
    _REPORTS_CACHE["sig"] = sig
    _REPORTS_CACHE["data"] = data
    return data


def _save_tabular_models(result: dict) -> None:
    TABULAR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, info in result["models"].items():
        joblib.dump(
            {
                "model": info["model"],
                "threshold": info["threshold"],
                "features": info["features"],
            },
            TABULAR_MODEL_DIR / f"{name}.pkl",
        )
        model = info["model"]
        if hasattr(model, "feature_importances_"):
            for feature, importance in zip(info["features"], model.feature_importances_):
                rows.append({"model": name, "feature": feature, "importance": float(importance)})
    if rows:
        pd.DataFrame(rows).to_csv(
            TABULAR_MODEL_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig"
        )


def run_imbalance_comparison() -> pd.DataFrame:
    from xgboost import XGBClassifier

    df = load_charls()
    exclude = {"id", "wave", "cesd_score", "depression"}
    feature_cols = [c for c in df.columns if c not in exclude and c != "depression"]
    X = prepare_X(df, feature_cols)
    y = df["depression"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr
    )

    reports = []
    detailed_reports = []
    for method in ["none", "smote", "smote_enn", "smote_tomek", "adasyn", "oversample", "undersample"]:
        X_res, y_res = resample_train(X_tr, y_tr, method=method, random_state=RANDOM_STATE)
        sw = scale_pos_weight(y_res)
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=sw,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        model.fit(X_res, y_res)
        va_prob = model.predict_proba(X_va)[:, 1]
        te_prob = model.predict_proba(X_te)[:, 1]
        threshold = best_threshold(y_va, va_prob, metric="f1")
        report = classification_report_frame(y_te, (te_prob >= threshold).astype(int), te_prob)
        report.insert(0, "resample_method", method)
        report["threshold"] = threshold
        reports.append(report)

        detailed = imbalanced_metrics_frame(y_te, (te_prob >= threshold).astype(int), te_prob)
        detailed.insert(0, "resample_method", method)
        detailed["threshold"] = threshold
        detailed_reports.append(detailed)

    result = pd.concat(reports, ignore_index=True)
    detailed_result = pd.concat(detailed_reports, ignore_index=True)
    IMBALANCE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(IMBALANCE_REPORT, index=False, encoding="utf-8-sig")
    detailed_result.to_csv(IMBALANCE_PAPER_REPORT, index=False, encoding="utf-8-sig")
    return result


def run_assessment(
    transcript: list[dict[str, str]],
    scale_answers: dict[str, list[float | int]],
    demographics: dict,
    expression_summary: dict,
    scale_name: str,
    mpdd_risk: float | None = None,
    language_cues: list[dict | str] | None = None,
    llm_complete=None,
) -> dict:
    assessor = DialogueAssessor(llm_complete=llm_complete)
    result = assessor.assess(
        transcript=transcript,
        scale_answers=scale_answers,
        demographics=demographics,
        expression_summary=expression_summary,
        scale_name=scale_name,
        mpdd_risk=mpdd_risk,
        language_cues=language_cues,
    )
    expected_items = {
        "cesd10": 10,
        "phq9": 9,
        "gds15": 15,
        "hamd24": 24,
    }.get(scale_name, 0)
    scale_items = (scale_answers or {}).get(scale_name, [])
    answered_items = sum(value is not None for value in scale_items)
    scale_complete = expected_items > 0 and answered_items >= expected_items
    user_messages = [
        item for item in transcript if item.get("role") == "user"
    ]
    available_signals = []
    if any(str(item.get("text") or "").strip() for item in user_messages):
        available_signals.append("dialogue")
    if scale_complete:
        available_signals.append("scale")
    if demographics:
        available_signals.append("demographics")
    if expression_summary.get("risk") is not None:
        available_signals.append("expression")
    if mpdd_risk is not None:
        available_signals.append("mpdd")
    warnings = []
    if not scale_complete:
        warnings.append(
            f"量表仅完成 {answered_items}/{expected_items} 项，当前结果仅供预览。"
        )
    if not user_messages:
        warnings.append("尚未进行对话访谈，语言线索可能不足。")
    result["data_quality"] = {
        "schema_version": "assessment-2026-11-10",
        "scale_name": scale_name,
        "scale_items_expected": int(expected_items),
        "scale_items_answered": int(answered_items),
        "scale_completion": (
            float(answered_items / expected_items) if expected_items else 0.0
        ),
        "scale_complete": bool(scale_complete),
        "available_signals": available_signals,
        "warnings": warnings,
        "evidence_scope": (
            "老年科研风险筛查；CMDC 混合成人样本未直接进入本次风险概率。"
        ),
    }
    return result


def expression_risk_from_openface(df: pd.DataFrame) -> dict:
    emotion_cols = {
        "happy": ["happy"],
        "sad": ["sad"],
        "angry": ["angry"],
        "surprise": ["surprise"],
        "fear": ["fear"],
        "neutral": ["neutral"],
    }
    summary: dict[str, float] = {"risk": 0.5}
    present = {}
    for key, names in emotion_cols.items():
        found = [c for c in df.columns if any(n in c.lower() for n in names)]
        if found:
            vals = pd.to_numeric(df[found[0]], errors="coerce")
            present[key] = float(vals.mean(skipna=True))
    if present:
        negative = present.get("sad", 0) + present.get("angry", 0) + present.get("fear", 0)
        positive = present.get("happy", 0) + present.get("neutral", 0)
        summary["risk"] = float(np.clip(0.5 + (negative - positive) / 2.0, 0.0, 1.0))
    summary["emotion_summary"] = present
    return summary


def train_multimodal(df: pd.DataFrame, threshold: int = 8) -> pd.DataFrame:
    from .models.multimodal import train_late_fusion

    df = add_mpdd_binary_target(df, threshold=threshold)
    df = df.dropna(subset=["mpdd_binary"]).copy()
    if df.empty:
        raise ValueError("MPDD 表格没有可用的 label2 或 PHQ-9 标签")
    prefixes = []
    for prefix in ["face", "audio", "gait", "personality"]:
        if any(c.startswith(prefix + "_") for c in df.columns):
            prefixes.append(prefix)
    if not prefixes:
        raise ValueError("未找到 face_/audio_/gait_ 特征列")
    result = train_late_fusion(
        df,
        prefixes=prefixes,
        label_col="mpdd_binary",
        output_dir=OUTPUT_DIR,
    )
    MULTIMODAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["fusion_model"], MULTIMODAL_MODEL_DIR / "fusion.pkl")
    return result["report"]


def status() -> dict:
    tabular_models = list(TABULAR_MODEL_DIR.glob("*.pkl")) if TABULAR_MODEL_DIR.exists() else []
    multimodal_models = list(MULTIMODAL_MODEL_DIR.glob("*.pkl")) if MULTIMODAL_MODEL_DIR.exists() else []
    cmdc_models = (
        list(CMDC_MODEL_DIR.glob("*.pkl"))
        if CMDC_MODEL_DIR.exists()
        else []
    )
    if CMDC_NESTED_MODEL.exists():
        cmdc_models.append(CMDC_NESTED_MODEL)
    if CMDC_HOLDOUT_MODEL.exists():
        cmdc_models.append(CMDC_HOLDOUT_MODEL)
    try:
        from .prompt_registry import prompt_versions

        prompts = prompt_versions()
    except Exception:
        prompts = {}
    from . import records as user_records

    try:
        user_stats = user_records.stats()
    except Exception:
        user_stats = {"users": 0, "assessments": 0, "db_size_bytes": 0}

    artifact_paths = [
        MODEL_REPORT,
        IMBALANCE_REPORT,
        FUSION_WEIGHTS_JSON,
        MULTIMODAL_SUBJECT_REPORT,
        SYNTHETIC_AUGMENTATION_REPORT,
        CHARLS_TEMPORAL_REPORT,
        CHARLS_LONGITUDINAL_REPORT,
        CMDC_CSV,
        CMDC_AUDIT_JSON,
        CMDC_REPORT,
        CMDC_MODEL_SUMMARY,
        CMDC_PAIRED_REPORT,
        CMDC_MISSING_MODALITY_REPORT,
        CMDC_MODEL_COMPARISON_REPORT,
        CMDC_CALIBRATION_CURVE,
        CMDC_DECISION_CURVE,
        CMDC_ADVANCED_SUMMARY,
        CMDC_HOLDOUT_SUMMARY,
        CMDC_HOLDOUT_PREDICTIONS,
        MODMA_METADATA,
        MODMA_AUDIT_JSON,
        MODMA_SUBJECT_INDEX,
        MODMA_AUDIO_SUMMARY,
        MODMA_AUDIO_FOLD_REPORT,
        MODMA_AUDIO_TRAINING_META,
        MODMA_TRANSFER_SUMMARY,
        MODMA_TRANSFER_COMPARISONS,
        MODMA_TRANSFER_META,
        INCREMENTAL_CHARLS_SUMMARY,
        INCREMENTAL_MPDD_SUMMARY,
        INCREMENTAL_CHARLS_COMPARISONS,
        INCREMENTAL_MPDD_COMPARISONS,
        INCREMENTAL_CROSS_SOURCE_AUDIT,
        DIALOGUE_INTENT_MODEL,
        DIALOGUE_RESPONSE_POLICY_MODEL,
        ETHICS_AUDIT_REPORT,
        MEDICAL_KB_PATH,
        *tabular_models,
        *multimodal_models,
        *cmdc_models,
    ]
    model_artifacts = []
    for path in artifact_paths:
        if not path.exists():
            continue
        stat = path.stat()
        model_artifacts.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                ),
            }
        )
    model_artifacts.sort(key=lambda x: x["modified_at"], reverse=True)

    ethics_audit_passed = False
    if ETHICS_AUDIT_REPORT.exists():
        try:
            ethics_payload = json.loads(
                ETHICS_AUDIT_REPORT.read_text(encoding="utf-8")
            )
            ethics_audit_passed = int(ethics_payload.get("failed", 1)) == 0
        except (OSError, ValueError, TypeError):
            ethics_audit_passed = False

    medical_kb_topics = 0
    if MEDICAL_KB_PATH.exists():
        try:
            medical_payload = json.loads(
                MEDICAL_KB_PATH.read_text(encoding="utf-8")
            )
            medical_kb_topics = int(medical_payload.get("topic_count", 0))
        except (OSError, ValueError, TypeError):
            medical_kb_topics = 0

    return {
        "service": "elderly-depression-backend",
        "charls_ready": charls_exists(),
        "mpdd_ready": mpdd_exists(),
        "cmdc_ready": cmdc_exists(),
        "cmdc_benchmark_ready": CMDC_REPORT.exists(),
        "cmdc_advanced_ready": CMDC_ADVANCED_SUMMARY.exists(),
        "cmdc_holdout_ready": CMDC_HOLDOUT_SUMMARY.exists(),
        "modma_ready": modma_exists(),
        "modma_audit_ready": MODMA_AUDIT_JSON.exists(),
        "modma_audio_baseline_ready": MODMA_AUDIO_SUMMARY.exists(),
        "modma_transfer_ready": MODMA_TRANSFER_META.exists(),
        "mpdd_subject_ready": MULTIMODAL_SUBJECT_REPORT.exists(),
        "tabular_report": MODEL_REPORT.exists(),
        "imbalance_report": IMBALANCE_REPORT.exists(),
        "charls_temporal_report": CHARLS_TEMPORAL_REPORT.exists(),
        "charls_longitudinal_report": CHARLS_LONGITUDINAL_REPORT.exists(),
        "fusion_ready": FUSION_WEIGHTS_JSON.exists(),
        "multimodal_report": MULTIMODAL_REPORT.exists(),
        "multimodal_subject_report": MULTIMODAL_SUBJECT_REPORT.exists(),
        "incremental_validation_ready": (
            INCREMENTAL_CHARLS_SUMMARY.exists()
            and INCREMENTAL_MPDD_SUMMARY.exists()
        ),
        "dialogue_intent_model": DIALOGUE_INTENT_MODEL.exists(),
        "dialogue_intent_report": DIALOGUE_INTENT_REPORT.exists(),
        "dialogue_response_policy_model": DIALOGUE_RESPONSE_POLICY_MODEL.exists(),
        "dialogue_response_policy_report": DIALOGUE_RESPONSE_POLICY_REPORT.exists(),
        "prompt_versions": prompts,
        "ethics_audit_passed": ethics_audit_passed,
        "medical_kb_ready": medical_kb_topics > 0,
        "medical_kb_topics": medical_kb_topics,
        "tabular_models": len(tabular_models),
        "multimodal_models": len(multimodal_models),
        "cmdc_models": len(cmdc_models),
        "users_count": user_stats["users"],
        "assessments_count": user_stats["assessments"],
        "db_size_bytes": user_stats["db_size_bytes"],
        "model_artifacts": model_artifacts[:20],
    }


def charls_summary() -> dict:
    if not charls_exists():
        return {"ready": False}
    df = load_charls()
    chronic_cols = [c for c in df.columns if c.startswith("chronic_")]
    return {
        "ready": True,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "positive_rate": float(df["depression"].mean()),
        "mean_age": float(df["age"].mean()),
        "mean_comorbidity": float(df["comorbidity_count"].mean()),
        "chronic_prevalence": {c: float(df[c].mean()) for c in chronic_cols},
    }


def mpdd_summary() -> dict:
    path = MPDD_UPLOAD_CSV if MPDD_UPLOAD_CSV.exists() else MPDD_CSV
    if not path.exists():
        return {"ready": False}
    df = pd.read_csv(path)
    labeled = add_mpdd_binary_target(df)
    audit = mpdd_label_audit(labeled)
    subjects = (
        labeled.drop_duplicates("subject_id")
        if "subject_id" in labeled.columns
        else labeled
    )
    target = pd.to_numeric(subjects["mpdd_binary"], errors="coerce")
    return {
        "ready": True,
        "path": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "has_label": bool(target.notna().any()),
        "positive_rate": float(target.mean()) if target.notna().any() else None,
        "label_audit": audit,
        "modalities": [
            p
            for p in ["face", "audio", "gait", "personality"]
            if any(c.startswith(p + "_") for c in df.columns)
        ],
    }


def cmdc_summary() -> dict:
    if not CMDC_CSV.exists():
        return {
            "ready": False,
            "source": "CMDC",
            "age_scope": "mixed_adults_not_older_specific",
            "paper_doi": "10.1109/TAFFC.2022.3181210",
            "dataset_doi": "10.21227/jng1-rn06",
            "usage_boundary": "仅用于非商业学术研究、外部验证和预训练。",
        }
    df = pd.read_csv(CMDC_CSV)
    labels = pd.to_numeric(df.get("label"), errors="coerce")
    metadata = {
        "source",
        "subject_id",
        "label",
        "group",
        "age_scope",
        "complete_multimodal",
        *[f"{name}_questions" for name in ("text", "visual", "audio_deep", "acoustic")],
    }
    feature_columns = [column for column in df.columns if column not in metadata]
    modalities = {}
    for name, label in [
        ("text", "文本"),
        ("visual", "面部视觉"),
        ("audio_deep", "语音嵌入"),
        ("acoustic", "openSMILE 声学"),
    ]:
        columns = [
            column
            for column in df.columns
            if column.startswith(f"{name}_")
            and not column.endswith("_questions")
        ]
        coverage = (
            float(df[columns].notna().any(axis=1).mean())
            if columns
            else 0.0
        )
        modalities[name] = {
            "label": label,
            "columns": len(columns),
            "subjects": int(
                df[f"{name}_questions"].fillna(0).gt(0).sum()
            )
            if f"{name}_questions" in df.columns
            else 0,
            "coverage": coverage,
        }
    audit = read_json_file(CMDC_AUDIT_JSON) or {}
    benchmark = read_json_file(CMDC_MODEL_SUMMARY) or {}
    advanced = read_json_file(CMDC_ADVANCED_SUMMARY) or {}
    holdout = read_json_file(CMDC_HOLDOUT_SUMMARY) or {}
    return {
        "ready": True,
        "source": "CMDC",
        "path": str(CMDC_CSV),
        "rows": int(len(df)),
        "feature_columns": len(feature_columns),
        "hc_subjects": int((labels == 0).sum()),
        "mdd_subjects": int((labels == 1).sum()),
        "positive_rate": float((labels == 1).mean()),
        "complete_multimodal_subjects": int(
            df.get("complete_multimodal", pd.Series(dtype=int))
            .fillna(0)
            .astype(int)
            .sum()
        ),
        "modalities": modalities,
        "archive_count": int(audit.get("archive_count", 0)),
        "audio_hours": float(audit.get("audio_hours", 0.0)),
        "questions_per_subject": int(audit.get("questions_per_subject", 12)),
        "age_scope": "mixed_adults_not_older_specific",
        "age_scope_warning": "CMDC 为混合成人样本，不是老年专属队列。",
        "paper_doi": "10.1109/TAFFC.2022.3181210",
        "dataset_doi": "10.21227/jng1-rn06",
        "license_boundary": (
            "遵守 CMDC EULA；仅限非商业学术研究，禁止再分发、公开原始文件或转交密码。"
        ),
        "raw_data_copied": bool(audit.get("raw_data_copied", False)),
        "password_persisted": bool(audit.get("password_persisted", False)),
        "benchmark_ready": CMDC_REPORT.exists() and CMDC_MODEL_SUMMARY.exists(),
        "advanced_ready": CMDC_ADVANCED_SUMMARY.exists(),
        "holdout_ready": CMDC_HOLDOUT_SUMMARY.exists(),
        "benchmark": benchmark,
        "advanced": advanced,
        "holdout": holdout,
    }


def modma_summary() -> dict:
    audit = read_json_file(MODMA_AUDIT_JSON) or {}
    baseline_report = load_csv(MODMA_AUDIO_SUMMARY)
    baseline = (
        baseline_report.iloc[0].to_dict()
        if baseline_report is not None and not baseline_report.empty
        else {}
    )
    training_meta = read_json_file(MODMA_AUDIO_TRAINING_META) or {}
    transfer_meta = read_json_file(MODMA_TRANSFER_META) or {}
    transfer_summary = load_csv(MODMA_TRANSFER_SUMMARY)
    return {
        "ready": modma_exists(),
        "source": "MODMA",
        "root": str(MODMA_ROOT),
        "metadata_path": str(MODMA_METADATA),
        "role": "auxiliary_cross_age_pretraining_and_robustness",
        "age_scope": "adults_18_55_not_older_specific",
        "primary_use": "跨年龄段中文临床语音预训练和 MPDD 目标域迁移",
        "not_intended_for": "老年外部验证、临床诊断或直接进入当前老年主评分",
        "audit_ready": bool(audit),
        "baseline_ready": bool(baseline),
        "transfer_ready": bool(transfer_meta),
        "audit": audit,
        "baseline": baseline,
        "training_meta": training_meta,
        "transfer_meta": transfer_meta,
        "transfer_summary": (
            transfer_summary.to_dict(orient="records")
            if transfer_summary is not None
            else []
        ),
    }


def run_modma_audit() -> dict:
    from .data.modma import build_modma_subject_index, write_modma_audit

    MODMA_RUN_DIR.mkdir(parents=True, exist_ok=True)
    index, audit = build_modma_subject_index(MODMA_ROOT, MODMA_METADATA)
    index.to_csv(MODMA_SUBJECT_INDEX, index=False)
    write_modma_audit(audit, MODMA_AUDIT_JSON)
    return modma_summary()


def run_modma_audio_baseline() -> dict:
    from .features.modma_audio import (
        aggregate_modma_subject_features,
        build_modma_audio_features,
        train_modma_subject_baseline,
    )

    if not MODMA_SUBJECT_INDEX.exists():
        run_modma_audit()
    subject_index = pd.read_csv(
        MODMA_SUBJECT_INDEX,
        dtype={"subject_id": str, "source_subject_id": str},
    )
    utterance, errors = build_modma_audio_features(MODMA_ROOT, subject_index)
    subject_features = aggregate_modma_subject_features(utterance)
    fold_report, summary = train_modma_subject_baseline(
        subject_features,
        n_splits=5,
        n_repeats=5,
    )
    utterance.drop(columns=["wav_path"], errors="ignore").to_csv(
        MODMA_RUN_DIR / "modma_audio_utterance_features.csv",
        index=False,
    )
    subject_features.to_csv(MODMA_AUDIO_SUBJECT_FEATURES, index=False)
    fold_report.to_csv(MODMA_AUDIO_FOLD_REPORT, index=False)
    summary.to_csv(MODMA_AUDIO_SUMMARY, index=False)
    (MODMA_RUN_DIR / "modma_audio_extraction_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit = read_json_file(MODMA_AUDIT_JSON) or {}
    MODMA_AUDIO_TRAINING_META.write_text(
        json.dumps(
            {
                "source": "MODMA",
                "role": "auxiliary_cross_age_pretraining_and_robustness",
                "is_older_adult_primary_dataset": False,
                "subjects": int(len(subject_features)),
                "readable_segments": int(len(utterance)),
                "feature_extractor": "numpy_scipy_mfcc_spectral_v1",
                "model": "standard_scaler_l2_logistic_C0.1",
                "threshold": 0.5,
                "splits": 5,
                "repeats": 5,
                "grouping": "subject_id",
                "audio_audit_complete": bool(
                    (audit.get("audio") or {}).get("complete_dataset", False)
                ),
                "warnings": audit.get("warnings") or [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return modma_summary()


def run_modma_transfer() -> dict:
    from .models.modma_transfer import (
        build_mpdd_raw_audio_features,
        run_modma_to_mpdd_transfer,
    )

    if not MODMA_AUDIO_SUBJECT_FEATURES.exists():
        run_modma_audio_baseline()
    source = pd.read_csv(
        MODMA_AUDIO_SUBJECT_FEATURES,
        dtype={"subject_id": str},
    )
    mpdd_audio_root = (
        MPDD_ROOT
        / "privacy-constrained-raw-Elder-train"
        / "audio"
    )
    mpdd_labels = (
        MPDD_ROOT
        / "Train-MPDD-Elder"
        / "Elder"
        / "split_labels_train.csv"
    )
    utterance, target, errors = build_mpdd_raw_audio_features(
        mpdd_audio_root,
        mpdd_labels,
    )
    fold_report, comparisons, meta = run_modma_to_mpdd_transfer(
        source,
        target,
        n_splits=5,
        n_repeats=5,
    )
    MODMA_TRANSFER_DIR.mkdir(parents=True, exist_ok=True)
    utterance.drop(columns=["wav_path"], errors="ignore").to_csv(
        MODMA_TRANSFER_DIR / "mpdd_audio_utterance_features.csv",
        index=False,
    )
    target.to_csv(
        MODMA_TRANSFER_DIR / "mpdd_audio_subject_features.csv",
        index=False,
    )
    fold_report.to_csv(
        MODMA_TRANSFER_DIR / "modma_to_mpdd_fold_report.csv",
        index=False,
    )
    comparisons.to_csv(MODMA_TRANSFER_COMPARISONS, index=False)
    summary = (
        fold_report.groupby("condition")
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            brier_mean=("brier", "mean"),
            ece_mean=("ece", "mean"),
            f1_mean=("f1", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(MODMA_TRANSFER_SUMMARY, index=False)
    (MODMA_TRANSFER_DIR / "mpdd_audio_extraction_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    MODMA_TRANSFER_META.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return modma_summary()


def data_health_report() -> dict:
    """Run a lightweight quality check across all data sources."""
    issues: list[dict] = []
    warnings: list[dict] = []
    sources: dict[str, dict] = {}

    charls_ready = charls_exists()
    charls: dict = {"ready": charls_ready}
    if charls_ready:
        df = load_charls()
        missing_rate = float(df.isna().mean().mean()) if len(df) else 1.0
        pos_rate = (
            float(df["depression"].mean()) if "depression" in df.columns else None
        )
        imbalance = None
        if pos_rate is not None:
            pos = max(pos_rate, 1.0 - pos_rate)
            neg = min(pos_rate, 1.0 - pos_rate)
            imbalance = float(pos / neg) if neg > 0 else None
        charls.update(
            rows=int(len(df)),
            columns=int(len(df.columns)),
            missing_rate=missing_rate,
            positive_rate=pos_rate,
            imbalance_ratio=imbalance,
            mean_age=float(df["age"].mean()) if "age" in df.columns else None,
            mean_comorbidity=(
                float(df["comorbidity_count"].mean())
                if "comorbidity_count" in df.columns
                else None
            ),
        )
        if missing_rate > 0.05:
            warnings.append(
                {
                    "source": "CHARLS",
                    "message": f"整体缺失率 {missing_rate:.1%}，建议先补全或说明缺失机制",
                }
            )
        if pos_rate is not None and (pos_rate < 0.05 or pos_rate > 0.95):
            warnings.append(
                {
                    "source": "CHARLS",
                    "message": f"抑郁标签阳性率 {pos_rate:.1%}，类别严重失衡，训练需采样处理",
                }
            )
        if imbalance is not None and imbalance > 9:
            warnings.append(
                {
                    "source": "CHARLS",
                    "message": f"标签比例失衡约 1:{imbalance:.1f}，建议使用不平衡处理方法",
                }
            )
    else:
        issues.append({"source": "CHARLS", "message": "标准表尚未生成，无法检查质量"})
    sources["charls"] = charls

    mpdd_path = MPDD_UPLOAD_CSV if MPDD_UPLOAD_CSV.exists() else MPDD_CSV
    mpdd_ready = mpdd_path.exists()
    mpdd: dict = {"ready": mpdd_ready, "path": str(mpdd_path) if mpdd_ready else None}
    if mpdd_ready:
        df = pd.read_csv(mpdd_path)
        missing_rate = float(df.isna().mean().mean()) if len(df) else 1.0
        modalities: dict[str, dict] = {}
        for prefix in ["face", "audio", "gait", "personality"]:
            cols = [c for c in df.columns if c.startswith(prefix + "_")]
            coverage = float(df[cols].notna().mean().mean()) if cols else 0.0
            modalities[prefix] = {"columns": len(cols), "coverage": coverage}
        labeled = add_mpdd_binary_target(df)
        audit = mpdd_label_audit(labeled)
        subjects = (
            labeled.drop_duplicates("subject_id")
            if "subject_id" in labeled.columns
            else labeled
        )
        target = pd.to_numeric(subjects["mpdd_binary"], errors="coerce")
        has_label = bool(target.notna().any())
        pos_rate = float(target.mean()) if has_label else None
        fold_counts = (
            df["fold"].value_counts().to_dict() if "fold" in df.columns else {}
        )
        mpdd.update(
            rows=int(len(df)),
            columns=int(len(df.columns)),
            missing_rate=missing_rate,
            has_label=has_label,
            positive_rate=pos_rate,
            label_audit=audit,
            fold_counts={str(k): int(v) for k, v in fold_counts.items()},
            modalities=modalities,
        )
        if missing_rate > 0.1:
            warnings.append(
                {
                    "source": "MPDD",
                    "message": f"整体缺失率 {missing_rate:.1%}，会影响单模态与融合训练",
                }
            )
        if has_label and pos_rate is not None and pos_rate < 0.05:
            warnings.append(
                {
                    "source": "MPDD",
                    "message": f"抑郁标签阳性率 {pos_rate:.1%}，类别失衡明显",
                }
            )
        if audit["threshold_disagreement"]:
            warnings.append(
                {
                    "source": "MPDD",
                    "message": (
                        f"官方 label2 与 PHQ-9 阈值口径有 "
                        f"{audit['threshold_disagreement']} 名受试者不一致；"
                        "训练已优先使用官方 label2"
                    ),
                }
            )
        for mod, info in modalities.items():
            if info["columns"] == 0:
                warnings.append(
                    {
                        "source": "MPDD",
                        "message": f"{mod} 模态特征缺失，融合模型会缺少该模态",
                    }
                )
            elif info["coverage"] < 0.9:
                warnings.append(
                    {
                        "source": "MPDD",
                        "message": f"{mod} 模态覆盖率为 {info['coverage']:.1%}，建议检查特征提取完整性",
                    }
                )
    else:
        issues.append({"source": "MPDD", "message": "特征表尚未构建，无法检查质量"})
    sources["mpdd"] = mpdd

    cmdc = cmdc_summary()
    sources["cmdc"] = cmdc
    if cmdc.get("ready"):
        if int(cmdc.get("rows", 0)) < 100:
            warnings.append(
                {
                    "source": "CMDC",
                    "message": (
                        f"仅 {cmdc.get('rows', 0)} 名受试者，"
                        "只能作为外部验证或预训练数据"
                    ),
                }
            )
        warnings.append(
            {
                "source": "CMDC",
                "message": "CMDC 是混合成人样本，不是老年专属队列",
            }
        )
        warnings.append(
            {
                "source": "CMDC",
                "message": "数据不含结构化年龄、性别、共病和量表分数，不能替代 CHARLS 主队列",
            }
        )
        visual = (cmdc.get("modalities") or {}).get("visual") or {}
        if float(visual.get("coverage", 0)) < 0.9:
            warnings.append(
                {
                    "source": "CMDC",
                    "message": (
                        f"视觉嵌入仅覆盖 {float(visual.get('coverage', 0)):.1%} "
                        "受试者，全模态分析属于完整病例子集"
                    ),
                }
            )
    else:
        warnings.append(
            {
                "source": "CMDC",
                "message": "CMDC 特征表尚未构建，当前不参与任何评估",
            }
        )

    modma = modma_summary()
    sources["modma"] = {
        "ready": bool(modma.get("ready")),
        "audit_ready": bool(modma.get("audit_ready")),
        "baseline_ready": bool(modma.get("baseline_ready")),
        "transfer_ready": bool(modma.get("transfer_ready")),
        "scope": modma.get("age_scope"),
        "audit": modma.get("audit") or {},
        "baseline": modma.get("baseline") or {},
        "transfer_meta": modma.get("transfer_meta") or {},
    }
    if modma.get("ready"):
        audit = modma.get("audit") or {}
        audio = audit.get("audio") or {}
        if int(audio.get("subject_directories", 0)) < 100:
            warnings.append(
                {
                    "source": "MODMA",
                    "message": (
                        f"仅 {audio.get('subject_directories', 0)} 名受试者，"
                        "只能作为辅助预训练或迁移数据"
                    ),
                }
            )
        warnings.append(
            {
                "source": "MODMA",
                "message": "18 至 55 岁临床 MDD/HC，不是老年专属队列",
            }
        )
        if not bool(audio.get("complete_dataset", False)):
            warnings.append(
                {
                    "source": "MODMA",
                    "message": (
                        f"1,508 个 WAV 中 "
                        f"{audio.get('unreadable_wav_files', 0)} 个无法解码；"
                        "异常片段已记录，不静默替换"
                    ),
                }
            )
        transfer_meta = modma.get("transfer_meta") or {}
        if transfer_meta and not bool(
            transfer_meta.get("recommended_for_main_model", False)
        ):
            warnings.append(
                {
                    "source": "MODMA",
                    "message": "MODMA→MPDD 迁移未显示稳定 AUC 增量，不进入主模型",
                }
            )
    else:
        issues.append(
            {
                "source": "MODMA",
                "message": "语音目录或受试者元数据缺失，无法执行辅助审计",
            }
        )

    syn = synthetic_status()
    v3 = synthetic_v3_status()
    dialogue_validation = syn.get("dialogue_validation") or {}
    v3_validation = v3.get("validation") or {}
    synthetic: dict = {
        "ready": bool(syn.get("ready", False)),
        "dialogue_rows": int((syn.get("dialogue") or {}).get("rows", 0)),
        "dialogue_feasible": bool(dialogue_validation.get("feasible", False)),
        "tabular_rows": int((syn.get("tabular") or {}).get("rows", 0)),
        "multimodal_rows": int((syn.get("multimodal") or {}).get("rows", 0)),
        "v3_feasible": bool(v3_validation.get("feasible", False)),
        "augmentation_ready": bool(v3.get("augmentation_ready", False)),
    }
    if synthetic["ready"] and not synthetic["dialogue_feasible"]:
        warnings.append(
            {
                "source": "模拟数据",
                "message": "对话模拟组未通过验证，不建议作为训练参数",
            }
        )
    if synthetic["ready"] and not synthetic["v3_feasible"]:
        warnings.append(
            {
                "source": "模拟数据",
                "message": "多模态 v3 聚合式模拟未通过分布验证，仅作实验探索",
            }
        )
    if synthetic["ready"] and not synthetic["augmentation_ready"]:
        warnings.append(
            {
                "source": "模拟数据",
                "message": "片段级增强尚未生成，它是当前官方推荐的训练参数",
            }
        )
    sources["synthetic"] = synthetic

    from . import records as user_records

    user_stats = user_records.stats()
    users = user_records.list_users()
    missing_fields = {"name": 0, "age": 0, "gender": 0}
    risk_counts = {"low": 0, "medium": 0, "high": 0, "none": 0}
    assessment_count = 0
    for user in users:
        if not str(user.get("name") or "").strip():
            missing_fields["name"] += 1
        if user.get("age") is None:
            missing_fields["age"] += 1
        if not user.get("gender"):
            missing_fields["gender"] += 1
        history = user_records.list_assessments(user["id"], limit=None)
        assessment_count += len(history)
        latest = history[0] if history else None
        risk_counts[latest.get("risk_level") if latest else "none"] += 1
    users_health = {
        "users": int(user_stats["users"]),
        "assessments": assessment_count,
        "missing_fields": missing_fields,
        "risk_distribution": risk_counts,
        "db_size_bytes": int(user_stats["db_size_bytes"]),
    }
    if sum(missing_fields.values()) > 0:
        warnings.append(
            {
                "source": "用户记录",
                "message": f"{sum(missing_fields.values())} 位用户缺少姓名/年龄/性别等关键字段",
            }
        )
    sources["users"] = users_health

    return {
        "generated_at": datetime.now().isoformat(),
        "healthy": len(issues) == 0,
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "sources": sources,
    }
