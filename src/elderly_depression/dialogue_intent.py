"""Lightweight dialogue-intent model for the local fast-response path."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .config import OUTPUT_DIR


MODEL_PATH = OUTPUT_DIR / "dialogue_intent_model.pkl"
REPORT_PATH = OUTPUT_DIR / "dialogue_intent_report.json"

CUE_TO_TOPIC = {
    "睡眠差": "sleep",
    "兴趣减退": "interest",
    "情绪低落": "mood",
    "躯体不适": "somatic",
    "认知下降": "cognition",
    "社交孤立": "social",
    "无望感": "negative_cognition",
    "自责/负担": "negative_cognition",
    "自杀意念": "negative_cognition",
    "sleep": "sleep",
    "interest": "interest",
    "mood": "mood",
    "somatic": "somatic",
    "cognition": "cognition",
    "social": "social",
    "negative_cognition": "negative_cognition",
    "safety": "negative_cognition",
}

_LOCK = threading.Lock()
_CACHE: dict = {"mtime": None, "bundle": None}


def train_intent_model(
    rows: list[dict],
    output_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> dict:
    df = pd.DataFrame(rows)
    if df.empty or not {"text", "cue"}.issubset(df.columns):
        raise ValueError("Training rows must contain text and cue")
    df = df.dropna(subset=["text", "cue"]).drop_duplicates()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    if len(df) < 10:
        raise ValueError("At least 10 labeled utterances are required")

    x = df["text"].tolist()
    y = df["cue"].tolist()
    groups = df["group"].astype(str).tolist() if "group" in df.columns else None
    counts = pd.Series(y).value_counts()
    validation_strategy = "random"
    if groups is not None and len(set(groups)) > 1:
        group_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(group_split.split(x, y, groups=groups))
        x_train = [x[i] for i in train_idx]
        x_test = [x[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]
        validation_strategy = "grouped_keyword_holdout"
    else:
        stratify = y if counts.min() >= 2 else None
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.2,
                random_state=42,
                stratify=stratify,
            )
        except ValueError:
            x_train, x_test, y_train, y_test = x, x, y, y

    evaluator = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=1,
                    max_features=6000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    evaluator.fit(x_train, y_train)
    predicted = evaluator.predict(x_test)
    probabilities = evaluator.predict_proba(x_test)
    max_probabilities = probabilities.max(axis=1)
    threshold_analysis = {}
    for threshold in [0.4, 0.5, 0.55, 0.6, 0.7, 0.8]:
        mask = max_probabilities >= threshold
        threshold_analysis[str(threshold)] = {
            "coverage": float(mask.mean()) if len(mask) else 0.0,
            "accuracy": (
                float(accuracy_score([y for y, keep in zip(y_test, mask) if keep], predicted[mask]))
                if mask.any()
                else 0.0
            ),
        }
    metrics = {
        "samples": int(len(df)),
        "labels": int(len(counts)),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "validation_strategy": validation_strategy,
        "threshold_analysis": threshold_analysis,
        "label_distribution": {str(k): int(v) for k, v in counts.items()},
    }

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=1,
                    max_features=6000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x, y)
    bundle = {
        "model": model,
        "labels": sorted(set(y)),
        "cue_to_topic": CUE_TO_TOPIC,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
    }
    output = Path(output_path or MODEL_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    report_file = Path(report_path or REPORT_PATH)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(
            {
                "model_path": str(output),
                **metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"model_path": str(output), **metrics}


def _new_intent_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=1,
                    max_features=8000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _fit_and_evaluate(
    x: list[str],
    y: list[str],
    groups: list[str],
) -> tuple[Pipeline, dict]:
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(split.split(x, y, groups=groups))
    x_train = [x[i] for i in train_idx]
    x_test = [x[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    evaluator = _new_intent_pipeline()
    evaluator.fit(x_train, y_train)
    predicted = evaluator.predict(x_test)
    probabilities = evaluator.predict_proba(x_test).max(axis=1)
    thresholds = {}
    for threshold in [0.4, 0.5, 0.55, 0.6, 0.7]:
        mask = probabilities >= threshold
        thresholds[str(threshold)] = {
            "coverage": float(mask.mean()) if len(mask) else 0.0,
            "accuracy": (
                float(
                    accuracy_score(
                        [label for label, keep in zip(y_test, mask) if keep],
                        predicted[mask],
                    )
                )
                if mask.any()
                else 0.0
            ),
        }
    metrics = {
        "samples": len(x),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "labels": len(set(y)),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(
            f1_score(y_test, predicted, average="macro", zero_division=0)
        ),
        "validation_strategy": "grouped_seed_holdout",
        "threshold_analysis": thresholds,
        "label_distribution": pd.Series(y).value_counts().to_dict(),
    }
    final_model = _new_intent_pipeline()
    final_model.fit(x, y)
    return final_model, metrics


def train_mode_intent_models(
    rows: list[dict],
    output_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> dict:
    """Train separate interview and companion intent classifiers."""
    df = pd.DataFrame(rows)
    if df.empty or not {"text", "cue", "group"}.issubset(df.columns):
        raise ValueError("Mode training rows must contain text, cue, and group")
    df = df.dropna(subset=["text", "cue", "group"]).drop_duplicates()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    models: dict[str, Pipeline] = {}
    metrics: dict[str, dict] = {}
    for mode in ["interview", "companion"]:
        subset = df[df["cue"].astype(str).str.startswith(f"{mode}:")]
        if subset.empty:
            continue
        labels = subset["cue"].astype(str).str.split(":").str[-1].tolist()
        x = subset["text"].astype(str).tolist()
        groups = subset["group"].astype(str).tolist()
        model, report = _fit_and_evaluate(x, labels, groups)
        models[mode] = model
        metrics[mode] = report
    if not models:
        raise ValueError("No mode-specific training rows found")

    output = Path(output_path or MODEL_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "models": models,
        "labels": {mode: sorted(set(metrics[mode]["label_distribution"])) for mode in models},
        "cue_to_topic": CUE_TO_TOPIC,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
    }
    joblib.dump(bundle, output)
    report_file = Path(report_path or REPORT_PATH)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(
            {"model_path": str(output), "modes": metrics},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"model_path": str(output), "modes": metrics}


def _load_bundle() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime_ns
    with _LOCK:
        if _CACHE["bundle"] is None or _CACHE["mtime"] != mtime:
            _CACHE["bundle"] = joblib.load(MODEL_PATH)
            _CACHE["mtime"] = mtime
        return _CACHE["bundle"]


def predict_intent(text: str, mode: str = "interview") -> dict | None:
    bundle = _load_bundle()
    if bundle is None or not str(text or "").strip():
        return None
    mode_models = bundle.get("models")
    if isinstance(mode_models, dict):
        model = mode_models.get(mode)
        if model is None:
            return None
        probabilities = model.predict_proba([str(text)])[0]
        classes = list(model.classes_)
        index = int(probabilities.argmax())
        intent = str(classes[index])
        return {
            "cue": intent,
            "intent": intent,
            "topic": bundle.get("cue_to_topic", {}).get(intent),
            "confidence": float(probabilities[index]),
            "label": intent,
            "mode": mode,
            "fastpath_allowed": bool(
                bundle.get("metrics", {}).get(mode, {}).get(
                    "use_model_fastpath", True
                )
            ),
        }
    model: Pipeline = bundle["model"]
    probabilities = model.predict_proba([str(text)])[0]
    classes = list(model.classes_)
    requested_prefix = f"{mode}:"
    candidates = [
        (index, str(label))
        for index, label in enumerate(classes)
        if str(label).startswith(requested_prefix)
    ]
    if candidates:
        index, label = max(candidates, key=lambda item: probabilities[item[0]])
        intent = label.split(":", 1)[1]
    elif mode == "interview":
        index = int(probabilities.argmax())
        label = str(classes[index])
        intent = label
    else:
        return None
    return {
        "cue": intent,
        "intent": intent,
        "topic": bundle.get("cue_to_topic", {}).get(intent),
        "confidence": float(probabilities[index]),
        "label": label,
        "mode": mode,
        "fastpath_allowed": True,
    }
