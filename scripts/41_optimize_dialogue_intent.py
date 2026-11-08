#!/usr/bin/env python3
"""Compare lightweight classifiers for interview and companion intents."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.dialogue_intent import CUE_TO_TOPIC


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        max_features=8000,
        sublinear_tf=True,
    )


def _candidates() -> dict[str, callable]:
    return {
        "logistic": lambda: LogisticRegression(
            max_iter=2500, class_weight="balanced", random_state=42
        ),
        "sgd_modified_huber": lambda: SGDClassifier(
            loss="modified_huber",
            class_weight="balanced",
            max_iter=2500,
            random_state=42,
        ),
        "complement_nb": lambda: ComplementNB(alpha=0.5),
        "knn_cosine": lambda: KNeighborsClassifier(
            n_neighbors=5,
            metric="cosine",
            weights="distance",
        ),
        "linear_svc_calibrated": lambda: CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=42),
            cv=3,
        ),
    }


def _evaluate(subset: pd.DataFrame, classifier_factory) -> dict:
    texts = subset["text"].astype(str).tolist()
    labels = subset["cue"].astype(str).str.split(":").str[-1].tolist()
    groups = subset["group"].astype(str).tolist()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for train_idx, test_idx in splitter.split(texts, labels, groups=groups):
        model = Pipeline(
            [
                ("tfidf", _vectorizer()),
                ("classifier", classifier_factory()),
            ]
        )
        model.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
        x_test = [texts[i] for i in test_idx]
        y_test = np.array([labels[i] for i in test_idx])
        predicted = model.predict(x_test)
        probabilities = model.predict_proba(x_test).max(axis=1)
        mask = probabilities >= 0.55
        rows.append(
            {
                "accuracy": accuracy_score(y_test, predicted),
                "macro_f1": f1_score(y_test, predicted, average="macro", zero_division=0),
                "coverage_055": float(mask.mean()),
                "high_conf_accuracy": (
                    float(accuracy_score(y_test[mask], predicted[mask]))
                    if mask.any()
                    else 0.0
                ),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "accuracy": float(frame["accuracy"].mean()),
        "macro_f1": float(frame["macro_f1"].mean()),
        "coverage_055": float(frame["coverage_055"].mean()),
        "high_conf_accuracy": float(frame["high_conf_accuracy"].mean()),
    }


def _score(metrics: dict) -> float:
    return (
        0.45 * metrics["macro_f1"]
        + 0.30 * metrics["accuracy"]
        + 0.15 * metrics["coverage_055"]
        + 0.10 * metrics["high_conf_accuracy"]
    )


def main() -> None:
    training_path = PROJECT_ROOT / "output" / "dialogue_intent_training.csv"
    output_path = PROJECT_ROOT / "output" / "dialogue_intent_model.pkl"
    report_path = PROJECT_ROOT / "output" / "dialogue_intent_optimization_report.json"
    if not training_path.exists():
        raise FileNotFoundError("Run scripts/40_train_dialogue_intent.py first")
    df = pd.read_csv(training_path).dropna(subset=["text", "cue", "group"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].drop_duplicates()

    models = {}
    selected = {}
    comparison = {}
    for mode in ["interview", "companion"]:
        subset = df[df["cue"].astype(str).str.startswith(f"{mode}:")].copy()
        comparison[mode] = {}
        eligible = []
        all_scores = {}
        best_name = None
        best_score = -1.0
        for name, factory in _candidates().items():
            metrics = _evaluate(subset, factory)
            comparison[mode][name] = metrics
            score = _score(metrics)
            all_scores[name] = score
            print(mode, name, {**metrics, "score": round(score, 4)})
            if metrics["high_conf_accuracy"] >= 0.90 and metrics["coverage_055"] > 0:
                eligible.append((name, score))
        use_model_fastpath = bool(eligible)
        if eligible:
            best_name, best_score = max(eligible, key=lambda item: item[1])
        else:
            best_name = max(all_scores, key=all_scores.get)
            best_score = all_scores[best_name]
        final_model = Pipeline(
            [
                ("tfidf", _vectorizer()),
                ("classifier", _candidates()[best_name]()),
            ]
        )
        final_model.fit(
            subset["text"].astype(str).tolist(),
            subset["cue"].astype(str).str.split(":").str[-1].tolist(),
        )
        models[mode] = final_model
        selected[mode] = {
            "model": best_name,
            "score": best_score,
            "use_model_fastpath": use_model_fastpath,
            **comparison[mode][best_name],
        }

    bundle = {
        "models": models,
        "labels": {
            mode: sorted(set(df[df["cue"].str.startswith(f"{mode}:")]["cue"].str.split(":").str[-1]))
            for mode in models
        },
        "cue_to_topic": CUE_TO_TOPIC,
        "metrics": selected,
        "optimization_comparison": comparison,
    }
    joblib.dump(bundle, output_path)
    report_path.write_text(
        json.dumps(
            {
                "selected": selected,
                "comparison": comparison,
                "model_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Selected:", json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
