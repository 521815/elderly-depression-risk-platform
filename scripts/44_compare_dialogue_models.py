#!/usr/bin/env python3
"""Compare scale-only, scale+dialogue, and scale+dialogue+multimodal models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.imbalance import best_threshold
from elderly_depression.research_dialogues import (
    SESSION_FILE,
    build_session_frame,
    load_sessions,
    subject_level_split,
)


def _demo_sessions(samples: int = 300) -> pd.DataFrame:
    source = PROJECT_ROOT / "data" / "synthetic" / "dialogue_samples.csv"
    df = pd.read_csv(source).head(samples)
    rng = np.random.default_rng(42)
    records = []
    for index, row in df.iterrows():
        probability = float(row.get("depression_probability") or 0)
        label = int(probability >= 0.5)
        scale_score = float(np.clip(round(probability * 20 + rng.normal(0, 2)), 0, 27))
        records.append(
            {
                "subject_id": f"demo-subject-{index:04d}",
                "session_id": f"demo-session-{index:04d}",
                "collected_at": "2026-10-21",
                "age": int(rng.integers(60, 90)),
                "gender": str(rng.choice(["female", "male"])),
                "comorbidity_count": int(rng.integers(0, 5)),
                "scale_name": "synthetic_demo",
                "scale_score": scale_score,
                "depression_label": label,
                "transcript_json": row.get("transcript_json"),
                "consent_code": f"demo-consent-{index:04d}",
                "mpdd_risk": float(np.clip(probability + rng.normal(0, 0.12), 0, 1)),
            }
        )
    return build_session_frame(records)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["gender_num"] = frame["gender"].astype(str).str.lower().map(
        {"female": 1, "女": 1, "male": 0, "男": 0}
    )
    frame["gender_num"] = frame["gender_num"].fillna(0)
    return frame


def _features(frame: pd.DataFrame) -> dict[str, list[str]]:
    scale = ["scale_score", "age", "gender_num", "comorbidity_count"]
    dialogue = [
        column
        for column in frame.columns
        if column.startswith("dlg_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    multimodal = [
        column
        for column in frame.columns
        if column == "mpdd_risk" or column.startswith("mm_")
    ]
    return {
        "scale": [column for column in scale if column in frame.columns],
        "dialogue": dialogue,
        "multimodal": multimodal,
    }


def _ece(y_true, probabilities, bins: int = 10) -> float:
    bins_edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(bins_edges[:-1], bins_edges[1:]):
        mask = (probabilities >= low) & (probabilities < high)
        if mask.any():
            error += float(mask.mean()) * abs(
                float(np.mean(y_true[mask])) - float(np.mean(probabilities[mask]))
            )
    return error


def _evaluate(frame: pd.DataFrame, columns: list[str], label_col: str) -> dict | None:
    if not columns or frame[label_col].nunique() < 2:
        return None
    split = frame["split"]
    train = frame[split == "train"]
    validation = frame[split == "validation"]
    test = frame[split == "test"]
    if train[label_col].nunique() < 2 or test.empty:
        return None
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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
    model.fit(train[columns], train[label_col].astype(int))
    val_prob = model.predict_proba(validation[columns])[:, 1]
    threshold = (
        best_threshold(validation[label_col].astype(int), val_prob, metric="youden")
        if validation[label_col].nunique() >= 2
        else 0.5
    )
    probabilities = model.predict_proba(test[columns])[:, 1]
    predicted = (probabilities >= threshold).astype(int)
    y_true = test[label_col].astype(int).to_numpy()
    specificity = (
        float(((predicted == 0) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))
    )
    return {
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_test": int(len(test)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "sensitivity": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece": float(_ece(y_true, probabilities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(SESSION_FILE))
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output"))
    args = parser.parse_args()

    if args.demo:
        frame = _demo_sessions()
        data_source = "demo_synthetic"
        clinical_evidence = False
    else:
        frame = load_sessions(args.data)
        data_source = str(args.data)
        clinical_evidence = True
    frame = subject_level_split(_prepare(frame))
    features = _features(frame)
    rows = []
    for name, columns in [
        ("scale_only", features["scale"]),
        ("scale_dialogue", features["scale"] + features["dialogue"]),
        (
            "scale_dialogue_multimodal",
            features["scale"] + features["dialogue"] + features["multimodal"],
        ),
    ]:
        metrics = _evaluate(frame, columns, "depression_label")
        if metrics:
            rows.append({"model": name, "features": len(columns), **metrics})
    if not rows:
        raise RuntimeError("No model could be evaluated; check labels and feature columns")
    report = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dialogue_three_model_report.csv"
    json_path = out_dir / "dialogue_three_model_report.json"
    report.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = {
        "data_source": data_source,
        "clinical_evidence": clinical_evidence,
        "subjects": int(frame["subject_id"].nunique()),
        "sessions": int(len(frame)),
        "report": report.to_dict(orient="records"),
        "note": (
            "Demo data is synthetic and only validates the pipeline; it is not clinical evidence."
            if not clinical_evidence
            else "This report uses imported de-identified research sessions."
        ),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report.to_string(index=False))
    print("\nSaved:", csv_path)


if __name__ == "__main__":
    main()
