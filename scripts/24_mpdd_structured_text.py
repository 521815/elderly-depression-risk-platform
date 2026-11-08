"""Extract structured background/text features from MPDD descriptions."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.imbalance import best_threshold


def parse_description(text: str) -> dict:
    text = (text or "")
    low = text.lower()
    financial_stress = int("financial stress" in low and "no financial stress" not in low)
    lives_alone = int(("lives with 0" in low) or ("lives alone" in low) or ("no family members" in low))
    m = re.search(r"neuroticism(?: score)? of (\d+)", low) or re.search(
        r"score of (\d+) for neuroticism", low
    )
    neuroticism = float(m.group(1)) if m else float("nan")
    negative_reg = int(
        any(w in low for w in ["emotional turmoil", "intense", "suppress", "avoiding", "struggle", "withdrawal"])
    )
    return {
        "financial_stress": financial_stress,
        "lives_alone": lives_alone,
        "neuroticism": neuroticism,
        "negative_regulation": negative_reg,
    }


def main() -> None:
    labels = pd.read_csv(
        PROJECT_ROOT / "data" / "mpdd" / "Train-MPDD-Elder" / "Elder" / "split_labels_train.csv"
    )
    descriptions = pd.read_csv(
        PROJECT_ROOT / "data" / "mpdd" / "Train-MPDD-Elder" / "Elder" / "personalized_descriptions.csv"
    )
    df = labels.merge(descriptions, left_on="ID", right_on="id", how="inner")
    df = df.rename(columns={"ID": "subject_id"})
    df["subject_id"] = df["subject_id"].astype(str)
    parsed = df["description"].apply(parse_description)
    parsed = pd.DataFrame(parsed.tolist(), index=df.index)
    parsed["neuroticism"] = parsed["neuroticism"].fillna(parsed["neuroticism"].median())
    df = pd.concat([df, parsed], axis=1)
    df["label"] = (df["PHQ-9"] >= 8).astype(int)

    features = ["financial_stress", "lives_alone", "neuroticism", "negative_regulation"]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = {f: [] for f in features}
    aucs["combined"] = []
    f1s = {f: [] for f in features}
    f1s["combined"] = []
    for tr_idx, te_idx in sgkf.split(df, df["label"], groups=df["subject_id"]):
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        for f in features:
            thr = best_threshold(test["label"], test[f], metric="f1")
            pred = (test[f] >= thr).astype(int)
            aucs[f].append(roc_auc_score(test["label"], test[f]))
            f1s[f].append(f1_score(test["label"], pred, zero_division=0))
        model = LogisticRegression(max_iter=2000)
        model.fit(train[features], train["label"])
        prob = model.predict_proba(test[features])[:, 1]
        thr = best_threshold(test["label"], prob, metric="f1")
        pred = (prob >= thr).astype(int)
        aucs["combined"].append(roc_auc_score(test["label"], prob))
        f1s["combined"].append(f1_score(test["label"], pred, zero_division=0))

    result = {
        feature: {
            "mean_auc": round(float(np.mean(aucs[feature])), 4),
            "mean_f1": round(float(np.mean(f1s[feature])), 4),
        }
        for feature in features
    }
    result["combined"] = {
        "mean_auc": round(float(np.mean(aucs["combined"])), 4),
        "mean_f1": round(float(np.mean(f1s["combined"])), 4),
    }
    result["rows"] = int(len(df))
    result["note"] = "从 MPDD 描述文本解析出的结构化背景特征，subject-level 5 折评估"
    out = PROJECT_ROOT / "output" / "mpdd_structured_text.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    df[["subject_id", "description", *features, "label"]].to_csv(
        PROJECT_ROOT / "data" / "mpdd_structured_background.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
