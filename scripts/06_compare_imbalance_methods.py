"""Compare class-imbalance strategies on the CHARLS table with one model family."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.config import RANDOM_STATE, TEST_SIZE
from elderly_depression.evaluation import classification_report_frame
from elderly_depression.imbalance import best_threshold, resample_train, scale_pos_weight
from elderly_depression.models.tabular import prepare_X

EXCLUDE = {"id", "wave", "cesd_score", "depression"}
METHODS = ["none", "smote", "smote_enn", "adasyn", "oversample", "undersample"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "charls_standardized.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output"))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Missing {input_path}. Run scripts/01_build_charls_dataset.py first.")

    df = pd.read_csv(input_path)
    feature_cols = [c for c in df.columns if c not in EXCLUDE and c != "depression"]
    X = prepare_X(df, feature_cols)
    y = df["depression"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr
    )

    reports = []
    for method in METHODS:
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
        report["train_pos"] = y_tr.sum()
        report["train_neg"] = (y_tr == 0).sum()
        reports.append(report)
        print(f"{method:12s} F1={report['f1'].iloc[0]:.4f} AUC={report['auc'].iloc[0]:.4f}")

    result = pd.concat(reports, ignore_index=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "imbalance_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out / 'imbalance_comparison.csv'}")


if __name__ == "__main__":
    main()
