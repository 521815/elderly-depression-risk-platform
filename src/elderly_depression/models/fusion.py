"""Learned fusion layer for the final depression-risk probability.

The final score can use either literature-based prior weights (see
``llm_assessor.py``) or a logistic fusion model trained on a dataset where
every row contains the signal probabilities (text/scale/context/expression
and optionally MPDD) plus a binary label.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ..config import RANDOM_STATE, TEST_SIZE
from ..evaluation import imbalanced_metrics_frame
from ..imbalance import best_threshold


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        pred = float(y_prob[mask].mean())
        obs = float(y_true[mask].mean())
        ece += (mask.sum() / len(y_true)) * abs(obs - pred)
    return ece


def _calibration_curve_rows(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    variant: str,
    stage: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "variant": variant,
                "stage": stage,
                "bin_mid": float((lo + hi) / 2),
                "observed_freq": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def _fit_variant(
    df: pd.DataFrame,
    features: list[str],
    label_col: str,
    variant_name: str,
    output_dir: Path,
) -> dict:
    y = df[label_col].astype(int)
    X = df[features].copy().fillna(0.0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    va_prob = model.predict_proba(X_va)[:, 1]
    te_prob = model.predict_proba(X_te)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(va_prob, y_va)
    cal_va = calibrator.predict(va_prob)
    cal_te = calibrator.predict(te_prob)
    threshold = best_threshold(y_va, cal_va, metric="f1")

    report = imbalanced_metrics_frame(y_te, (cal_te >= threshold).astype(int), cal_te)
    report.insert(0, "variant", variant_name)
    report["threshold"] = threshold
    report["brier_uncalibrated"] = float(
        imbalanced_metrics_frame(y_te, (te_prob >= threshold).astype(int), te_prob)["brier"].iloc[0]
    )
    report["brier_calibrated"] = report["brier"].iloc[0]
    report["ece_uncalibrated"] = expected_calibration_error(np.asarray(y_te), te_prob)
    report["ece_calibrated"] = expected_calibration_error(np.asarray(y_te), cal_te)

    payload = {
        "model": model,
        "calibrator": calibrator,
        "threshold": float(threshold),
        "features": features,
        "intercept": float(model.intercept_[0]),
        "coefficients": {f: float(c) for f, c in zip(features, model.coef_[0])},
        "report": report,
        "curve": pd.concat(
            [
                _calibration_curve_rows(
                    np.asarray(y_te), te_prob, variant_name, "raw"
                ),
                _calibration_curve_rows(
                    np.asarray(y_te), cal_te, variant_name, "calibrated"
                ),
            ],
            ignore_index=True,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_dir / f"fusion_{variant_name}.pkl")
    joblib.dump(calibrator, output_dir / f"fusion_{variant_name}_calibrator.pkl")
    return payload


def train_fusion(
    df: pd.DataFrame,
    label_col: str = "depression",
    output_dir: Path | str | None = None,
) -> dict:
    """Fit logistic fusion models with and without the MPDD signal."""
    required = {"text_risk", "scale_risk", "context_risk"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"融合训练数据缺少列: {sorted(missing)}")
    if label_col not in df.columns:
        raise ValueError(f"融合训练数据缺少标签列: {label_col}")
    if df[label_col].nunique() < 2:
        raise ValueError("融合训练标签需要同时包含 0 和 1")

    out = Path(output_dir) if output_dir is not None else None
    data = df.copy()
    if "expression_risk" not in data.columns:
        data["expression_risk"] = 0.5

    variants: dict[str, dict] = {}
    no_mpdd_features = ["text_risk", "scale_risk", "context_risk", "expression_risk"]
    variants["no_mpdd"] = _fit_variant(
        data, no_mpdd_features, label_col, "no_mpdd", out or Path(".")
    )

    if "mpdd_risk" in data.columns and data["mpdd_risk"].notna().sum() >= 20:
        mpdd_data = data.dropna(subset=["mpdd_risk"])
        with_mpdd_features = no_mpdd_features + ["mpdd_risk"]
        variants["with_mpdd"] = _fit_variant(
            mpdd_data, with_mpdd_features, label_col, "with_mpdd", out or Path(".")
        )

    reports = [v["report"] for v in variants.values()]
    report_df = pd.concat(reports, ignore_index=True)
    weights = {
        "status": "learned",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "note": "由训练数据学出的逻辑回归融合系数；无训练数据时评估器自动回退到文献先验权重。",
        "variants": {
            name: {
                "features": v["features"],
                "intercept": v["intercept"],
                "coefficients": v["coefficients"],
                "coefficients_vector": [v["coefficients"][f] for f in v["features"]],
                "threshold": v["threshold"],
            }
            for name, v in variants.items()
        },
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(out / "fusion_report.csv", index=False, encoding="utf-8-sig")
        pd.concat([v["curve"] for v in variants.values()], ignore_index=True).to_csv(
            out / "calibration_curve.csv", index=False, encoding="utf-8-sig"
        )
        (out / "fusion_weights.json").write_text(
            json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {"report": report_df, "weights": weights, "variants": variants}


_CACHE: dict = {"mtime": None, "data": None}


def load_weights(path: Path | str | None = None) -> dict | None:
    """Read cached learned fusion weights, returning None when unavailable."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        mtime = p.stat().st_mtime
        if _CACHE["mtime"] == mtime and _CACHE["data"] is not None:
            return _CACHE["data"]
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("status") != "learned":
            return None
        for name in list(data.get("variants", {})):
            calibrator_path = p.parent / f"fusion_{name}_calibrator.pkl"
            if calibrator_path.exists():
                try:
                    data["variants"][name]["calibrator"] = joblib.load(calibrator_path)
                except Exception:
                    data["variants"][name]["calibrator"] = None
        _CACHE["mtime"] = mtime
        _CACHE["data"] = data
        return data
    except (OSError, json.JSONDecodeError):
        return None


def predict_with_weights(signals: dict[str, float], weights: dict) -> float:
    """Apply a learned logistic fusion payload to a signal dict."""
    features = weights["features"]
    x = np.array([float(signals.get(f, 0.5)) for f in features])
    linear = weights["intercept"] + float(np.dot(weights["coefficients_vector"], x))
    raw = _sigmoid(linear)
    calibrator = weights.get("calibrator")
    if calibrator is not None:
        return float(calibrator.predict(np.array([[raw]]))[0])
    return raw


def prepare_variant_weights(payload: dict) -> dict:
    """Flatten a stored variant payload into predict-with-weights form."""
    return {
        "features": payload["features"],
        "intercept": payload["intercept"],
        "coefficients_vector": [
            payload["coefficients"][f] for f in payload["features"]
        ],
        "calibrator": payload.get("calibrator"),
    }
