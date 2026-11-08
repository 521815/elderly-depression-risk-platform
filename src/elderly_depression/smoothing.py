"""Small-sample probability shrinkage and robust aggregation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR
from .models.multimodal import train_subject_late_fusion


def beta_binomial_shrink(
    positive: int,
    total: int,
    global_rate: float,
    prior_strength: float = 10.0,
) -> dict:
    """Shrink a small-sample rate toward the global rate with a Beta prior."""
    positive = max(0, int(positive))
    total = max(0, int(total))
    global_rate = float(np.clip(global_rate, 1e-6, 1 - 1e-6))
    prior_alpha = global_rate * prior_strength + 1.0
    prior_beta = (1.0 - global_rate) * prior_strength + 1.0
    post_alpha = positive + prior_alpha
    post_beta = total - positive + prior_beta
    mean = post_alpha / (post_alpha + post_beta)
    try:
        from scipy.stats import beta

        lower, upper = beta.ppf([0.025, 0.975], post_alpha, post_beta)
    except Exception:
        se = float(np.sqrt(mean * (1 - mean) / max(total + prior_strength, 1)))
        lower, upper = max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)
    return {
        "positive": positive,
        "total": total,
        "raw_rate": positive / total if total else global_rate,
        "shrunk_rate": float(mean),
        "ci_low": float(lower),
        "ci_high": float(upper),
    }


def select_prior_strength(
    rows: list[dict],
    global_rate: float,
    candidates: tuple[float, ...] = (1, 2, 5, 10, 20, 50),
) -> float:
    """Choose prior strength by Beta-Binomial marginal likelihood."""
    try:
        from scipy.stats import betabinom
    except Exception:
        return 10.0
    best_strength = 10.0
    best_score = -np.inf
    for strength in candidates:
        alpha = global_rate * strength + 1.0
        beta_value = (1.0 - global_rate) * strength + 1.0
        score = 0.0
        for row in rows:
            n = int(row.get("n") or 0)
            positive = int(round(float(row.get("positive_rate") or 0) * n))
            if n <= 0:
                continue
            score += float(betabinom.logpmf(positive, n, alpha, beta_value))
        if score > best_score:
            best_score = score
            best_strength = strength
    return float(best_strength)


def run_subgroup_shrinkage(
    input_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    prior_strength: float | None = None,
) -> dict:
    """Apply empirical-Bayes shrinkage to the existing subgroup report."""
    source = Path(input_path or OUTPUT_DIR / "charls_subgroup_report.csv")
    if not source.exists():
        raise FileNotFoundError(f"Subgroup report not found: {source}")
    df = pd.read_csv(source)
    if df.empty:
        raise ValueError("Subgroup report is empty")
    required = {"group", "value", "n", "positive_rate"}
    if not required.issubset(df.columns):
        raise ValueError(f"Subgroup report must contain {sorted(required)}")
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
    df["positive_rate"] = pd.to_numeric(df["positive_rate"], errors="coerce").fillna(0.0)
    total_n = int(df["n"].sum())
    total_positive = float((df["n"] * df["positive_rate"]).sum())
    global_rate = total_positive / total_n if total_n else float(df["positive_rate"].mean())
    strength = float(prior_strength or select_prior_strength(df.to_dict("records"), global_rate))

    rows = []
    for row in df.to_dict("records"):
        n = int(row["n"])
        positive = int(round(float(row["positive_rate"]) * n))
        shrunk = beta_binomial_shrink(positive, n, global_rate, strength)
        rows.append(
            {
                **row,
                **shrunk,
                "absolute_change": abs(shrunk["shrunk_rate"] - shrunk["raw_rate"]),
            }
        )
    report = pd.DataFrame(rows)
    out_dir = Path(output_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "charls_subgroup_shrunk.csv"
    json_path = out_dir / "charls_subgroup_shrunk.json"
    report.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = {
        "method": "beta_binomial_shrinkage",
        "global_rate": round(global_rate, 6),
        "prior_strength": strength,
        "rows": int(len(report)),
        "mean_absolute_change": round(float(report["absolute_change"].mean()), 6),
        "report": report.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _aggregate_subjects(
    utterance_path: Path,
    method: str,
    threshold: int = 8,
) -> pd.DataFrame:
    df = pd.read_csv(utterance_path)
    score_col = "hamd_score" if "hamd_score" in df.columns else "phq9_score"
    if score_col not in df.columns:
        raise ValueError("MPDD utterance table has no hamd_score or phq9_score")
    df = df.dropna(subset=[score_col]).copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["_label"] = (pd.to_numeric(df[score_col], errors="coerce") >= threshold).astype(int)
    feature_cols = [
        col
        for col in df.columns
        if col.startswith(("face_", "audio_", "gait_", "personality_"))
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not feature_cols:
        raise ValueError("No numeric multimodal feature columns found")

    df[feature_cols] = df[feature_cols].astype(float)
    grouped = df.groupby("subject_id", sort=False)
    values = grouped[feature_cols]
    if method == "median":
        features = values.median()
    elif method == "trimmed_mean":
        from scipy.stats import trim_mean

        features = values.agg(lambda col: trim_mean(col.dropna(), 0.1))
    else:
        features = values.mean()
    labels = grouped["_label"].first().astype(int)
    subject = features.join(labels).reset_index()
    return subject


def aggregate_mpdd_subjects(
    utterance_path: Path | str,
    method: str = "median",
    threshold: int = 8,
) -> pd.DataFrame:
    """Public wrapper for subject-level robust feature aggregation."""
    if method not in {"mean", "median", "trimmed_mean"}:
        raise ValueError("method must be mean, median, or trimmed_mean")
    return _aggregate_subjects(Path(utterance_path), method, threshold=threshold)


def run_robust_aggregation_ablation(
    utterance_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    repeats: int = 2,
    splits: int = 5,
    pca_dims: int = 32,
) -> dict:
    """Compare mean, median, and trimmed-mean aggregation on MPDD utterances."""
    source = Path(utterance_path or DATA_DIR / "mpdd_utterance_features.csv")
    if not source.exists():
        raise FileNotFoundError(f"MPDD utterance table not found: {source}")
    labels = ["mean", "median", "trimmed_mean"]
    prefixes = ["face", "audio", "gait", "personality"]
    rows = []
    for method in labels:
        subject = aggregate_mpdd_subjects(source, method)
        result = train_subject_late_fusion(
            subject,
            prefixes=prefixes,
            label_col="_label",
            subject_col="subject_id",
            output_dir=None,
            n_splits=splits,
            n_repeats=repeats,
            pca_dims=pca_dims,
        )
        summary = result["summary"].copy()
        summary.insert(0, "aggregation", method)
        rows.append(summary)
    report = pd.concat(rows, ignore_index=True)
    out_dir = Path(output_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "robust_aggregation_report.csv"
    json_path = out_dir / "robust_aggregation_report.json"
    report.to_csv(csv_path, index=False, encoding="utf-8-sig")

    best_rows = []
    for method, group in report.groupby("aggregation", sort=False):
        fusion = group[group["modality"] == "fusion"]
        if fusion.empty:
            continue
        row = fusion.iloc[0]
        best_rows.append(
            {
                "aggregation": method,
                "fusion_auc": float(row.get("auc_mean", 0)),
                "fusion_f1": float(row.get("f1_mean", 0)),
                "fusion_brier": float(row.get("brier_mean", 1)),
                "score": float(row.get("auc_mean", 0) - row.get("brier_mean", 1)),
            }
        )
    best = pd.DataFrame(best_rows).sort_values("score", ascending=False)
    recommended = best.iloc[0]["aggregation"] if len(best) else "mean"
    summary = {
        "method": "robust_aggregation_ablation",
        "repeats": repeats,
        "splits": splits,
        "recommended": recommended,
        "report": report.to_dict(orient="records"),
        "fusion_comparison": best.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def exponential_moving_average(values: list[float], alpha: float = 0.35) -> list[float]:
    """Return a simple EMA for turn-level language-risk sequences."""
    if not values:
        return []
    alpha = float(np.clip(alpha, 0.01, 1.0))
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * float(value) + (1 - alpha) * result[-1])
    return result
