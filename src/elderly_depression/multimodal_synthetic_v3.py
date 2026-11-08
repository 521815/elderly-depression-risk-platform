"""Hierarchical, cross-modal synthetic generation with validation gates.

Generation models subject-level latent structure first, then samples multiple
utterance segments per synthetic subject from a residual latent space. The
resulting synthetic group is validated against the real distribution before it
is allowed as a training parameter.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, OUTPUT_DIR
from .data.feature_cache import load_parquet_features
from .imbalance import best_threshold

SYNTHETIC_V3_PARQUET = DATA_DIR / "synthetic" / "mpdd_synthetic_v3.parquet"
AUGMENTED_PARQUET = DATA_DIR / "synthetic" / "mpdd_augmented.parquet"
VALIDATION_JSON = DATA_DIR / "synthetic" / "mpdd_synthetic_v3_validation.json"
GRID_RESULTS_CSV = DATA_DIR / "synthetic" / "mpdd_synthetic_v3_grid_results.csv"
BEST_PARAMS_JSON = DATA_DIR / "synthetic" / "mpdd_synthetic_v3_best_params.json"
MODALITY_PREFIXES = ["face_", "audio_", "gait_", "personality_"]


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in MODALITY_PREFIXES)]


def _subject_binary(df: pd.DataFrame) -> pd.Series:
    return df.groupby("subject_id")["hamd_binary"].first()


def _cov_psd(cov: np.ndarray, scale: float = 0.05) -> np.ndarray:
    cov = (cov + cov.T) / 2.0
    cov = cov + np.eye(cov.shape[0]) * max(1e-6, float(np.trace(cov)) / cov.shape[0] * scale)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    return (eigvecs * eigvals) @ eigvecs.T


def generate_hierarchical_synthetic(
    df: pd.DataFrame,
    n_subjects: int = 120,
    segments_per_subject: int = 5,
    seed: int = 42,
    n_latent: int = 24,
    n_residual: int = 24,
    separation_scale: float = 0.35,
    residual_scale: float = 0.5,
    residual_scales: dict[str, float] | None = None,
    calibrate_marginals: bool = False,
) -> pd.DataFrame:
    df = df.dropna(subset=["hamd_score"]).copy()
    df["hamd_binary"] = (df["hamd_score"] >= 8).astype(int)
    feat_cols = _feature_cols(df)
    df[feat_cols] = df[feat_cols].fillna(0)
    rng = np.random.default_rng(seed)

    subject_mean = df.groupby("subject_id")[feat_cols].mean()
    subject_label = _subject_binary(df).reindex(subject_mean.index)
    Xs = subject_mean[feat_cols].to_numpy(dtype="float64")
    scaler = StandardScaler()
    Xs_scaled = scaler.fit_transform(Xs)
    pca = PCA(n_components=min(n_latent, Xs_scaled.shape[1], Xs_scaled.shape[0] - 1), random_state=seed)
    S = pca.fit_transform(Xs_scaled)

    residuals = df[feat_cols].to_numpy(dtype="float64") - subject_mean.reindex(df["subject_id"]).to_numpy(dtype="float64")
    res_scaler = StandardScaler()
    residuals_scaled = res_scaler.fit_transform(residuals)
    res_pca = PCA(
        n_components=min(n_residual, residuals_scaled.shape[1], residuals_scaled.shape[0] - 1),
        random_state=seed,
    )
    R = res_pca.fit_transform(residuals_scaled)

    global_mean = S.mean(axis=0)
    global_cov = _cov_psd(np.cov(S, rowvar=False))
    class_effects = {}
    for label in sorted(subject_label.unique()):
        mask = subject_label.values == label
        class_effects[label] = S[mask].mean(axis=0) - global_mean
    res_std = R.std(axis=0)

    rows = []
    residual_scales = residual_scales or {
        "face_": 0.35,
        "audio_": 0.45,
        "gait_": 0.35,
        "personality_": 0.05,
    }
    mod_idx = {
        prefix: [i for i, c in enumerate(feat_cols) if c.startswith(prefix)]
        for prefix in MODALITY_PREFIXES
    }
    for i in range(n_subjects):
        label = int(rng.integers(0, 2))
        latent = rng.multivariate_normal(
            global_mean + class_effects[label] * separation_scale,
            global_cov,
            check_valid="ignore",
        )
        subject_vec = scaler.inverse_transform(
            pca.inverse_transform(latent.reshape(1, -1))
        )[0]
        for k in range(segments_per_subject):
            res_latent = rng.normal(
                0, res_std * residual_scale, size=res_pca.n_components_
            )
            residual = res_scaler.inverse_transform(
                res_pca.inverse_transform(res_latent.reshape(1, -1))
            )[0]
            for prefix, scale in residual_scales.items():
                idxs = mod_idx.get(prefix, [])
                if idxs:
                    residual[idxs] *= scale
            row = subject_vec + residual
            rows.append(
                {
                    "subject_id": f"synv3-{i:04d}",
                    "segment": f"synv3-seg-{i:04d}-{k}",
                    "fold": "synthetic",
                    "hamd_binary": label,
                    "hamd_score": 12.0 if label == 1 else 4.0,
                    "label2": label,
                    "label3": 2 if label == 1 else 0,
                    **dict(zip(feat_cols, row)),
                }
            )
    out = pd.DataFrame(rows, columns=["subject_id", "segment", "fold", "hamd_binary", "hamd_score", "label2", "label3"] + feat_cols)
    for col in feat_cols:
        out[col] = out[col].astype("float32")
    out = _match_covariance_per_modality(out, df)
    if calibrate_marginals:
        out = _calibrate_marginals(out, df)
    out.to_parquet(SYNTHETIC_V3_PARQUET, index=False, compression="zstd", engine="pyarrow")
    return out


def _calibrate_marginals(syn_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    """Quantile-map synthetic features to the real marginal distribution."""
    feat_cols = _feature_cols(syn_df)
    real_sample = real_df[feat_cols].sample(min(2000, len(real_df)), random_state=42)
    out = syn_df.copy()
    for col in feat_cols:
        real_vals = np.sort(real_sample[col].fillna(0).to_numpy(dtype="float64"))
        syn_vals = out[col].to_numpy(dtype="float64")
        ranks = rankdata(syn_vals, method="average") / (len(syn_vals) + 1)
        out[col] = np.interp(ranks, np.linspace(0, 1, len(real_vals)), real_vals)
    return out


def _match_covariance_per_modality(syn_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    """Whiten synthetic PCA scores and rotate them to the real score covariance."""
    out = syn_df.copy()
    for prefix in MODALITY_PREFIXES:
        cols = [c for c in real_df.columns if c.startswith(prefix)]
        if not cols:
            continue
        n_comp = min(32, len(cols), len(real_df) - 1)
        if n_comp < 2:
            continue
        pca = PCA(n_components=n_comp, random_state=42).fit(
            real_df[cols].fillna(0).to_numpy(dtype="float64")
        )
        real_scores = pca.transform(real_df[cols].fillna(0).to_numpy(dtype="float64"))
        syn_scores = pca.transform(out[cols].fillna(0).to_numpy(dtype="float64"))
        real_mean = real_scores.mean(axis=0)
        real_cov = np.cov(real_scores, rowvar=False)
        syn_cov = np.cov(syn_scores, rowvar=False)
        try:
            Lr = np.linalg.cholesky(real_cov + np.eye(real_cov.shape[0]) * 1e-8)
            Ls = np.linalg.cholesky(syn_cov + np.eye(syn_cov.shape[0]) * 1e-8)
        except np.linalg.LinAlgError:
            continue
        adj = (syn_scores - syn_scores.mean(axis=0)) @ np.linalg.inv(Ls).T @ Lr.T + real_mean
        matched = pca.inverse_transform(adj)
        out[cols] = matched.astype("float32")
    return out


def generate_segment_augmented(
    df: pd.DataFrame,
    k: int = 3,
    noise_scale: float = 0.03,
    seed: int = 42,
    modality_noise_scales: dict[str, float] | None = None,
    balance_classes: bool = False,
) -> pd.DataFrame:
    """Augment real utterance segments within subjects with small feature noise."""
    df = df.dropna(subset=["hamd_score"]).copy()
    df["hamd_binary"] = (df["hamd_score"] >= 8).astype(int)
    feat_cols = _feature_cols(df)
    if not feat_cols:
        raise ValueError("多模态输入缺少 face_/audio_/gait_/personality_ 特征列")
    rng = np.random.default_rng(seed)
    base = df.loc[df.index.repeat(k)].copy()
    std = df[feat_cols].std().fillna(0).to_numpy(dtype="float64")
    noise = rng.normal(0, noise_scale, size=(len(base), len(feat_cols))) * std
    if modality_noise_scales:
        for prefix, scale in modality_noise_scales.items():
            idxs = [i for i, c in enumerate(feat_cols) if c.startswith(prefix)]
            if idxs:
                noise[:, idxs] *= scale / noise_scale
    base[feat_cols] = base[feat_cols].to_numpy(dtype="float64") + noise
    if balance_classes:
        pos = base[base["hamd_binary"] == 1]
        neg = base[base["hamd_binary"] == 0]
        target = max(len(pos), len(neg))
        extra = []
        if len(pos) < target:
            extra.append(pos.sample(target - len(pos), replace=True, random_state=seed))
        if len(neg) < target:
            extra.append(neg.sample(target - len(neg), replace=True, random_state=seed))
        if extra:
            base = pd.concat([base] + extra, ignore_index=True)
    base["segment"] = [
        f"{seg}-aug-{i}"
        for i, seg in enumerate(base["segment"].astype(str))
    ]
    base["fold"] = "synthetic"
    for col in feat_cols:
        base[col] = base[col].astype("float32")
    return base


def _gate_ok(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(all(value.values()))
    return False


def _pc1_per_modality(df: pd.DataFrame, real_fit: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for prefix in MODALITY_PREFIXES:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        pca = PCA(n_components=1, random_state=42).fit(real_fit[cols].fillna(0).to_numpy(dtype="float64"))
        out[prefix] = pd.Series(
            pca.transform(df[cols].fillna(0).to_numpy(dtype="float64"))[:, 0],
            index=df.index,
        )
    return out


def _class_separation(df: pd.DataFrame, prefix: str) -> float | None:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols or df["hamd_binary"].nunique() != 2:
        return None
    pca = PCA(n_components=min(5, len(cols), len(df) - 1), random_state=42)
    pc = pca.fit_transform(df[cols].fillna(0).to_numpy(dtype="float64"))
    yy = df["hamd_binary"].to_numpy()
    m0, m1 = pc[yy == 0].mean(axis=0), pc[yy == 1].mean(axis=0)
    s = np.sqrt((pc[yy == 0].var(axis=0) + pc[yy == 1].var(axis=0)) / 2)
    s[s == 0] = 1
    return float(np.max(np.abs((m1 - m0) / s)))


def _within_between_ratio(df: pd.DataFrame, prefix: str) -> float | None:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return None
    groups = [g for _, g in df.groupby("subject_id") if len(g) > 1]
    if len(groups) < 2:
        return None
    within = float(np.mean([g[cols].var(axis=0).mean() for g in groups]))
    between = float(df.groupby("subject_id")[cols].mean().var(axis=0).mean())
    return within / between if between > 0 else None


def _cross_modal_corr(df: pd.DataFrame, real_fit: pd.DataFrame) -> float:
    pcs = _pc1_per_modality(df, real_fit)
    names = [p for p in MODALITY_PREFIXES if p in pcs]
    if len(names) < 2:
        return 0.0
    mat = np.column_stack([pcs[p] for p in names])
    corr = np.corrcoef(mat, rowvar=False)
    off = corr - np.eye(corr.shape[0])
    return float(np.abs(off).mean())


def validate_vs_real(syn_df: pd.DataFrame, real_df: pd.DataFrame) -> dict:
    def ensure_binary(df: pd.DataFrame) -> pd.DataFrame:
        if "hamd_binary" not in df.columns and "hamd_score" in df.columns:
            df = df.copy()
            df["hamd_binary"] = (df["hamd_score"] >= 8).astype(int)
        return df

    real_df = ensure_binary(real_df)
    syn_df = ensure_binary(syn_df)
    metrics = {}
    gates = {}
    for prefix in MODALITY_PREFIXES:
        real_d = _class_separation(real_df, prefix)
        syn_d = _class_separation(syn_df, prefix)
        real_ratio = _within_between_ratio(real_df, prefix)
        syn_ratio = _within_between_ratio(syn_df, prefix)
        if real_ratio is not None and np.isnan(real_ratio):
            real_ratio = None
        if syn_ratio is not None and np.isnan(syn_ratio):
            syn_ratio = None
        if real_ratio is not None and real_ratio < 1e-6:
            real_ratio = None
        if syn_ratio is not None and syn_ratio < 1e-6:
            syn_ratio = None
        real_pc1 = _pc1_per_modality(real_df, real_df).get(prefix)
        syn_pc1 = _pc1_per_modality(syn_df, real_df).get(prefix)
        ks_result = (
            ks_2samp(real_pc1, syn_pc1)
            if real_pc1 is not None and syn_pc1 is not None
            else None
        )
        ks_p = float(ks_result.pvalue) if ks_result is not None else None
        ks_d = float(ks_result.statistic) if ks_result is not None else None
        metrics[prefix.rstrip("_")] = {
            "real_class_separation_d": real_d,
            "synthetic_class_separation_d": syn_d,
            "real_within_between_ratio": real_ratio,
            "synthetic_within_between_ratio": syn_ratio,
            "ks_pvalue_pc1": ks_p,
            "ks_d_pc1": ks_d,
        }
        sep_ok = syn_d is None or real_d is None or (0.6 <= syn_d / real_d <= 1.4)
        ratio_ok = (
            syn_ratio is None
            or real_ratio is None
            or (0.3 <= syn_ratio / real_ratio <= 3.0)
        )
        ks_ok = ks_d is None or ks_d <= 0.15
        gates[prefix.rstrip("_")] = {"separation": bool(sep_ok), "within_between": bool(ratio_ok), "ks": bool(ks_ok)}

    real_corr = _cross_modal_corr(real_df, real_df)
    syn_corr = _cross_modal_corr(syn_df, real_df)
    corr_ok = abs(syn_corr - real_corr) <= 0.25
    metrics["cross_modal_mean_abs_corr"] = {"real": real_corr, "synthetic": syn_corr}
    gates["cross_modal_corr"] = bool(corr_ok)
    all_ok = all(_gate_ok(v) for v in gates.values())
    result = {"metrics": metrics, "gates": gates, "feasible": all_ok}
    result["note"] = (
        "通过全部分布验证门槛，可用于训练侧实验"
        if all_ok
        else "未通过分布验证门槛，不建议作为训练参数"
    )
    Path(VALIDATION_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(VALIDATION_JSON).write_text(
        __import__("json").dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _subject_probs(model, scaler, pca, df, cols):
    x = df[cols].fillna(0).to_numpy(dtype="float64")
    prob = model.predict_proba(pca.transform(scaler.transform(x)))[:, 1]
    return pd.Series(prob, index=df.index).groupby(df["subject_id"]).mean()


def _bootstrap_auc_ci(
    subject_ids: pd.Series,
    y: pd.Series,
    prob: pd.Series,
    n_boot: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject_ids.astype(str).to_numpy())
    aucs = []
    for _ in range(n_boot):
        sample = set(rng.choice(subjects, size=len(subjects), replace=True))
        mask = subject_ids.astype(str).isin(sample).to_numpy()
        yy = y.to_numpy()[mask]
        pp = prob.to_numpy()[mask]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, pp))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate_subject_aware(
    syn_df: pd.DataFrame,
    real_df: pd.DataFrame,
    n_splits: int = 5,
    use_segment_augmentation: bool = False,
    augmentation_k: int = 3,
    augmentation_noise: float = 0.03,
    augmentation_modality_scales: dict[str, float] | None = None,
) -> pd.DataFrame:
    real = real_df.dropna(subset=["hamd_score"]).copy()
    real["hamd_binary"] = (real["hamd_score"] >= 8).astype(int)
    syn = syn_df.copy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_rows = []
    for fold, (tr_idx, te_idx) in enumerate(
        sgkf.split(real, real["hamd_binary"], groups=real["subject_id"])
    ):
        real_tr = real.iloc[tr_idx]
        real_te = real.iloc[te_idx]
        for source, train_df in [("real", real_tr), ("synthetic", syn)]:
            if source == "synthetic" and use_segment_augmentation:
                train_df = generate_segment_augmented(
                    real_tr,
                    k=augmentation_k,
                    noise_scale=augmentation_noise,
                    seed=100 + fold,
                    modality_noise_scales=augmentation_modality_scales,
                )
            modality_probs_tr = {}
            modality_probs_te = {}
            for prefix in MODALITY_PREFIXES:
                cols = [c for c in real.columns if c.startswith(prefix)]
                if not cols:
                    continue
                scaler = StandardScaler()
                pca = PCA(n_components=min(48, len(cols), len(train_df) - 1), random_state=42)
                x = train_df[cols].fillna(0).to_numpy(dtype="float64")
                model = RandomForestClassifier(
                    n_estimators=150, class_weight="balanced", n_jobs=-1, random_state=42
                )
                model.fit(pca.fit_transform(scaler.fit_transform(x)), train_df["hamd_binary"])
                prob_tr = _subject_probs(model, scaler, pca, real_tr, cols)
                prob_te = _subject_probs(model, scaler, pca, real_te, cols)
                y_tr = _subject_binary(real_tr).reindex(prob_tr.index)
                y_te = _subject_binary(real_te).reindex(prob_te.index)
                threshold = best_threshold(y_tr, prob_tr.to_numpy(), metric="f1")
                pred = (prob_te.to_numpy() >= threshold).astype(int)
                fold_rows.append(
                    {
                        "fold": fold,
                        "source": source,
                        "model": prefix.rstrip("_"),
                        "auc": float(roc_auc_score(y_te, prob_te.to_numpy())),
                        "f1": float(f1_score(y_te, pred, zero_division=0)),
                    }
                )
                modality_probs_tr[prefix] = prob_tr
                modality_probs_te[prefix] = prob_te

            # Late fusion on subject-level modality probabilities.
            names = list(modality_probs_tr.keys())
            if len(names) >= 2:
                X_tr = np.column_stack([modality_probs_tr[p] for p in names])
                X_te = np.column_stack([modality_probs_te[p] for p in names])
                y_tr = _subject_binary(real_tr).reindex(modality_probs_tr[names[0]].index)
                y_te = _subject_binary(real_te).reindex(modality_probs_te[names[0]].index)
                fusion = LogisticRegression(class_weight="balanced", max_iter=2000)
                fusion.fit(X_tr, y_tr)
                f_tr = fusion.predict_proba(X_tr)[:, 1]
                f_te = fusion.predict_proba(X_te)[:, 1]
                threshold = best_threshold(y_tr, f_tr, metric="f1")
                pred = (f_te >= threshold).astype(int)
                fold_rows.append(
                    {
                        "fold": fold,
                        "source": source,
                        "model": "fusion",
                        "auc": float(roc_auc_score(y_te, f_te)),
                        "f1": float(f1_score(y_te, pred, zero_division=0)),
                    }
                )

    report = pd.DataFrame(fold_rows)
    summary = (
        report.groupby(["source", "model"])[["auc", "f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["source", "model", "auc_mean", "auc_std", "f1_mean", "f1_std"]
    return summary


def run_v3_pipeline(
    n_subjects: int = 120,
    segments_per_subject: int = 5,
    separation_scale: float = 0.35,
    residual_scale: float = 0.5,
    calibrate_marginals: bool = True,
) -> dict:
    real = load_parquet_features("mpdd_utterance_features")
    if real is None:
        real = pd.read_csv(DATA_DIR / "mpdd_utterance_features.csv")
    syn = generate_hierarchical_synthetic(
        real,
        n_subjects=n_subjects,
        segments_per_subject=segments_per_subject,
        separation_scale=separation_scale,
        residual_scale=residual_scale,
        calibrate_marginals=calibrate_marginals,
    )
    validation = validate_vs_real(syn, real)
    report = evaluate_subject_aware(syn, real)
    return {"validation": validation, "report": report}


def grid_search_v3(
    settings: list[tuple[float, float]] | None = None,
    n_subjects: int = 300,
    segments_per_subject: int = 5,
) -> dict:
    settings = settings or [
        (0.4, 0.15),
        (0.5, 0.25),
        (0.6, 0.35),
        (0.8, 0.25),
    ]
    real = load_parquet_features("mpdd_utterance_features")
    if real is None:
        real = pd.read_csv(DATA_DIR / "mpdd_utterance_features.csv")
    real = real.dropna(subset=["hamd_score"]).copy()

    rows = []
    best = None
    for sep, res in settings:
        syn = generate_hierarchical_synthetic(
            real,
            n_subjects=n_subjects,
            segments_per_subject=segments_per_subject,
            separation_scale=sep,
            residual_scale=res,
            calibrate_marginals=True,
        )
        validation = validate_vs_real(syn, real)
        gate_count = sum(_gate_ok(v) for v in validation["gates"].values())
        rows.append(
            {
                "separation_scale": sep,
                "residual_scale": res,
                "feasible": validation["feasible"],
                "gate_count": gate_count,
                "note": validation["note"],
            }
        )
        if best is None or gate_count > best["gate_count"]:
            best = {
                "separation_scale": sep,
                "residual_scale": res,
                "feasible": validation["feasible"],
                "gate_count": gate_count,
            }

    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_RESULTS_CSV, index=False, encoding="utf-8-sig")
    BEST_PARAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_JSON.write_text(
        __import__("json").dumps(best, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_syn = generate_hierarchical_synthetic(
        real,
        n_subjects=n_subjects,
        segments_per_subject=segments_per_subject,
        separation_scale=best["separation_scale"],
        residual_scale=best["residual_scale"],
        calibrate_marginals=True,
    )
    report = evaluate_subject_aware(best_syn, real)
    real_rows = report[report["source"] == "real"].set_index("model")
    syn_rows = report[report["source"] == "synthetic"].set_index("model")
    deltas = [
        float(syn_rows.loc[model, "auc_mean"] - real_rows.loc[model, "auc_mean"])
        for model in real_rows.index
        if model in syn_rows.index
    ]
    downstream_ok = (
        all(d >= -0.03 for d in deltas)
        and bool(syn_rows["auc_mean"].max() < 0.98)
    )
    best["downstream_ok"] = bool(downstream_ok)
    best["feasible"] = bool(best.get("feasible") and downstream_ok)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_DIR / "synthetic_v3_subject_report.csv", index=False, encoding="utf-8-sig")
    return {"grid": grid, "best": best, "report": report}


def run_augmentation_pipeline(
    noise_scale: float = 0.03,
    k: int = 3,
    modality_noise_scales: dict[str, float] | None = None,
) -> dict:
    if modality_noise_scales is None:
        modality_noise_scales = {
            "face_": 0.03,
            "audio_": 0.03,
            "gait_": 0.005,
            "personality_": 0.005,
        }
    real = load_parquet_features("mpdd_utterance_features")
    if real is None:
        real = pd.read_csv(DATA_DIR / "mpdd_utterance_features.csv")
    real = real.dropna(subset=["hamd_score"]).copy()
    source = real[real["fold"] == "train"] if "fold" in real.columns else real
    augmented = generate_segment_augmented(
        source,
        k=k,
        noise_scale=noise_scale,
        modality_noise_scales=modality_noise_scales,
    )
    AUGMENTED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_parquet(AUGMENTED_PARQUET, index=False, compression="zstd", engine="pyarrow")
    validation = validate_vs_real(augmented, real)
    report = evaluate_subject_aware(
        augmented,
        real,
        use_segment_augmentation=True,
        augmentation_k=k,
        augmentation_noise=noise_scale,
        augmentation_modality_scales=modality_noise_scales,
    )
    real_rows = report[report["source"] == "real"]
    syn_rows = report[report["source"] == "synthetic"]
    auc_delta = float(syn_rows["auc_mean"].mean() - real_rows["auc_mean"].mean())
    downstream_ok = bool(auc_delta >= -0.03 and syn_rows["auc_mean"].max() < 0.98)
    validation["downstream"] = {
        "mean_auc_delta": round(auc_delta, 4),
        "ok": downstream_ok,
    }
    validation["feasible"] = bool(validation.get("feasible") and downstream_ok)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(
        OUTPUT_DIR / "synthetic_augmentation_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {"validation": validation, "report": report, "params": {"noise_scale": noise_scale, "k": k}}


def split_subject_holdout(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(df["subject_id"].astype(str).unique()))
    n_test = max(1, int(round(len(subjects) * test_ratio)))
    test_subjects = set(rng.choice(subjects, size=n_test, replace=False))
    test_df = df[df["subject_id"].astype(str).isin(test_subjects)]
    tune_df = df[~df["subject_id"].astype(str).isin(test_subjects)]
    return tune_df, test_df


def evaluate_final_holdout(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 3,
    noise_scale: float = 0.03,
    modality_noise_scales: dict[str, float] | None = None,
    balance_classes: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    train_df = train_df.dropna(subset=["hamd_score"]).copy()
    train_df["hamd_binary"] = (train_df["hamd_score"] >= 8).astype(int)
    test_df = test_df.dropna(subset=["hamd_score"]).copy()
    test_df["hamd_binary"] = (test_df["hamd_score"] >= 8).astype(int)
    aug_train = generate_segment_augmented(
        train_df,
        k=k,
        noise_scale=noise_scale,
        seed=seed,
        modality_noise_scales=modality_noise_scales,
        balance_classes=balance_classes,
    )

    def run_source(source_train: pd.DataFrame, source_name: str) -> list[dict]:
        rows: list[dict] = []
        modality_probs_tr = {}
        modality_probs_te = {}
        for prefix in MODALITY_PREFIXES:
            cols = [c for c in train_df.columns if c.startswith(prefix)]
            if not cols:
                continue
            scaler = StandardScaler()
            pca = PCA(
                n_components=min(48, len(cols), len(source_train) - 1),
                random_state=42,
            )
            model = RandomForestClassifier(
                n_estimators=150, class_weight="balanced", n_jobs=-1, random_state=42
            )
            model.fit(
                pca.fit_transform(
                    scaler.fit_transform(
                        source_train[cols].fillna(0).to_numpy(dtype="float64")
                    )
                ),
                source_train["hamd_binary"],
            )
            prob_tr = _subject_probs(model, scaler, pca, train_df, cols)
            prob_te = _subject_probs(model, scaler, pca, test_df, cols)
            y_tr = _subject_binary(train_df).reindex(prob_tr.index)
            y_te = _subject_binary(test_df).reindex(prob_te.index)
            threshold = best_threshold(y_tr, prob_tr.to_numpy(), metric="f1")
            pred = (prob_te.to_numpy() >= threshold).astype(int)
            rows.append(
                {
                    "source": source_name,
                    "model": prefix.rstrip("_"),
                    "auc": float(roc_auc_score(y_te, prob_te.to_numpy())),
                    "f1": float(f1_score(y_te, pred, zero_division=0)),
                }
            )
            lo, hi = _bootstrap_auc_ci(
                prob_te.index.to_series(),
                y_te,
                prob_te,
            )
            rows[-1]["auc_ci_low"] = round(lo, 4)
            rows[-1]["auc_ci_high"] = round(hi, 4)
            modality_probs_tr[prefix] = prob_tr
            modality_probs_te[prefix] = prob_te

        names = list(modality_probs_tr.keys())
        if len(names) >= 2:
            X_tr = np.column_stack([modality_probs_tr[p] for p in names])
            X_te = np.column_stack([modality_probs_te[p] for p in names])
            y_tr = _subject_binary(train_df).reindex(modality_probs_tr[names[0]].index)
            y_te = _subject_binary(test_df).reindex(modality_probs_te[names[0]].index)
            fusion = LogisticRegression(class_weight="balanced", max_iter=2000)
            fusion.fit(X_tr, y_tr)
            f_tr = fusion.predict_proba(X_tr)[:, 1]
            f_te = fusion.predict_proba(X_te)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(f_tr, y_tr)
            f_tr = calibrator.predict(f_tr)
            f_te = calibrator.predict(f_te)
            threshold = best_threshold(y_tr, f_tr, metric="f1")
            pred = (f_te >= threshold).astype(int)
            rows.append(
                {
                    "source": source_name,
                    "model": "fusion",
                    "auc": float(roc_auc_score(y_te, f_te)),
                    "f1": float(f1_score(y_te, pred, zero_division=0)),
                }
            )
            lo, hi = _bootstrap_auc_ci(
                prob_te.index.to_series(),
                y_te,
                pd.Series(f_te, index=prob_te.index),
            )
            rows[-1]["auc_ci_low"] = round(lo, 4)
            rows[-1]["auc_ci_high"] = round(hi, 4)
        return rows

    return pd.DataFrame(
        run_source(train_df, "real_final")
        + run_source(aug_train, "synthetic_final")
    )


def run_augmentation_optimization(
    settings: list[dict] | None = None,
    test_ratio: float = 0.2,
) -> dict:
    real = load_parquet_features("mpdd_utterance_features")
    if real is None:
        real = pd.read_csv(DATA_DIR / "mpdd_utterance_features.csv")
    real = real.dropna(subset=["hamd_score"]).copy()
    tune_df, test_df = split_subject_holdout(real, test_ratio=test_ratio)

    settings = settings or [
        {"face_": 0.005, "audio_": 0.03, "gait_": 0.005, "personality_": 0.01, "k": 3},
        {"face_": 0.03, "audio_": 0.03, "gait_": 0.005, "personality_": 0.005, "k": 3},
        {"face_": 0.03, "audio_": 0.03, "gait_": 0.005, "personality_": 0.05, "k": 3},
        {"face_": 0.05, "audio_": 0.03, "gait_": 0.005, "personality_": 0.01, "k": 3},
        {"face_": 0.03, "audio_": 0.03, "gait_": 0.005, "personality_": 0.01, "k": 5},
        {"face_": 0.01, "audio_": 0.01, "gait_": 0.005, "personality_": 0.01, "k": 5},
    ]
    rows = []
    for idx, scales in enumerate(settings):
        k = int(scales.get("k", 3))
        modality_scales = {key: value for key, value in scales.items() if key != "k"}
        report = evaluate_subject_aware(
            tune_df,
            tune_df,
            use_segment_augmentation=True,
            augmentation_k=k,
            augmentation_noise=0.03,
            augmentation_modality_scales=modality_scales,
        )
        syn = report[report["source"] == "synthetic"]
        real_rows = report[report["source"] == "real"]
        mean_delta = float(syn["auc_mean"].mean() - real_rows["auc_mean"].mean())
        aug = generate_segment_augmented(
            tune_df,
            k=k,
            noise_scale=0.03,
            seed=idx,
            modality_noise_scales=modality_scales,
        )
        validation = validate_vs_real(aug, tune_df)
        rows.append(
            {
                "config": idx,
                "face_noise": modality_scales.get("face_"),
                "audio_noise": modality_scales.get("audio_"),
                "gait_noise": modality_scales.get("gait_"),
                "personality_noise": modality_scales.get("personality_"),
                "k": k,
                "mean_auc_delta": round(mean_delta, 4),
                "validation_feasible": validation["feasible"],
                "gait_auc": float(syn[syn["model"] == "gait"]["auc_mean"].iloc[0]),
                "fusion_auc": float(syn[syn["model"] == "fusion"]["auc_mean"].iloc[0]),
                "personality_auc": float(syn[syn["model"] == "personality"]["auc_mean"].iloc[0]),
            }
        )
        print("tuning", idx, rows[-1])

    grid = pd.DataFrame(rows)
    feasible = grid[grid["validation_feasible"]]
    best_row = (
        feasible.loc[feasible["mean_auc_delta"].idxmax()]
        if not feasible.empty
        else grid.loc[grid["mean_auc_delta"].idxmax()]
    )
    best_scales = {
        "face_": float(best_row["face_noise"]),
        "audio_": float(best_row["audio_noise"]),
        "gait_": float(best_row["gait_noise"]),
        "personality_": float(best_row["personality_noise"]),
    }
    best_k = int(best_row.get("k", 3))
    final_report = evaluate_final_holdout(
        tune_df,
        test_df,
        k=best_k,
        noise_scale=0.03,
        modality_noise_scales=best_scales,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT_DIR / "augmentation_tuning_grid.csv", index=False, encoding="utf-8-sig")
    final_report.to_csv(
        OUTPUT_DIR / "synthetic_augmentation_final_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_params = {
        "k": best_k,
        "noise_scale": 0.03,
        "modality_noise_scales": best_scales,
        "selected_on": "tuning_subjects",
        "final_evaluated_on": "held_out_test_subjects",
    }
    DATA_DIR.joinpath("synthetic", "augmentation_best_params.json").write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "tuning_grid": grid,
        "best": best_row.to_dict(),
        "final_report": final_report,
        "params": best_params,
    }


def _default_best_augmentation_params() -> dict:
    path = DATA_DIR / "synthetic" / "augmentation_best_params.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "k": 3,
        "noise_scale": 0.03,
        "modality_noise_scales": {
            "face_": 0.03,
            "audio_": 0.03,
            "gait_": 0.005,
            "personality_": 0.005,
        },
    }


def run_repeated_holdout(n_repeats: int = 5, params: dict | None = None) -> dict:
    real = load_parquet_features("mpdd_utterance_features")
    if real is None:
        real = pd.read_csv(DATA_DIR / "mpdd_utterance_features.csv")
    real = real.dropna(subset=["hamd_score"]).copy()
    params = params or _default_best_augmentation_params()
    scales = params.get("modality_noise_scales", {})
    k = int(params.get("k", 3))
    noise = float(params.get("noise_scale", 0.03))

    reports = []
    for seed in range(1, n_repeats + 1):
        tune_df, test_df = split_subject_holdout(real, test_ratio=0.2, seed=seed)
        report = evaluate_final_holdout(
            tune_df,
            test_df,
            k=k,
            noise_scale=noise,
            modality_noise_scales=scales,
        )
        report["repeat"] = seed
        reports.append(report)
    report = pd.concat(reports, ignore_index=True)
    summary = (
        report.groupby(["source", "model"])["auc"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(
        OUTPUT_DIR / "synthetic_repeated_holdout_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUTPUT_DIR / "synthetic_repeated_holdout_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {"report": report, "summary": summary}


def build_augmented_pipeline(
    df: pd.DataFrame,
    k: int = 3,
    noise_scale: float = 0.03,
    modality_noise_scales: dict[str, float] | None = None,
    output_path: Path | str | None = None,
    seed: int = 42,
    augment: bool = True,
) -> dict:
    """Train a deployable subject pipeline on augmented segments."""
    df = df.dropna(subset=["hamd_score"]).copy()
    df["hamd_binary"] = (df["hamd_score"] >= 8).astype(int)
    aug = (
        generate_segment_augmented(
            df,
            k=k,
            noise_scale=noise_scale,
            seed=seed,
            modality_noise_scales=modality_noise_scales,
        )
        if augment
        else df.copy()
    )
    modalities = {}
    for prefix in MODALITY_PREFIXES:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        scaler = StandardScaler()
        pca = PCA(n_components=min(48, len(cols), len(aug) - 1), random_state=seed)
        model = RandomForestClassifier(
            n_estimators=150, class_weight="balanced", n_jobs=1, random_state=seed
        )
        x = aug[cols].fillna(0).to_numpy(dtype="float64")
        model.fit(pca.fit_transform(scaler.fit_transform(x)), aug["hamd_binary"])
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        top_features = []
        for idx in top_idx:
            comp = pca.components_[idx]
            top_features.append(cols[int(np.argmax(np.abs(comp)))])
        name = prefix.rstrip("_")
        modalities[name] = {
            "columns": cols,
            "scaler": scaler,
            "pca": pca,
            "model": model,
            "missing_prob": 0.5,
            "top_features": top_features,
            "pca_importance": {f"pc{idx}": float(importances[idx]) for idx in top_idx},
        }

    fusion_matrix = []
    for prefix in MODALITY_PREFIXES:
        name = prefix.rstrip("_")
        if name not in modalities:
            continue
        spec = modalities[name]
        x = aug[spec["columns"]].fillna(0).to_numpy(dtype="float64")
        prob = spec["model"].predict_proba(
            spec["pca"].transform(spec["scaler"].transform(x))
        )[:, 1]
        fusion_matrix.append(
            pd.Series(prob, index=aug.index).groupby(aug["subject_id"]).mean()
        )

    fusion_options = {}
    risk_bins = {"low": 0.4, "medium": 0.7}
    if len(fusion_matrix) >= 2:
        idx = fusion_matrix[0].index
        X = np.column_stack([m.reindex(idx) for m in fusion_matrix])
        y = aug.groupby("subject_id")["hamd_binary"].first().reindex(idx)
        fusion = LogisticRegression(class_weight="balanced", max_iter=2000)
        fusion.fit(X, y)
        f_prob = fusion.predict_proba(X)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(f_prob, y)
        calibrated = calibrator.predict(f_prob)
        threshold = float(best_threshold(y, calibrated, metric="f1"))
        fusion_options["|".join(modalities.keys())] = {
            "prefixes": list(modalities.keys()),
            "model": fusion,
            "threshold": threshold,
            "calibrator": calibrator,
        }
        risk_bins = {
            "low": float(np.quantile(calibrated, 0.4)),
            "medium": float(np.quantile(calibrated, 0.8)),
        }

    pipeline = {
        "version": 3,
        "subject_col": "subject_id",
        "label_col": "hamd_binary",
        "prefixes": list(modalities.keys()),
        "modalities": modalities,
        "fusion": fusion_options,
        "risk_bins": risk_bins,
        "trained_on": "augmented" if augment else "real",
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, out)
    return pipeline
