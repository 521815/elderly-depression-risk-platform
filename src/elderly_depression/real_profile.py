"""Real-data characteristic profiling used to guide synthetic generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import DATA_DIR, OUTPUT_DIR
from .data.feature_cache import load_parquet_features

PROFILE_DIR = OUTPUT_DIR / "real_data_profile"
PROFILE_JSON = PROFILE_DIR / "real_data_profile.json"
MODALITY_PREFIXES = ["face_", "audio_", "gait_", "personality_"]


def _age_band(age: float) -> str:
    if age < 70:
        return "60-69"
    if age < 80:
        return "70-79"
    return "80+"


def analyze_tabular(df: pd.DataFrame, feature_cols: list[str], label_col: str = "depression") -> dict:
    df = df.dropna(subset=[label_col]).copy()
    y = df[label_col].astype(int)
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    missing = float(df[numeric_cols].isna().mean().mean()) if numeric_cols else 0.0

    age_bands = []
    if "age" in df.columns:
        for band, sub in df.groupby(df["age"].apply(_age_band)):
            age_bands.append(
                {"band": band, "n": int(len(sub)), "positive_rate": float(sub[label_col].mean())}
            )
    comorbidity_bands = []
    if "comorbidity_count" in df.columns:
        for band in ["0", "1-2", "3+"]:
            if band == "0":
                mask = df["comorbidity_count"] == 0
            elif band == "1-2":
                mask = df["comorbidity_count"].isin([1, 2])
            else:
                mask = df["comorbidity_count"] >= 3
            sub = df[mask]
            if len(sub):
                comorbidity_bands.append(
                    {"band": band, "n": int(len(sub)), "positive_rate": float(sub[label_col].mean())}
                )

    sample = df[numeric_cols].sample(min(12000, len(df)), random_state=42)
    corr = sample.corr(numeric_only=True)
    corr_rows = []
    if corr.shape[0] > 1:
        stack = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        stack = stack.reindex(stack.abs().sort_values(ascending=False).index)[:10]
        corr_rows = [
            {"pair": f"{i} | {j}", "r": round(float(v), 3)} for (i, j), v in stack.items()
        ]

    class_means = {}
    for label in sorted(y.unique()):
        sub = df[y == label][numeric_cols].mean(numeric_only=True)
        top = sub.reindex(sub.abs().sort_values(ascending=False).index)[:8]
        class_means[str(int(label))] = {
            str(k): round(float(v), 4) for k, v in top.items()
        }

    return {
        "rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "numeric_features": len(numeric_cols),
        "missing_rate": round(float(missing), 4),
        "age_bands": age_bands,
        "comorbidity_bands": comorbidity_bands,
        "top_correlations": corr_rows,
        "class_means": class_means,
    }


def _modality_profile(df: pd.DataFrame, prefix: str) -> dict:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return {}
    X = df[cols].fillna(0).to_numpy(dtype="float64")
    n = min(600, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=n, replace=False)
    sample = X[idx]

    pca = PCA(
        n_components=min(10, sample.shape[1], sample.shape[0] - 1),
        random_state=42,
    )
    pca.fit(sample)
    evr = pca.explained_variance_ratio_.cumsum()

    subject_col = "subject_id" if "subject_id" in df.columns else None
    within_var = None
    between_var = None
    if subject_col is not None and df[subject_col].nunique() > 1:
        sub = df.assign(_idx=np.arange(len(df))).groupby(subject_col)
        per_sub = [df.loc[g, cols].var(axis=0).mean() for _, g in sub.groups.items() if len(g) > 1]
        means = sub[cols].mean()
        within_var = float(np.mean(per_sub)) if per_sub else None
        between_var = float(means.var(axis=0).mean())

    # Class separation on top 5 principal components.
    y = df["hamd_binary"] if "hamd_binary" in df.columns else None
    d = None
    if y is not None:
        pc = pca.transform(sample)[:, : min(5, pca.n_components_)]
        yy = y.to_numpy()[idx]
        if len(np.unique(yy)) == 2:
            m0 = pc[yy == 0].mean(axis=0)
            m1 = pc[yy == 1].mean(axis=0)
            s = np.sqrt(
                (pc[yy == 0].var(axis=0) + pc[yy == 1].var(axis=0)) / 2
            )
            s[s == 0] = 1
            d = float(np.max(np.abs((m1 - m0) / s)))

    return {
        "n_features": len(cols),
        "pca_cumulative_variance": [round(float(v), 4) for v in evr.tolist()],
        "within_subject_variance": within_var,
        "between_subject_variance": between_var,
        "max_class_separation_d": d,
    }


def analyze_multimodal(df: pd.DataFrame) -> dict:
    df = df.copy()
    if "hamd_binary" not in df.columns and "hamd_score" in df.columns:
        df["hamd_binary"] = (df["hamd_score"] >= 8).astype(int)
    return {
        prefix.rstrip("_"): _modality_profile(df, prefix)
        for prefix in MODALITY_PREFIXES
        if any(c.startswith(prefix) for c in df.columns)
    }


def analyze_dialogue() -> dict:
    from .synthetic import TOPIC_KEYWORDS

    counts = {topic: 0 for topic in TOPIC_KEYWORDS}
    sentences = 0
    chars = 0
    records_dir = DATA_DIR / "user_records"
    if not records_dir.exists():
        return {"records": 0, "sentences": 0, "topic_counts": counts}
    for path in records_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for item in record.get("transcript") or []:
                if item.get("role") != "user":
                    continue
                text = str(item.get("text", ""))
                sentences += max(1, len([s for s in text.replace("。", "\n").split("\n") if s.strip()]))
                chars += len(text)
                for topic, words in TOPIC_KEYWORDS.items():
                    if any(w in text for w in words):
                        counts[topic] += 1
    return {
        "records": sum(1 for _ in records_dir.glob("*.jsonl")),
        "sentences": int(sentences),
        "avg_sentence_chars": round(chars / max(1, sentences), 2),
        "topic_counts": counts,
    }


def generate_profile() -> dict:
    charls = pd.read_csv(DATA_DIR / "charls_standardized.csv")
    exclude = {"id", "wave", "cesd_score", "depression"}
    feature_cols = [c for c in charls.columns if c not in exclude and c != "depression"]
    tabular = analyze_tabular(charls, feature_cols)

    multimodal_df = load_parquet_features("mpdd_utterance_features")
    if multimodal_df is None:
        multimodal_df = pd.read_csv(DATA_DIR / "mpdd_utterance_features.csv")
    multimodal = analyze_multimodal(multimodal_df)
    dialogue = analyze_dialogue()

    profile = {
        "tabular": tabular,
        "multimodal": multimodal,
        "dialogue": dialogue,
    }
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_JSON.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = []
    for band in tabular.get("age_bands", []):
        rows.append({"type": "age_band", **band})
    for band in tabular.get("comorbidity_bands", []):
        rows.append({"type": "comorbidity_band", **band})
    for pair in tabular.get("top_correlations", []):
        rows.append({"type": "correlation", **pair})
    pd.DataFrame(rows).to_csv(
        PROFILE_DIR / "tabular_profile.csv", index=False, encoding="utf-8-sig"
    )
    return profile


def profile_status() -> dict:
    if not PROFILE_JSON.exists():
        return {"ready": False}
    try:
        profile = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"ready": False}
    return {"ready": True, **profile}
