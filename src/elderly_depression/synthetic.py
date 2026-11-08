"""Generate simulated training groups that stay separate from real data.

Methods:
- Dialogue: topic-level sentence recombination from real saved transcripts plus
  a built-in clinical phrase bank.
- Tabular/multimodal: within-class interpolation (SMOTE-like) plus small noise.

Scientific boundary: synthetic rows are never merged with real data; real data
is only used for evaluation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA

from .config import DATA_DIR
from .features.text_features import extract_text_features
from .models.tabular import prepare_X

SYNTHETIC_DIR = DATA_DIR / "synthetic"
DIALOGUE_CSV = SYNTHETIC_DIR / "dialogue_samples.csv"
TABULAR_CSV = SYNTHETIC_DIR / "charls_synthetic.csv"
MULTIMODAL_PARQUET = SYNTHETIC_DIR / "mpdd_synthetic.parquet"
MULTIMODAL_CSV = SYNTHETIC_DIR / "mpdd_synthetic.csv"
REPORT_JSON = SYNTHETIC_DIR / "report.json"
DIALOGUE_VALIDATION_JSON = SYNTHETIC_DIR / "dialogue_validation.json"

MODALITY_PREFIXES = ["face_", "audio_", "gait_", "personality_"]

TOPIC_PHRASES = {
    "睡眠差": [
        "最近总是睡不着，翻来覆去到很晚。",
        "半夜经常醒，醒了就很难再睡着。",
        "早上醒得特别早，白天也没有精神。",
        "睡觉比以前差多了，一晚上要醒好几次。",
    ],
    "兴趣减退": [
        "以前喜欢下棋，现在完全提不起兴趣。",
        "什么都不想动，连门都不想出。",
        "以前爱听戏，现在觉得没意思。",
        "干什么都觉得没劲，心里空落落的。",
    ],
    "情绪低落": [
        "这段时间心里很难受，总想哭。",
        "经常觉得闷闷不乐，开心不起来。",
        "心里像堵着一块石头，特别压抑。",
        "一想到家里的事就烦，情绪很低落。",
    ],
    "躯体不适": [
        "腰一直疼，走路都没力气。",
        "最近胃口很差，吃不下饭。",
        "经常头晕、心慌，胸口发闷。",
        "浑身不舒服，又说不出具体哪里疼。",
    ],
    "认知下降": [
        "记性差了很多，经常忘记事情。",
        "注意力很难集中，看电视也看不进去。",
        "脑子反应比以前慢，有时候糊里糊涂的。",
    ],
    "社交孤立": [
        "平时一个人待着，没人说话。",
        "子女不在身边，邻居也很少来往。",
        "感觉特别孤单，好像没人真正理解我。",
    ],
    "无望感": [
        "觉得生活没什么盼头了。",
        "看不到希望，不知道以后怎么过。",
        "有时候觉得这样熬下去没有意义。",
    ],
    "自责/负担": [
        "觉得自己拖累了家里人。",
        "总觉得自己没用，是家里的负担。",
        "对不起孩子们，让他们为我操心。",
    ],
    "自杀意念": [
        "有时候觉得活着没意思，还不如走了。",
        "不想再活下去了，觉得太累了。",
        "有想过结束生命，但是又怕家人难过。",
    ],
    "保护因素": [
        "子女经常打电话来，心里踏实一些。",
        "邻居会来串门，有人陪着说说话。",
        "每天出门散步，感觉心情好一点。",
        "有家人照顾，心里踏实多了。",
    ],
}

TOPIC_KEYWORDS = {
    "睡眠差": ["睡", "醒", "失眠", "精神"],
    "兴趣减退": ["兴趣", "没意思", "不想动", "不出门"],
    "情绪低落": ["难受", "想哭", "压抑", "烦", "低落", "闷闷不乐", "开心不起来", "心情不好"],
    "躯体不适": ["疼", "痛", "没力气", "吃不下", "头晕", "心慌", "不舒服"],
    "认知下降": ["记", "忘", "注意力", "反应慢", "糊涂"],
    "社交孤立": ["一个人", "没人", "孤单", "孤独"],
    "无望感": ["盼头", "希望", "熬"],
    "自责/负担": ["拖累", "负担", "没用", "对不起"],
    "自杀意念": ["死", "结束生命", "不想活", "伤害自己"],
    "保护因素": ["子女", "老伴", "朋友", "邻居", "散步", "照顾", "说话"],
}


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[。！？!?；;\n]+", text or "") if s.strip()]


def _tag_sentence(text: str) -> str | None:
    for topic, words in TOPIC_KEYWORDS.items():
        if any(w in text for w in words):
            return topic
    return None


def _collect_source_sentences() -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {topic: list(sentences) for topic, sentences in TOPIC_PHRASES.items()}
    records_dir = DATA_DIR / "user_records"
    if not records_dir.exists():
        return pools
    for path in records_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            transcript = record.get("transcript") or []
            for item in transcript:
                if item.get("role") != "user":
                    continue
                for sentence in _split_sentences(item.get("text", "")):
                    topic = _tag_sentence(sentence)
                    if topic and sentence not in pools[topic]:
                        pools[topic].append(sentence)
    return pools


def generate_dialogue_samples(
    n: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pools = _collect_source_sentences()
    risk_probs = {"low": (0.05, 0.35), "medium": (0.40, 0.65), "high": (0.70, 0.95)}
    rows = []
    for i in range(n):
        risk = rng.choice(["low", "medium", "high"], p=[0.35, 0.45, 0.20])
        mild = ["睡眠差", "兴趣减退", "情绪低落", "躯体不适", "社交孤立", "认知下降"]
        if risk == "low":
            selected = [str(rng.choice(mild)), "保护因素"]
        elif risk == "medium":
            selected = list(rng.choice(mild, size=int(rng.integers(2, 4)), replace=False))
        else:
            selected = list(rng.choice(mild, size=2, replace=False))
            selected += ["无望感", "自责/负担", "自杀意念"]
        turns = [
            {
                "role": "user",
                "text": str(rng.choice(pools[topic])),
            }
            for topic in selected
        ]
        transcript = [
            {
                "role": "assistant",
                "text": "您好，我先陪您慢慢聊一会儿。最近这一周感觉怎么样？",
            }
        ] + turns
        lo, hi = risk_probs[risk]
        rows.append(
            {
                "sample_id": f"dia-syn-{i:04d}",
                "risk_level": risk,
                "depression_probability": round(float(rng.uniform(lo, hi)), 4),
                "topics": "、".join(selected),
                "cues": "、".join(selected),
                "transcript_json": json.dumps(transcript, ensure_ascii=False),
            }
        )
    df = pd.DataFrame(rows)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIALOGUE_CSV, index=False, encoding="utf-8-sig")
    return df


def _interpolate(
    X: pd.DataFrame,
    y: pd.Series,
    n_per_class: int,
    seed: int,
    noise_scale: float = 0.02,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X_out: list[np.ndarray] = []
    y_out: list[int] = []
    for label in sorted(y.unique()):
        sub = X[y == label]
        arr = sub.to_numpy(dtype="float64")
        if len(arr) < 2:
            arr = np.repeat(arr, 2, axis=0)
        std = arr.std(axis=0)
        std[std == 0] = 1.0
        for _ in range(n_per_class):
            i, j = rng.integers(0, len(arr), size=2)
            alpha = float(rng.uniform(0.1, 0.9))
            syn = arr[i] + alpha * (arr[j] - arr[i])
            syn = syn + rng.normal(0, noise_scale, size=syn.shape) * std
            X_out.append(syn)
            y_out.append(int(label))
    return pd.DataFrame(X_out, columns=X.columns), pd.Series(y_out, name=y.name)


def _interpolate_grouped(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_per_class: int,
    seed: int,
    noise_scale: float = 0.02,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X_out: list[np.ndarray] = []
    y_out: list[int] = []
    for label in sorted(y.unique()):
        label_idx = y[y == label].index
        generated = 0
        for group in sorted(groups.unique()):
            idx = label_idx.intersection(groups[groups == group].index)
            if len(idx) < 2:
                continue
            share = len(idx) / max(1, len(label_idx))
            n = max(1, round(n_per_class * share))
            arr = X.loc[idx].to_numpy(dtype="float64")
            std = arr.std(axis=0)
            std[std == 0] = 1.0
            for _ in range(n):
                i, j = rng.integers(0, len(arr), size=2)
                alpha = float(rng.uniform(0.1, 0.9))
                syn = arr[i] + alpha * (arr[j] - arr[i])
                syn = syn + rng.normal(0, noise_scale, size=syn.shape) * std
                X_out.append(syn)
                y_out.append(int(label))
            generated += n
        if generated < n_per_class:
            arr = X.loc[label_idx].to_numpy(dtype="float64")
            if len(arr) < 2:
                arr = np.repeat(arr, 2, axis=0)
            std = arr.std(axis=0)
            std[std == 0] = 1.0
            for _ in range(n_per_class - generated):
                i, j = rng.integers(0, len(arr), size=2)
                alpha = float(rng.uniform(0.1, 0.9))
                syn = arr[i] + alpha * (arr[j] - arr[i])
                syn = syn + rng.normal(0, noise_scale, size=syn.shape) * std
                X_out.append(syn)
                y_out.append(int(label))
    return pd.DataFrame(X_out, columns=X.columns), pd.Series(y_out, name=y.name)


def _age_band(age: float) -> str:
    if age < 70:
        return "60-69"
    if age < 80:
        return "70-79"
    return "80+"


def generate_tabular_synthetic(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "depression",
    n_per_class: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    df = df.dropna(subset=[label_col]).copy()
    y = df[label_col].astype(int)
    X = prepare_X(df, feature_cols).loc[y.index]
    X_syn, y_syn = _interpolate(X, y, n_per_class=n_per_class, seed=seed)
    out = X_syn.copy()
    out[label_col] = y_syn.values
    out.insert(0, "sample_id", [f"tab-syn-{i:05d}" for i in range(len(out))])
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABULAR_CSV, index=False, encoding="utf-8-sig")
    return out


def generate_tabular_synthetic_enhanced(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "depression",
    n_per_class: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    df = df.dropna(subset=[label_col]).copy()
    y = df[label_col].astype(int)
    X = prepare_X(df, feature_cols).loc[y.index]
    if "age" in df.columns:
        groups = df.loc[y.index, "age"].apply(_age_band)
    else:
        groups = pd.Series("all", index=y.index)
    X_syn, y_syn = _interpolate_grouped(
        X, y, groups, n_per_class=n_per_class, seed=seed, noise_scale=0.02
    )
    out = X_syn.copy()
    out[label_col] = y_syn.values
    out.insert(0, "sample_id", [f"tab-syn-{i:05d}" for i in range(len(out))])
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABULAR_CSV, index=False, encoding="utf-8-sig")
    return out


def generate_multimodal_synthetic(
    df: pd.DataFrame,
    n_per_class: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    df = df.dropna(subset=["hamd_score"]).copy()
    df["hamd_binary"] = (df["hamd_score"] >= 8).astype(int)
    feature_cols = [
        c
        for c in df.columns
        if any(c.startswith(p) for p in MODALITY_PREFIXES)
    ]
    if not feature_cols:
        raise ValueError("多模态输入缺少 face_/audio_/gait_/personality_ 特征列")
    X = df[feature_cols].fillna(0).astype("float32")
    X_syn, y_syn = _interpolate(
        X, df["hamd_binary"], n_per_class=n_per_class, seed=seed, noise_scale=0.01
    )
    out = X_syn.copy()
    out.insert(0, "subject_id", [f"syn-{i:05d}" for i in range(len(out))])
    out.insert(1, "segment", [f"syn-seg-{i:05d}" for i in range(len(out))])
    out["fold"] = "synthetic"
    out["hamd_binary"] = y_syn.values
    out["hamd_score"] = np.where(y_syn.values == 1, 12.0, 4.0)
    out["label2"] = y_syn.values
    out["label3"] = np.where(y_syn.values == 1, 2, 0)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(MULTIMODAL_PARQUET, index=False, compression="zstd", engine="pyarrow")
    out.head(200).to_csv(MULTIMODAL_CSV, index=False, encoding="utf-8-sig")
    return out


def generate_multimodal_synthetic_enhanced(
    df: pd.DataFrame,
    n_per_class: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate in the real PCA subspace with class-conditional covariance."""
    df = df.dropna(subset=["hamd_score"]).copy()
    df["hamd_binary"] = (df["hamd_score"] >= 8).astype(int)
    feature_cols = [
        c
        for c in df.columns
        if any(c.startswith(p) for p in MODALITY_PREFIXES)
    ]
    if not feature_cols:
        raise ValueError("多模态输入缺少 face_/audio_/gait_/personality_ 特征列")
    X = df[feature_cols].fillna(0).astype("float32")
    y = df["hamd_binary"].to_numpy()
    rng = np.random.default_rng(seed)

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for label in sorted(np.unique(y)):
        Xc = X[y == label].to_numpy(dtype="float64")
        n_comp = min(32, Xc.shape[1], Xc.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=seed)
        scores = pca.fit_transform(Xc)
        mean = scores.mean(axis=0)
        cov = np.cov(scores, rowvar=False)
        cov = (cov + cov.T) / 2.0
        cov = cov + np.eye(cov.shape[0]) * max(1e-6, float(np.trace(cov)) / cov.shape[0] * 0.05)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        cov = (eigvecs * eigvals) @ eigvecs.T
        generated = rng.multivariate_normal(
            mean,
            cov,
            size=n_per_class,
            check_valid="ignore",
        )
        generated = generated + rng.normal(
            0, scores.std(axis=0) * 0.08, size=generated.shape
        )
        syn = pca.inverse_transform(generated)
        rows.append(syn)
        labels.extend([int(label)] * n_per_class)

    X_syn = np.vstack(rows)
    out = pd.DataFrame(X_syn, columns=X.columns, dtype="float32")
    out.insert(0, "subject_id", [f"syn-{i:05d}" for i in range(len(out))])
    out.insert(1, "segment", [f"syn-seg-{i:05d}" for i in range(len(out))])
    out["fold"] = "synthetic"
    out["hamd_binary"] = labels
    out["hamd_score"] = np.where(np.asarray(labels) == 1, 12.0, 4.0)
    out["label2"] = labels
    out["label3"] = np.where(np.asarray(labels) == 1, 2, 0)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(MULTIMODAL_PARQUET, index=False, compression="zstd", engine="pyarrow")
    out.head(200).to_csv(MULTIMODAL_CSV, index=False, encoding="utf-8-sig")
    return out


def generate_all(
    dialogue_n: int = 80,
    tabular_n: int = 400,
    multimodal_n: int = 300,
    seed: int = 42,
) -> dict:
    dialogue = generate_dialogue_samples(n=dialogue_n, seed=seed)
    dialogue_validation = validate_dialogue_samples(dialogue)
    tabular = generate_tabular_synthetic_enhanced(
        pd.read_csv(DATA_DIR / "charls_standardized.csv"),
        feature_cols=[
            c
            for c in pd.read_csv(DATA_DIR / "charls_standardized.csv", nrows=1).columns
            if c not in {"id", "wave", "cesd_score", "depression"}
        ],
        n_per_class=tabular_n,
        seed=seed,
    )
    from .data.feature_cache import load_parquet_features

    multimodal_source = load_parquet_features("mpdd_utterance_features")
    if multimodal_source is None:
        multimodal_source = pd.read_csv(DATA_DIR / "mpdd_utterance_features.csv")
    multimodal = generate_multimodal_synthetic_enhanced(
        multimodal_source,
        n_per_class=multimodal_n,
        seed=seed,
    )
    report = {
        "trained_at": pd.Timestamp.now().isoformat(),
        "dialogue": {
            "rows": len(dialogue),
            "path": str(DIALOGUE_CSV),
            "validation": dialogue_validation,
            "risk_counts": {
                str(k): int(v)
                for k, v in dialogue["risk_level"].value_counts().to_dict().items()
            },
        },
        "tabular": {
            "rows": len(tabular),
            "path": str(TABULAR_CSV),
            "positive_rate": float(tabular["depression"].mean()),
        },
        "multimodal": {
            "rows": len(multimodal),
            "path": str(MULTIMODAL_PARQUET),
            "positive_rate": float(multimodal["hamd_binary"].mean()),
        },
        "methods": {
            "dialogue": "主题片段重组（真实语句池 + 内置临床语句）",
            "tabular": "年龄分层类内插值 + 小噪声",
            "multimodal": "真实 PCA 子空间 + 类别条件协方差生成",
        },
        "note": "模拟组仅用于训练侧实验，不与真实数据混合；评估只用真实数据。",
    }
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def synthetic_status() -> dict:
    if not REPORT_JSON.exists():
        return {"ready": False, "note": "尚未生成模拟数据"}
    try:
        report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"ready": False, "note": "模拟数据报告损坏"}
    sizes = {}
    for key, path in [
        ("dialogue", DIALOGUE_CSV),
        ("tabular", TABULAR_CSV),
        ("multimodal", MULTIMODAL_PARQUET),
    ]:
        sizes[key] = path.stat().st_size if path.exists() else 0
    report["sizes"] = sizes
    report["dialogue_validation"] = report.get("dialogue", {}).get("validation")
    return {"ready": True, **report}


def delete_synthetic() -> bool:
    removed = False
    for path in [
        DIALOGUE_CSV,
        TABULAR_CSV,
        MULTIMODAL_PARQUET,
        MULTIMODAL_CSV,
        REPORT_JSON,
        DIALOGUE_VALIDATION_JSON,
    ]:
        if path.exists():
            path.unlink()
            removed = True
    return removed


def _risk_from_cues(cues: list[str]) -> str:
    high = {"自杀意念"}
    medium_high = {"无望感", "自责/负担"}
    mild = {
        "睡眠差",
        "兴趣减退",
        "情绪低落",
        "躯体不适",
        "认知下降",
        "社交孤立",
    }
    if any(c in high for c in cues):
        return "high"
    if any(c in medium_high for c in cues):
        return "high"
    count = sum(1 for c in cues if c in mild)
    if count >= 4:
        return "high"
    if count >= 2:
        return "medium"
    return "low"


def validate_dialogue_samples(dialogue_df: pd.DataFrame | None = None) -> dict:
    if dialogue_df is None:
        if not DIALOGUE_CSV.exists():
            return {"ready": False, "note": "尚未生成对话模拟组"}
        dialogue_df = pd.read_csv(DIALOGUE_CSV)

    user_texts: list[str] = []
    generated_topics: set[str] = set()
    generated_cues: list[list[str]] = []
    for transcript_json in dialogue_df["transcript_json"]:
        try:
            transcript = json.loads(transcript_json)
        except (json.JSONDecodeError, TypeError):
            continue
        cues: list[str] = []
        for item in transcript:
            if item.get("role") != "user":
                continue
            text = str(item.get("text", ""))
            user_texts.append(text)
            for topic, words in TOPIC_KEYWORDS.items():
                if any(w in text for w in words):
                    generated_topics.add(topic)
                    if topic not in cues:
                        cues.append(topic)
        generated_cues.append(cues)

    gen_feats = pd.DataFrame([extract_text_features(t) for t in user_texts])
    gen_len = gen_feats["char_count"].to_numpy(dtype="float64")
    gen_neg = gen_feats["negative_density"].to_numpy(dtype="float64")

    pools = _collect_source_sentences()
    source_sentences = [s for sentences in pools.values() for s in sentences]
    src_feats = pd.DataFrame([extract_text_features(s) for s in source_sentences])
    src_len = src_feats["char_count"].to_numpy(dtype="float64")
    src_neg = src_feats["negative_density"].to_numpy(dtype="float64")

    ks_len = float(ks_2samp(gen_len, src_len).statistic)
    ks_neg = float(ks_2samp(gen_neg, src_neg).statistic)
    topic_coverage = float(
        len(generated_topics.intersection(TOPIC_PHRASES.keys()))
        / max(1, len(TOPIC_PHRASES))
    )

    expected = dialogue_df["risk_level"].tolist()
    inferred = [_risk_from_cues(cues) for cues in generated_cues]
    risk_order = {"low": 0, "medium": 1, "high": 2}
    exact = sum(a == b for a, b in zip(expected, inferred)) / max(1, len(expected))
    relaxed = sum(
        abs(risk_order[a] - risk_order[b]) <= 1
        for a, b in zip(expected, inferred)
    ) / max(1, len(expected))

    gates = {
        "sentence_length_ks": bool(ks_len <= 0.15),
        "negative_density_ks": bool(ks_neg <= 0.15),
        "topic_coverage": bool(topic_coverage >= 0.8),
        "risk_label_exact": bool(exact >= 0.7),
        "risk_label_relaxed": bool(relaxed >= 0.9),
    }
    feasible = all(gates.values())
    result = {
        "ready": True,
        "rows": int(len(dialogue_df)),
        "metrics": {
            "sentence_length_ks_d": round(ks_len, 4),
            "negative_density_ks_d": round(ks_neg, 4),
            "topic_coverage": round(topic_coverage, 4),
            "risk_label_exact": round(exact, 4),
            "risk_label_relaxed": round(relaxed, 4),
        },
        "gates": gates,
        "feasible": feasible,
        "note": "通过对话分布与风险标签一致性验证" if feasible else "未通过对话验证门槛",
    }
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    DIALOGUE_VALIDATION_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
