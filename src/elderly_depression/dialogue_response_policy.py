"""Domain response-policy pretraining for fast local dialogue replies."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from itertools import product
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import OUTPUT_DIR
from .dialogue_policy import is_situational_late_sleep


MODEL_PATH = OUTPUT_DIR / "dialogue_response_policy.pkl"
REPORT_PATH = OUTPUT_DIR / "dialogue_response_policy_report.json"

STRATEGIES = [
    "late_compensated",
    "late_compensated_tired",
    "late_recurrent",
    "late_acute",
    "sleep_onset",
    "sleep_maintenance",
    "sleep_early",
    "sleep_positive",
]

_LOCK = threading.Lock()
_CACHE: dict = {"mtime": None, "bundle": None}

ENOUGH_SLEEP_TERMS = [
    "睡够了",
    "睡够",
    "睡足了",
    "睡眠时长够",
    "时长睡够",
    "总时长够",
    "补够了",
    "补足了",
    "多睡了",
    "八小时",
    "八个小时",
    "九小时",
    "九个小时",
    "7小时",
    "8小时",
    "9小时",
]

FATIGUE_TERMS = [
    "没精神",
    "精神不好",
    "很累",
    "特别累",
    "白天困",
    "犯困",
    "头晕",
    "影响白天",
    "状态不好",
]

RECOVERY_TERMS = [
    "精神还好",
    "精神挺好",
    "精神不错",
    "精神很好",
    "不影响",
    "不受影响",
    "状态还行",
    "状态很好",
    "白天挺好",
    "白天没问题",
]

RECURRING_TERMS = [
    "经常",
    "总是",
    "最近总",
    "每天",
    "连续",
    "最近一直",
    "好几晚",
    "一周两三次",
    "一周三次",
]


def _contains(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def rule_sleep_strategy(text: str) -> str:
    compact = re.sub(r"\s+", "", str(text or ""))
    late = is_situational_late_sleep(compact)
    if late:
        enough = _contains(compact, ENOUGH_SLEEP_TERMS)
        fatigued = _contains(compact, FATIGUE_TERMS) and not _contains(
            compact,
            ["不累", "不太累", "没有影响", "不影响"],
        )
        recovered = _contains(compact, RECOVERY_TERMS)
        recurring = _contains(compact, RECURRING_TERMS)
        if recurring:
            return "late_recurrent"
        if enough and fatigued:
            return "late_compensated_tired"
        if enough and (recovered or not fatigued):
            return "late_compensated"
        return "late_acute"
    if any(term in compact for term in ["每晚", "睡不着", "入睡困难", "入睡难", "躺很久"]):
        return "sleep_onset"
    if any(term in compact for term in ["半夜醒", "老醒", "总醒", "睡不踏实", "容易醒"]):
        return "sleep_maintenance"
    if any(term in compact for term in ["早醒", "醒得早", "醒太早", "凌晨醒"]):
        return "sleep_early"
    if any(
        term in compact
        for term in ["睡得挺好", "睡得不错", "睡眠很好", "睡眠不错", "睡得踏实"]
    ):
        return "sleep_positive"
    return "unknown"


def build_strategy_response(strategy: str) -> str | None:
    responses = {
        "late_compensated": (
            "那就明白了，您这次只是睡得晚，但总时长睡够了，第二天精神也没受影响。"
            "偶尔一次不用当成失眠。您平时也常晚睡晚起，还是只有昨晚这样？"
        ),
        "late_compensated_tired": (
            "虽然总睡眠时长补够了，但白天还是累，就要再看夜里睡得踏不踏实。"
            "这种情况持续几天了，夜里会醒吗？"
        ),
        "late_recurrent": (
            "如果晚睡晚起经常出现，就算总时长够，作息也会越来越难稳定。"
            "可以先固定每天起床时间，白天午睡不超过半小时，再逐步提前上床。"
        ),
        "late_acute": (
            "偶尔因为看电影或别的事情晚睡一次，很常见。"
            "更值得看的是第二天有没有明显没精神、会不会靠白天长时间补觉。"
            "这种情况一周大概几次？"
        ),
    }
    return responses.get(strategy)


def _load_bundle() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime_ns
    with _LOCK:
        if _CACHE["bundle"] is None or _CACHE["mtime"] != mtime:
            _CACHE["bundle"] = joblib.load(MODEL_PATH)
            _CACHE["mtime"] = mtime
        return _CACHE["bundle"]


def predict_sleep_strategy(text: str) -> dict:
    deterministic = rule_sleep_strategy(text)
    bundle = _load_bundle()
    if bundle is None:
        return {
            "strategy": deterministic,
            "confidence": 1.0 if deterministic != "unknown" else 0.0,
            "source": "rules",
        }
    model: Pipeline = bundle["model"]
    probabilities = model.predict_proba([str(text)])[0]
    classes = list(model.classes_)
    index = int(probabilities.argmax())
    predicted = str(classes[index])
    confidence = float(probabilities[index])
    if deterministic != "unknown":
        return {
            "strategy": deterministic,
            "confidence": confidence,
            "source": "rules+pretrained",
            "model_strategy": predicted,
        }
    if confidence >= 0.62 and predicted.startswith(("late_", "sleep_")):
        return {
            "strategy": predicted,
            "confidence": confidence,
            "source": "pretrained",
        }
    return {
        "strategy": deterministic,
        "confidence": confidence,
        "source": "pretrained_low_confidence",
        "model_strategy": predicted,
    }


def warmup_response_policy() -> dict:
    bundle = _load_bundle()
    return {
        "ready": bundle is not None,
        "model_path": str(MODEL_PATH),
    }


def generate_training_rows() -> list[dict]:
    rows: list[dict] = []

    late_prefixes = [
        "昨晚看电影睡晚了",
        "昨晚追剧看到很晚",
        "前天和朋友聊到很晚才睡",
        "昨晚刷手机睡得晚",
        "昨晚聚会回来得晚",
        "最近总熬夜",
        "这阵子经常凌晨才睡",
    ]
    enough_parts = [
        "但是睡够了",
        "不过总时长睡够了",
        "因为早上多睡了一会儿，时长够了",
        "虽然睡得晚但补足了八小时",
        "总共也睡了九个小时",
    ]
    recovery_parts = [
        "所以精神还好",
        "第二天不影响",
        "白天精神不错",
        "状态还行",
    ]
    fatigue_parts = [
        "但第二天还是很累",
        "白天没精神",
        "虽然睡够但白天犯困",
        "第二天状态不好",
    ]
    for late, enough, recovery in product(
        late_prefixes[1:3], enough_parts, recovery_parts
    ):
        rows.append(
            {
                "text": f"{late}，{enough}，{recovery}",
                "strategy": "late_compensated",
                "group": f"late-compensated:{late}:{enough}",
            }
        )
    for late, enough, fatigue in product(late_prefixes[:3], enough_parts, fatigue_parts):
        rows.append(
            {
                "text": f"{late}，{enough}，{fatigue}",
                "strategy": "late_compensated_tired",
                "group": f"late-tired:{late}:{fatigue}",
            }
        )
    for late in late_prefixes[-2:]:
        for tail in [
            "，最近一直这样",
            "，一周两三次",
            "，连续好几晚了",
            "，几乎每天都这样",
            "，最近经常拖到凌晨",
            "，已经持续一周了",
        ]:
            rows.append(
                {
                    "text": late + tail,
                    "strategy": "late_recurrent",
                    "group": f"late-recurrent:{late}",
                }
            )
    for late in late_prefixes:
        for tail in ["", "，只这一次", "，第二天还行", "，有点担心", "，平时不这样"]:
            rows.append(
                {
                    "text": late + tail,
                    "strategy": "late_acute",
                    "group": f"late-acute:{late}",
                }
            )

    component_rows = {
        "sleep_onset": [
            "躺下很久还是睡不着",
            "晚上翻来覆去难以入睡",
            "每天入睡都很困难",
        ],
        "sleep_maintenance": [
            "半夜总是醒",
            "晚上容易醒好几次",
            "睡得不安稳，老醒",
        ],
        "sleep_early": [
            "早上醒得太早",
            "天没亮就醒来",
            "每天凌晨醒了睡不着",
        ],
        "sleep_positive": [
            "最近睡得挺好",
            "睡眠不错，白天也有精神",
            "这几天睡得踏实",
            "作息规律以后白天也轻松一些",
            "这两天睡够了，精神头不错",
        ],
    }
    for strategy, phrases in component_rows.items():
        for index, phrase in enumerate(phrases):
            for prefix in ["", "最近", "这阵子", "我最近"]:
                for suffix in ["", "，有点困扰", "，想问问怎么调整"]:
                    rows.append(
                        {
                            "text": f"{prefix}{phrase}{suffix}",
                            "strategy": strategy,
                            "group": f"{strategy}:{index}",
                        }
                    )
    return rows


def train_response_policy(
    output_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> dict:
    rows = generate_training_rows()
    frame = pd.DataFrame(rows).drop_duplicates(subset=["text", "strategy"])
    x = frame["text"].astype(str).tolist()
    y = frame["strategy"].astype(str).tolist()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for train_index, test_index in splitter.split(x, y):
        fold_model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(2, 5),
                        min_df=1,
                        max_features=12000,
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
        fold_model.fit(
            [x[index] for index in train_index],
            [y[index] for index in train_index],
        )
        predicted = fold_model.predict([x[index] for index in test_index])
        expected = [y[index] for index in test_index]
        probabilities = fold_model.predict_proba(
            [x[index] for index in test_index]
        ).max(axis=1)
        high_confidence = probabilities >= 0.62
        folds.append(
            {
                "accuracy": float(accuracy_score(expected, predicted)),
                "macro_f1": float(
                    f1_score(expected, predicted, average="macro", zero_division=0)
                ),
                "coverage": float(high_confidence.mean()),
                "high_conf_accuracy": (
                    float(
                        accuracy_score(
                            [
                                label
                                for label, keep in zip(expected, high_confidence)
                                if keep
                            ],
                            predicted[high_confidence],
                        )
                    )
                    if high_confidence.any()
                    else 0.0
                ),
            }
        )
    metrics = {
        "samples": len(frame),
        "labels": len(set(y)),
        "accuracy": float(pd.DataFrame(folds)["accuracy"].mean()),
        "macro_f1": float(pd.DataFrame(folds)["macro_f1"].mean()),
        "high_confidence_coverage": float(
            pd.DataFrame(folds)["coverage"].mean()
        ),
        "high_confidence_accuracy": float(
            pd.DataFrame(folds)["high_conf_accuracy"].mean()
        ),
        "folds": folds,
    }
    final_model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=1,
                    max_features=12000,
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
    final_model.fit(x, y)
    output = Path(output_path or MODEL_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final_model,
        "labels": sorted(set(y)),
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
    }
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
    with _LOCK:
        _CACHE["bundle"] = bundle
        _CACHE["mtime"] = output.stat().st_mtime_ns
    return {"model_path": str(output), **metrics}
