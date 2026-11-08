"""Utilities for de-identified research dialogue collection and evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR
from .negation import has_any_unnegated_keyword


RESEARCH_DIR = DATA_DIR / "research_dialogues"
SESSION_FILE = RESEARCH_DIR / "dialogue_sessions.csv"

SESSION_FIELDS = [
    "subject_id",
    "session_id",
    "collected_at",
    "age",
    "gender",
    "comorbidity_count",
    "scale_name",
    "scale_score",
    "depression_label",
    "transcript_json",
    "consent_code",
    "mpdd_risk",
]

SYMPTOM_PATTERNS = {
    "sleep": ["睡不着", "失眠", "半夜醒", "醒得早", "睡眠", "多梦", "作息", "难调"],
    "interest": ["没意思", "没兴趣", "不想动", "不想出门", "提不起兴趣", "什么都不想做", "不想做事"],
    "mood": ["难过", "伤心", "低落", "想哭", "心里堵", "压抑"],
    "somatic": [
        "疼",
        "痛",
        "乏力",
        "没力气",
        "没有力气",
        "没劲",
        "吃不下",
        "胃口",
        "头晕",
        "胸闷",
        "肩膀",
        "肩",
        "脖子",
        "颈椎",
        "腰",
        "背",
        "膝盖",
        "关节",
        "髋",
        "肚子",
        "胃",
        "胸口",
        "手臂",
        "手指",
        "脚踝",
        "脚",
        "牙",
    ],
    "cognition": ["记不住", "忘事", "注意力", "反应慢", "记性"],
    "social": ["孤独", "孤单", "没人说话", "没人陪", "一个人"],
    "negative_cognition": ["没盼头", "绝望", "拖累", "负担", "没用", "对不起"],
}

SAFETY_PATTERNS = [
    "想死",
    "自杀",
    "不想活",
    "活不下去",
    "活着没意思",
    "活着没意义",
    "不如走了",
    "走了算了",
    "伤害自己",
]

FUNCTIONAL_PATTERNS = [
    "做不了",
    "影响生活",
    "需要帮忙",
    "不想出门",
    "没法",
    "困难",
    "需要人照顾",
]


def _parse_transcript(value) -> list[dict]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def extract_structured_labels(transcript) -> dict:
    messages = _parse_transcript(transcript)
    user_texts = [
        str(message.get("text") or "").strip()
        for message in messages
        if message.get("role") == "user"
    ]
    text = "\n".join(user_texts)
    symptoms = [
        label
        for label, keywords in SYMPTOM_PATTERNS.items()
        if has_any_unnegated_keyword(text, keywords)
    ]
    safety_risk = has_any_unnegated_keyword(text, SAFETY_PATTERNS)
    functional_impact = has_any_unnegated_keyword(text, FUNCTIONAL_PATTERNS)
    duration_match = re.search(
        r"(\d+|[一二三四五六七八九十]+)\s*(天|周|星期|个月|月|年)",
        text,
    )
    if duration_match:
        duration_text = duration_match.group(0)
    elif any(word in text for word in ["最近", "这阵子", "这段时间"]):
        duration_text = "近期"
    else:
        duration_text = ""
    result = {
        "dlg_user_turns": len(user_texts),
        "dlg_text_length": len(text),
        "dlg_symptom_count": len(symptoms),
        "dlg_functional_impact": int(functional_impact),
        "dlg_safety_risk": int(safety_risk),
        "dlg_duration_text": duration_text,
        "dlg_symptoms": "、".join(symptoms),
    }
    for label in SYMPTOM_PATTERNS:
        result[f"dlg_{label}"] = int(label in symptoms)
    return result


def normalize_session(record: dict) -> dict:
    subject_id = str(record.get("subject_id") or "").strip()
    session_id = str(record.get("session_id") or "").strip()
    if not subject_id or not session_id:
        raise ValueError("Each session requires subject_id and session_id")
    if "depression_label" not in record:
        raise ValueError("Each session requires depression_label")
    transcript = record.get("transcript_json") or record.get("transcript") or []
    structured = extract_structured_labels(transcript)
    normalized = {
        key: record.get(key)
        for key in SESSION_FIELDS
        if key not in {"transcript_json"}
    }
    normalized["subject_id"] = subject_id
    normalized["session_id"] = session_id
    normalized["transcript_json"] = (
        json.dumps(transcript, ensure_ascii=False)
        if not isinstance(transcript, str)
        else transcript
    )
    normalized.update(structured)
    return normalized


def build_session_frame(records: list[dict]) -> pd.DataFrame:
    rows = [normalize_session(record) for record in records]
    frame = pd.DataFrame(rows)
    for column in ["age", "comorbidity_count", "scale_score", "depression_label", "mpdd_risk"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_sessions(path: Path | str | None = None) -> pd.DataFrame:
    source = Path(path or SESSION_FILE)
    if not source.exists():
        raise FileNotFoundError(f"Research dialogue file not found: {source}")
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("sessions", [])
    else:
        records = pd.read_csv(source).to_dict("records")
    return build_session_frame(records)


def save_sessions(frame: pd.DataFrame, path: Path | str | None = None) -> Path:
    target = Path(path or SESSION_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def subject_level_split(
    frame: pd.DataFrame,
    label_col: str = "depression_label",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Session frame is empty")
    if "subject_id" not in frame.columns or label_col not in frame.columns:
        raise ValueError("subject_id and label column are required")
    subject_table = (
        frame.groupby("subject_id", as_index=False)[label_col]
        .max()
        .dropna(subset=[label_col])
    )
    if len(subject_table) < 6:
        raise ValueError("At least 6 labeled subjects are required")
    labels = subject_table[label_col].astype(int).to_numpy()
    stratify = labels if np.bincount(labels, minlength=2).min() >= 2 else None
    train_subjects, test_subjects = train_test_split(
        subject_table["subject_id"].astype(str).to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_labels = subject_table.set_index("subject_id").loc[train_subjects, label_col].astype(int)
    inner_test_size = val_size / (1.0 - test_size)
    stratify_inner = train_labels if train_labels.value_counts().min() >= 2 else None
    train_subjects, val_subjects = train_test_split(
        train_subjects,
        test_size=inner_test_size,
        random_state=random_state + 1,
        stratify=stratify_inner,
    )
    split_map = {
        **{str(subject): "train" for subject in train_subjects},
        **{str(subject): "validation" for subject in val_subjects},
        **{str(subject): "test" for subject in test_subjects},
    }
    result = frame.copy()
    result["split"] = result["subject_id"].astype(str).map(split_map)
    if result["split"].isna().any():
        raise ValueError("Some sessions have no subject-level split")
    overlap = (
        result.groupby("split")["subject_id"].apply(lambda values: set(values))
    )
    if overlap["train"] & overlap["validation"] or overlap["train"] & overlap["test"] or overlap["validation"] & overlap["test"]:
        raise AssertionError("Subject leakage detected across splits")
    return result
