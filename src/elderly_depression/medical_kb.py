"""Offline MedlinePlus health-topic cache and retrieval."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
KB_PATH = DATA_DIR / "medical_kb" / "medlineplus_topics.json"

TOPIC_SPECS = [
    {
        "id": "depression",
        "query": "depression",
        "title_zh": "抑郁障碍",
        "triggers": ["抑郁", "情绪低落", "没兴趣", "没盼头", "想哭"],
    },
    {
        "id": "anxiety",
        "query": "anxiety",
        "title_zh": "焦虑",
        "triggers": ["焦虑", "紧张", "担心", "心慌", "害怕"],
    },
    {
        "id": "insomnia",
        "query": "insomnia",
        "title_zh": "失眠",
        "triggers": ["失眠", "睡不着", "入睡难", "半夜醒", "早醒", "睡眠"],
    },
    {
        "id": "high_blood_pressure",
        "query": "high blood pressure",
        "title_zh": "高血压",
        "triggers": ["高血压", "血压高", "血压"],
    },
    {
        "id": "diabetes",
        "query": "diabetes",
        "title_zh": "糖尿病",
        "triggers": ["糖尿病", "血糖", "口渴", "尿多"],
    },
    {
        "id": "heart_failure",
        "query": "heart failure",
        "title_zh": "心力衰竭",
        "triggers": ["心衰", "心力衰竭", "脚肿", "气短"],
    },
    {
        "id": "stroke",
        "query": "stroke",
        "title_zh": "卒中",
        "triggers": ["中风", "卒中", "单侧无力", "说话不清", "嘴歪"],
    },
    {
        "id": "copd",
        "query": "COPD",
        "title_zh": "慢性阻塞性肺疾病",
        "triggers": ["慢阻肺", "COPD", "咳痰", "气喘", "呼吸困难"],
    },
    {
        "id": "urinary_incontinence",
        "query": "urinary incontinence",
        "title_zh": "尿失禁",
        "triggers": ["尿失禁", "漏尿", "尿频", "尿急"],
    },
    {
        "id": "constipation",
        "query": "constipation",
        "title_zh": "便秘",
        "triggers": ["便秘", "排便困难", "大便干"],
    },
    {
        "id": "osteoarthritis",
        "query": "osteoarthritis",
        "title_zh": "骨关节炎",
        "triggers": ["关节炎", "关节痛", "膝盖痛", "骨关节炎"],
    },
    {
        "id": "hypothyroidism",
        "query": "hypothyroidism",
        "title_zh": "甲状腺功能减退",
        "triggers": ["甲减", "甲状腺", "怕冷", "反应慢"],
    },
    {
        "id": "anemia",
        "query": "anemia",
        "title_zh": "贫血",
        "triggers": ["贫血", "脸色白", "血红蛋白", "头晕乏力"],
    },
    {
        "id": "dehydration",
        "query": "dehydration",
        "title_zh": "脱水",
        "triggers": ["脱水", "口干", "喝水少", "尿少", "腹泻"],
    },
    {
        "id": "falls",
        "query": "falls",
        "title_zh": "跌倒",
        "triggers": ["跌倒", "摔倒", "站不稳", "平衡"],
    },
    {
        "id": "dementia",
        "query": "dementia",
        "title_zh": "痴呆",
        "triggers": ["痴呆", "记不住", "忘事", "迷路", "认知"],
    },
    {
        "id": "delirium",
        "query": "delirium",
        "title_zh": "谵妄",
        "triggers": ["突然糊涂", "意识不清", "急性意识", "谵妄"],
    },
    {
        "id": "grief",
        "query": "grief",
        "title_zh": "哀伤反应",
        "triggers": ["去世", "丧偶", "哀伤", "亲人离开", "思念"],
    },
]

_LOCK = threading.Lock()
_CACHE: dict = {"mtime": None, "topics": []}


def topic_spec(topic_id: str) -> dict | None:
    return next((item for item in TOPIC_SPECS if item["id"] == topic_id), None)


def _load_topics() -> list[dict]:
    if not KB_PATH.exists():
        return []
    mtime = KB_PATH.stat().st_mtime_ns
    with _LOCK:
        if _CACHE["mtime"] != mtime:
            payload = json.loads(KB_PATH.read_text(encoding="utf-8"))
            _CACHE["topics"] = payload.get("topics", [])
            _CACHE["mtime"] = mtime
        return list(_CACHE["topics"])


def search_medical_knowledge(text: str, max_results: int = 2) -> list[dict]:
    compact = re.sub(r"\s+", "", str(text or ""))
    scored = []
    for item in _load_topics():
        spec = topic_spec(str(item.get("id") or ""))
        if not spec:
            continue
        score = sum(
            1
            for trigger in spec.get("triggers", [])
            if trigger and trigger.lower() in compact.lower()
        )
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("title", "")))
    return [item for _, item in scored[:max_results]]


def format_medical_kb_for_prompt(items: list[dict]) -> str:
    if not items:
        return ""
    parts = []
    for item in items:
        parts.append(
            f"{item.get('title', '')}（MedlinePlus）: "
            f"{item.get('summary', '')[:900]}"
        )
    return " ".join(parts)
