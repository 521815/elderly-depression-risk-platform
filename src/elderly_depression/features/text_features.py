"""Rule-based text features for Chinese elderly dialogue, with optional model hooks."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

NEGATIVE_WORDS = [
    "难过", "伤心", "悲伤", "孤独", "没意思", "没劲", "不想", "不愿",
    "累", "疲惫", "烦", "烦躁", "失眠", "睡不着", "没用", "没用处",
    "拖累", "负担", "绝望", "想死", "自杀", "害怕", "担心", "痛苦",
    "提不起兴趣", "没兴趣", "空虚", "没盼头", "受够了", "熬不住",
]

POSITIVE_WORDS = [
    "开心", "高兴", "喜欢", "不错", "舒服", "幸福", "满意", "期待",
    "轻松", "愉快", "有意思", "有盼头", "舒服多了", "挺好",
]

SLEEP_WORDS = ["睡觉", "入睡", "失眠", "睡不着", "醒", "睡不好"]
INTEREST_WORDS = ["兴趣", "爱好", "想玩", "没兴趣", "提不起兴趣", "不想动"]
SOMATIC_WORDS = ["疼", "痛", "没力气", "吃不下", "心慌", "头晕", "身体不舒服"]
HOPELESS_WORDS = ["没盼头", "绝望", "没希望", "没有出路", "熬不住", "熬不下去"]
GUILT_WORDS = ["拖累", "负担", "对不起", "累赘", "没用", "废物", "拖后腿"]
SOCIAL_WORDS = ["孤独", "孤单", "没人说话", "没人陪", "不出门", "一个人"]
COGNITION_WORDS = ["记不住", "记性", "注意力", "脑子", "忘事", "糊涂", "反应慢"]
APPETITE_WORDS = ["吃不下", "胃口", "没食欲", "体重", "不想吃"]
SELF_HARM_WORDS = ["想死", "自杀", "不想活", "活不下去", "结束生命", "伤害自己", "自残"]

FIRST_PERSON_WORDS = ["我", "我们", "咱", "自己", "我这人"]


def _count_hits(text: str, words: Iterable[str]) -> int:
    return sum(text.count(w) for w in words)


def extract_text_features(text: str) -> dict[str, float]:
    text = (text or "").strip()
    if not text:
        return {
            "char_count": 0,
            "sentence_count": 0,
            "negative_hits": 0,
            "positive_hits": 0,
            "negative_density": 0.0,
            "positive_density": 0.0,
            "valence": 0.0,
            "sleep_hits": 0,
            "interest_hits": 0,
            "somatic_hits": 0,
            "hopeless_hits": 0,
            "guilt_hits": 0,
            "social_hits": 0,
            "cognition_hits": 0,
            "appetite_hits": 0,
            "self_harm_hits": 0,
            "first_person_ratio": 0.0,
            "negation_ratio": 0.0,
            "hesitation_ratio": 0.0,
        }

    sentences = [s for s in re.split(r"[。！？!?；;\n]+", text) if s.strip()]
    chars = len(re.sub(r"\s+", "", text))
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]+", text)

    neg = _count_hits(text, NEGATIVE_WORDS)
    pos = _count_hits(text, POSITIVE_WORDS)
    first_person = _count_hits(text, FIRST_PERSON_WORDS)
    negations = _count_hits(text, ["不", "没", "没有", "别", "不要"])
    hesitations = _count_hits(text, ["嗯", "啊", "呃", "那个", "怎么说", "唉"])

    return {
        "char_count": float(chars),
        "sentence_count": float(len(sentences)),
        "negative_hits": float(neg),
        "positive_hits": float(pos),
        "negative_density": neg / max(1.0, float(chars)),
        "positive_density": pos / max(1.0, float(chars)),
        "valence": float(pos - neg),
        "sleep_hits": float(_count_hits(text, SLEEP_WORDS)),
        "interest_hits": float(_count_hits(text, INTEREST_WORDS)),
        "somatic_hits": float(_count_hits(text, SOMATIC_WORDS)),
        "hopeless_hits": float(_count_hits(text, HOPELESS_WORDS)),
        "guilt_hits": float(_count_hits(text, GUILT_WORDS)),
        "social_hits": float(_count_hits(text, SOCIAL_WORDS)),
        "cognition_hits": float(_count_hits(text, COGNITION_WORDS)),
        "appetite_hits": float(_count_hits(text, APPETITE_WORDS)),
        "self_harm_hits": float(_count_hits(text, SELF_HARM_WORDS)),
        "first_person_ratio": first_person / max(1.0, float(len(tokens))),
        "negation_ratio": negations / max(1.0, float(len(tokens))),
        "hesitation_ratio": hesitations / max(1.0, float(len(tokens))),
    }


def summarize_dialogue(transcript: list[dict[str, str]]) -> pd.DataFrame:
    """transcript: [{"role": "user"/"assistant", "text": "..."}]"""
    rows = []
    for i, item in enumerate(transcript):
        text = item.get("text", "")
        feats = extract_text_features(text)
        feats["turn"] = i
        feats["role"] = item.get("role", "unknown")
        rows.append(feats)
    return pd.DataFrame(rows)


class TransformerTextEncoder:
    """Optional wrapper for a local Chinese sentiment/embedding model."""

    def __init__(self, model_name: str = "uer/roberta-base-finetuned-jd-binary-chinese"):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise RuntimeError("transformers is not installed")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        import torch

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inputs).last_hidden_state.mean(dim=1)
        return out.tolist()
