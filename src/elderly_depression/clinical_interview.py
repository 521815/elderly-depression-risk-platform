"""Structured, non-diagnostic clinical interview findings."""

from __future__ import annotations

import re


DURATION_PATTERN = re.compile(
    r"(半[个多]?月|半[个多]?年|一两[天周月年]|两三[天周月年]|"
    r"\d+个?[天周月年]|[一二两三四五六七八九十]+个?[天周月年])"
)

FREQUENCY_TERMS = [
    "偶尔",
    "有时候",
    "有时",
    "经常",
    "大多时候",
    "多数天",
    "几乎每天",
    "每天",
    "天天",
    "一周两三次",
    "一周三次",
]

IMPACT_TERMS = [
    "影响",
    "做不了",
    "干不了",
    "起不来",
    "下不了床",
    "不想出门",
    "不出门",
    "吃不下",
    "没胃口",
    "睡不着",
    "睡不好",
    "没精神",
    "需要帮忙",
    "没法工作",
    "无法工作",
]

SLEEP_TERMS = ["睡不着", "睡不好", "失眠", "半夜醒", "早醒", "睡不踏实"]
APPETITE_TERMS = [
    "吃不下",
    "没胃口",
    "不想吃",
    "体重下降",
    "吃得少",
    "影响吃饭",
]
MEDICATION_TERMS = ["药", "漏服", "停药", "副作用", "吃药"]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _first_present(text: str, terms: list[str]) -> str:
    return next((term for term in terms if term in text), "")


def update_clinical_findings(findings: dict, text: str) -> dict:
    """Update only explicitly stated findings; never infer a diagnosis."""

    compact = _compact(text)
    result = dict(findings or {})
    duration = DURATION_PATTERN.search(compact)
    if duration:
        result["duration"] = duration.group(1)
    frequency = _first_present(compact, FREQUENCY_TERMS)
    if frequency:
        result["frequency"] = frequency
    if any(
        phrase in compact
        for phrase in ["不影响", "还能正常", "没影响", "照常", "可以正常"]
    ):
        result["functional_impact"] = "无"
    elif any(term in compact for term in IMPACT_TERMS):
        result["functional_impact"] = "有"
    if any(term in compact for term in SLEEP_TERMS):
        result["sleep"] = "受影响"
    elif any(phrase in compact for phrase in ["睡得好", "睡眠正常", "睡得踏实"]):
        result["sleep"] = "正常"
    if any(term in compact for term in APPETITE_TERMS):
        result["appetite"] = "受影响"
    elif any(phrase in compact for phrase in ["吃得下", "胃口正常", "食欲正常"]):
        result["appetite"] = "正常"
    if any(term in compact for term in MEDICATION_TERMS):
        result["medication_mentioned"] = "是"
    return result


def build_clinical_summary(findings: dict) -> str:
    labels = [
        ("frequency", "频率"),
        ("duration", "持续"),
        ("functional_impact", "功能影响"),
        ("sleep", "睡眠"),
        ("exercise_end", "健身结束"),
        ("sleep_time", "上床时间"),
        ("appetite", "食欲"),
    ]
    parts = [
        f"{label}：{findings[key]}"
        for key, label in labels
        if findings.get(key)
    ]
    return "；".join(parts)


def meets_two_week_threshold(duration: str) -> bool:
    value = str(duration or "")
    if any(
        marker in value
        for marker in ["两周", "2周", "半个月", "一个月", "1个月", "两个月", "半年", "年"]
    ):
        return True
    match = re.search(r"([三四五六七八九]|\d+)(周|个月|月)", value)
    if not match:
        return False
    number_text, unit = match.groups()
    if unit == "周":
        numbers = {
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        number = numbers.get(number_text, int(number_text) if number_text.isdigit() else 0)
        return number >= 2
    return True
