"""Small negation helpers for rule-based clinical language cues."""

from __future__ import annotations


NEGATION_SUFFIXES = [
    "没有",
    "并没有",
    "没",
    "并不",
    "并未",
    "从未",
    "从来没",
    "不是",
    "不再",
    "不会",
    "不怎么",
    "不太",
    "没有那么",
    "没有这么",
    "没那么",
    "没这么",
]

CONTRAST_MARKERS = [
    "但是",
    "不过",
    "可是",
    "然而",
    "只是",
    "就是",
    "问题是",
    "主要问题",
    "主要是",
    "而是",
    "但",
]


def is_negated(text: str, start: int, window: int = 6) -> bool:
    prefix = str(text or "")[max(0, start - window) : start].rstrip()
    if any(prefix.endswith(token) for token in NEGATION_SUFFIXES):
        return True
    return prefix.endswith("不")


def has_unnegated_keyword(text: str, keyword: str) -> bool:
    value = str(text or "")
    if not keyword:
        return False
    start = value.find(keyword)
    while start >= 0:
        if not is_negated(value, start):
            return True
        start = value.find(keyword, start + len(keyword))
    return False


def has_any_unnegated_keyword(text: str, keywords: list[str]) -> bool:
    return any(has_unnegated_keyword(text, keyword) for keyword in keywords)


def focus_clause(text: str) -> str:
    value = str(text or "").strip()
    positions = []
    for marker in sorted(CONTRAST_MARKERS, key=len, reverse=True):
        start = value.find(marker)
        if start > 0:
            positions.append((start, marker))
    if positions:
        start, marker = min(positions, key=lambda item: (item[0], -len(item[1])))
        tail = value[start + len(marker) :].strip("，,。；; ")
        if tail:
            return tail
    return value
