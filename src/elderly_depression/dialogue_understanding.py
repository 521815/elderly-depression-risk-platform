"""Clause-level understanding for natural multi-part replies.

The dialogue agent receives many answers that contain a positive background and
an unresolved concern in the same sentence.  Treating the whole sentence as one
polarity often produces a positive reply and misses the actual request.  This
module keeps the distinction explicit without depending on an external model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .negation import has_any_unnegated_keyword


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

REQUEST_TERMS = [
    "想调整",
    "想调",
    "想改善",
    "想解决",
    "想改变",
    "希望调整",
    "希望改善",
    "帮我",
    "怎么办",
    "怎么调整",
    "怎么改善",
    "如何调整",
    "如何改善",
    "困难",
    "难调",
    "做不到",
]

POSITIVE_TERMS = [
    "还不错",
    "还可以",
    "还行",
    "还好",
    "挺好",
    "很好",
    "不错",
    "正常",
    "稳定",
    "有人陪",
    "有人说话",
    "有精神",
    "睡得着",
    "吃得下",
    "基本规律",
]

ISSUE_TERMS = [
    "睡不着",
    "睡不好",
    "失眠",
    "半夜醒",
    "老醒",
    "容易醒",
    "醒太早",
    "早醒",
    "入睡难",
    "入睡困难",
    "白天困",
    "没精神",
    "睡晚",
    "睡得晚",
    "睡得比较晚",
    "睡得很晚",
    "睡得挺晚",
    "睡得有点晚",
    "睡得偏晚",
    "晚睡",
    "熬夜",
    "熬到",
    "睡迟",
    "看晚了",
    "聊晚了",
    "作息",
    "难调",
    "困难",
    "没意思",
    "没兴趣",
    "不想动",
    "不想出门",
    "难过",
    "伤心",
    "低落",
    "想哭",
    "心里堵",
    "烦",
    "孤单",
    "孤独",
    "没人陪",
    "没人说话",
    "疼",
    "痛",
    "不舒服",
    "没力气",
    "乏力",
    "头晕",
    "心慌",
    "记不住",
    "忘事",
    "注意力",
    "拖累",
    "负担",
    "没盼头",
    "绝望",
    "忘记",
    "漏服",
    "站不稳",
    "跌倒",
    "吃不下",
    "没胃口",
    "体重下降",
    "看不清",
    "听不清",
]

CLAUSE_PATTERN = re.compile(
    r"(?:但是|不过|可是|然而|只是|就是|问题是|主要问题|主要是|而是|但)"
    r"|[，,。；;！!？?\n]+"
)


@dataclass(frozen=True)
class UtteranceAnalysis:
    """Structured interpretation of one user turn."""

    text: str
    focus_text: str
    background_text: str
    positive_context: str
    clauses: tuple[str, ...]
    is_correction: bool
    requests_help: bool
    focus_reason: str

    @property
    def has_focused_issue(self) -> bool:
        return self.focus_reason != "whole_text" or self.requests_help

    def as_dict(self) -> dict:
        return {
            "focus": self.focus_text,
            "background": self.background_text,
            "positive_context": self.positive_context,
            "requests_help": self.requests_help,
            "focus_reason": self.focus_reason,
        }


def _clean(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip()


def _split_clauses(text: str) -> list[str]:
    parts = CLAUSE_PATTERN.split(text)
    return [part.strip("，,。；;！!？? ") for part in parts if part.strip("，,。；;！!？? ")]


def _issue_score(text: str) -> int:
    score = 0
    for term in ISSUE_TERMS:
        if has_any_unnegated_keyword(text, [term]):
            score += 2 if term not in {"困难", "烦"} else 1
    for term in REQUEST_TERMS:
        if term in text:
            score += 2
    return score


def _positive_score(text: str) -> int:
    return sum(1 for term in POSITIVE_TERMS if term in text)


def _first_contrast_marker(text: str) -> tuple[int, str] | None:
    positions = []
    for marker in sorted(CONTRAST_MARKERS, key=len, reverse=True):
        start = text.find(marker)
        if start > 0 and not (
            marker in {"只是", "就是"} and text[start - 1] == "不"
        ):
            positions.append((start, marker))
    if not positions:
        return None
    return min(positions, key=lambda item: (item[0], -len(item[1])))


def analyze_utterance(text: str) -> UtteranceAnalysis:
    """Separate a reply's background from the part the speaker wants addressed."""

    value = _clean(text)
    clauses = _split_clauses(value)
    if len(clauses) <= 1:
        clauses = [value] if value else []

    is_correction = bool(
        re.search(r"(?:^|[，,；;])不是", value)
        or re.search(r"(?:不是|并非).{0,18}(?:而是|是)", value)
    )
    requests_help = any(term in value for term in REQUEST_TERMS)
    explicit_focus_match = re.search(
        r"(?:我说的是|我说是|我指的是|我的意思是)([^，,。；;]+)",
        value,
    )
    explicit_focus = (
        explicit_focus_match.group(1).strip()
        if explicit_focus_match
        else ""
    )

    focus_reason = "whole_text"
    focus_index = 0
    background_text = ""

    marker = _first_contrast_marker(value)
    if explicit_focus:
        focus_text = explicit_focus
        focus_reason = "explicit_correction"
        background_text = value.split("我说", 1)[0].strip("，,。；; ")
    elif marker is not None:
        marker_index, marker_text = marker
        tail = value[marker_index + len(marker_text) :].strip("，,。；; ")
        if tail:
            background_text = value[:marker_index].strip("，,。；; ")
            if marker_text == "而是" or re.match(r"^是(?!不是|否)", tail):
                tail = re.sub(r"^(?:而是|是)", "", tail).strip()
            focus_text = tail
            focus_reason = "contrast_marker"
            if clauses:
                for index, clause in enumerate(clauses):
                    if clause and clause in tail:
                        focus_index = index
                        break
        else:
            focus_text = value
    elif len(clauses) > 1:
        scores = [_issue_score(clause) for clause in clauses]
        best_index = max(range(len(clauses)), key=lambda index: (scores[index], index))
        has_positive_background = any(
            _positive_score(clause) and _issue_score(clause) == 0
            for clause in clauses
        )
        if scores[best_index] > 0 and (is_correction or has_positive_background):
            focus_index = best_index
            focus_text = clauses[focus_index]
            background_text = "，".join(clauses[:focus_index])
            focus_reason = "issue_clause"
        else:
            focus_text = value
            background_text = ""
    else:
        focus_text = value

    if is_correction:
        focus_text = re.sub(r"^(?:而是|是)(?!不是|否)", "", focus_text).strip()

    if clauses and not background_text:
        background_text = "，".join(clauses[:focus_index])

    positive_parts = []
    for clause in clauses:
        if clause == focus_text or clause in focus_text:
            continue
        if _positive_score(clause) and _issue_score(clause) == 0:
            positive_parts.append(clause)
    positive_context = "，".join(positive_parts)

    return UtteranceAnalysis(
        text=value,
        focus_text=focus_text or value,
        background_text=background_text,
        positive_context=positive_context,
        clauses=tuple(clauses),
        is_correction=is_correction,
        requests_help=requests_help,
        focus_reason=focus_reason,
    )
