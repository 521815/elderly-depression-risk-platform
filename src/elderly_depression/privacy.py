"""Privacy-minimization helpers for local storage and third-party model calls."""

from __future__ import annotations

import re


PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
NAME_PATTERN = re.compile(
    r"(我叫|我的名字是)([\u4e00-\u9fa5A-Za-z0-9]{1,12})"
)
ADDRESS_PATTERN = re.compile(
    r"((?:住在|家住))([^，。；\s]{2,24})"
)


def redact_text(text: str) -> str:
    value = str(text or "")
    value = PHONE_PATTERN.sub("[手机号已隐藏]", value)
    value = ID_PATTERN.sub("[证件号已隐藏]", value)
    value = EMAIL_PATTERN.sub("[邮箱已隐藏]", value)
    value = NAME_PATTERN.sub(r"\1[姓名已隐藏]", value)
    value = ADDRESS_PATTERN.sub(r"\1[地址已隐藏]", value)
    return value


def redact_known_values(text: str, values: list[str]) -> str:
    cleaned = str(text or "")
    for value in sorted(
        {str(item).strip() for item in values if str(item).strip()},
        key=len,
        reverse=True,
    ):
        cleaned = cleaned.replace(value, "[已隐藏]")
    return cleaned


def redact_messages(
    messages: list[dict[str, str]],
    known_values: list[str] | None = None,
) -> list[dict[str, str]]:
    cleaned = []
    for message in messages or []:
        text = redact_text(str(message.get("text") or ""))
        text = redact_known_values(text, known_values or [])
        cleaned.append(
            {
                **message,
                "text": text,
            }
        )
    return cleaned


def memory_for_cloud(memory: dict[str, str]) -> dict[str, str]:
    blocked = {"姓名", "居住地"}
    return {
        str(key): redact_text(str(value))
        for key, value in (memory or {}).items()
        if key not in blocked
    }
