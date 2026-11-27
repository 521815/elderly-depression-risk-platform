"""Versioned runtime prompt templates for the dialogue agent."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_ROOT = PROJECT_ROOT / "prompts"

PROMPT_SPECS = {
    "interview": {
        "version": "interview_v3",
        "path": PROMPT_ROOT / "runtime" / "interview_v3.md",
    },
    "companion": {
        "version": "companion_v3",
        "path": PROMPT_ROOT / "runtime" / "companion_v3.md",
    },
}


def prompt_versions() -> dict[str, str]:
    return {
        mode: str(spec["version"])
        for mode, spec in PROMPT_SPECS.items()
    }


def load_prompt(
    mode: str,
    *,
    current_topic: str = "",
    completed_topics: str = "",
    planned_topic: str = "",
) -> tuple[str, str]:
    """Load and fill a runtime template without treating JSON braces as syntax."""

    spec = PROMPT_SPECS.get(mode)
    if spec is None:
        raise ValueError(f"未知对话模式：{mode}")
    path = Path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"Prompt 模板不存在：{path}")
    text = path.read_text(encoding="utf-8").strip()
    replacements = {
        "[[CURRENT_TOPIC]]": current_topic or "未设置",
        "[[COMPLETED_TOPICS]]": completed_topics or "无",
        "[[PLANNED_TOPIC]]": planned_topic or current_topic or "未设置",
    }
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    if "[[" in text:
        raise ValueError(f"Prompt 模板存在未替换变量：{path}")
    return text, str(spec["version"])
