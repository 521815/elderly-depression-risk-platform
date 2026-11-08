"""Dialogue-based risk assessment with optional LLM and heuristic fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Callable

import numpy as np

from ..config import OUTPUT_DIR
from ..features.text_features import summarize_dialogue
from ..scales import severity_level
from .fusion import load_weights, predict_with_weights, prepare_variant_weights


FUSION_WEIGHTS_JSON = OUTPUT_DIR / "fusion_models" / "fusion_weights.json"
CONTEXT_COMPONENT_JSON = OUTPUT_DIR / "context_component_learning.json"


LANGUAGE_CUE_RULES = [
    ("自杀意念", ["想死", "自杀", "不想活", "伤害自己", "自残", "活不下去"]),
    ("无望感", ["没盼头", "绝望", "没希望", "没有出路", "活着没意思"]),
    ("自责/负担", ["拖累", "负担", "对不起", "累赘", "没用"]),
    ("睡眠差", ["失眠", "睡不着", "睡不好", "半夜醒", "醒得早", "没精神"]),
    ("兴趣减退", ["没意思", "没兴趣", "提不起兴趣", "不想动", "不想出门"]),
    ("情绪低落", ["难过", "伤心", "低落", "烦", "难受", "哭", "压抑"]),
    ("躯体不适", ["疼", "痛", "乏力", "没力气", "吃不下", "头晕", "心慌", "胸闷"]),
    ("认知下降", ["记不住", "记性", "忘事", "糊涂", "反应慢"]),
    ("社交孤立", ["孤独", "孤单", "没人说话", "没人陪"]),
]


def _clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def _load_fusion_weights():
    try:
        return load_weights(FUSION_WEIGHTS_JSON)
    except Exception:
        return None


def openai_completer(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable[..., str]:
    """Create an OpenAI chat completion callable for DialogueAssessor."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package is not installed")

    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
    )

    def complete(messages: list[dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    return complete


class DialogueAssessor:
    """Combines dialogue, scale answers, demographics/comorbidity and expression signals."""

    def __init__(self, llm_complete: Callable[..., str] | None = None):
        self.llm_complete = llm_complete

    def assess(
        self,
        transcript: list[dict[str, str]],
        scale_answers: dict[str, list[float | int]] | None = None,
        demographics: dict | None = None,
        expression_summary: dict | None = None,
        scale_name: str = "cesd10",
        mpdd_risk: float | None = None,
        language_cues: list[dict | str] | None = None,
    ) -> dict:
        demographics = demographics or {}
        expression_summary = expression_summary or {}
        text_df = summarize_dialogue(transcript)
        cues = self._normalize_language_cues(language_cues)
        if not cues:
            cues = self._extract_language_cues(transcript)
        text_risk = self._text_risk(text_df, cues)
        scale_risk = self._scale_risk(scale_answers, scale_name)
        scale_score = self._raw_scale_score(scale_answers, scale_name)
        context_risk = self._context_risk(demographics)

        expression_risk = (
            _clamp(float(expression_summary.get("risk", 0.5))) if expression_summary else None
        )
        learned = _load_fusion_weights()
        variant = None
        if learned:
            variants = learned.get("variants", {})
            if mpdd_risk is not None and "with_mpdd" in variants:
                variant = "with_mpdd"
            elif expression_risk is not None and "no_mpdd" in variants:
                variant = "no_mpdd"

        if variant is not None:
            signals = {
                "text_risk": text_risk,
                "scale_risk": scale_risk,
                "context_risk": context_risk,
                "expression_risk": expression_risk if expression_risk is not None else 0.5,
            }
            if mpdd_risk is not None:
                signals["mpdd_risk"] = _clamp(float(mpdd_risk))
            weights = learned["variants"][variant]
            score = predict_with_weights(signals, prepare_variant_weights(weights))
            used_weights = dict(zip(weights["features"], weights["coefficients_vector"]))
            fusion_mode = "learned"
        else:
            # Literature-based prior weights; see docs/06_论文综述与对话Agent改良.md.
            weights = {"text": 0.2, "scale": 0.4, "context": 0.2, "expression": 0.2}
            if mpdd_risk is not None:
                weights = {
                    "text": 0.15,
                    "scale": 0.30,
                    "context": 0.20,
                    "expression": 0.10,
                    "mpdd": 0.25,
                }
            score = (
                weights["text"] * text_risk
                + weights["scale"] * scale_risk
                + weights["context"] * context_risk
            )
            if expression_risk is not None:
                score += weights["expression"] * expression_risk
            if mpdd_risk is not None:
                score += weights["mpdd"] * _clamp(float(mpdd_risk))
            used_weights = weights
            fusion_mode = "literature_prior"
        if expression_risk is None:
            used_weights = {k: v for k, v in used_weights.items() if k != "expression"}

        result = {
            "depression_probability": round(_clamp(score), 4),
            "risk_level": self._risk_level(score),
            "text_risk": round(text_risk, 4),
            "text_method": "rule_llm" if cues else "rule",
            "scale_risk": round(scale_risk, 4),
            "scale_score": scale_score,
            "context_risk": round(context_risk, 4),
            "context_method": (
                "learned_calibrated"
                if CONTEXT_COMPONENT_JSON.exists()
                else "heuristic_prior"
            ),
            "scale_level": severity_level(scale_score, scale_name),
            "expression_risk": round(expression_risk, 4) if expression_risk is not None else None,
            "mpdd_risk": round(_clamp(float(mpdd_risk)), 4) if mpdd_risk is not None else None,
            "fusion_mode": fusion_mode,
            "fusion_variant": variant or "literature_prior",
            "fusion_weights": used_weights,
            "language_cues": cues,
            "disclaimer": "仅用于科研风险筛查，不构成医学诊断。",
        }
        if self.llm_complete is not None:
            result["llm_summary"] = self._llm_summary(
                transcript, scale_risk, context_risk, expression_summary, result, mpdd_risk
            )
        return result

    @staticmethod
    def _normalize_language_cues(language_cues: list[dict | str] | None) -> list[dict]:
        result: list[dict] = []
        for cue in language_cues or []:
            if isinstance(cue, dict):
                name = str(cue.get("cue") or cue.get("label") or "")
                if name:
                    result.append(
                        {
                            "cue": name,
                            "confidence": float(cue.get("confidence", 1.0)),
                        }
                    )
            elif cue:
                result.append({"cue": str(cue), "confidence": 1.0})
        return result

    @staticmethod
    def _extract_language_cues(transcript: list[dict[str, str]]) -> list[dict]:
        text = " ".join(
            str(item.get("text", ""))
            for item in transcript or []
            if item.get("role") == "user"
        )
        cues: list[dict] = []
        for label, words in LANGUAGE_CUE_RULES:
            if any(word in text for word in words):
                confidence = 1.0 if label == "自杀意念" else 0.8
                cues.append({"cue": label, "confidence": confidence})
        return cues

    def _text_risk(self, text_df, language_cues: list[dict] | None = None) -> float:
        if text_df.empty:
            rule_risk = 0.5
        else:
            neg_density = float(text_df["negative_density"].mean())
            pos_density = float(text_df["positive_density"].mean())
            sleep = float(text_df["sleep_hits"].sum())
            interest = float(text_df["interest_hits"].sum())
            somatic = float(text_df["somatic_hits"].sum())
            hopeless = float(text_df["hopeless_hits"].sum())
            guilt = float(text_df["guilt_hits"].sum())
            social = float(text_df["social_hits"].sum())
            cognition = float(text_df["cognition_hits"].sum())
            appetite = float(text_df["appetite_hits"].sum())
            self_harm = float(text_df["self_harm_hits"].sum())
            base = 0.5 + (neg_density - pos_density) * 10
            cue_bonus = min(
                0.28,
                0.02 * sleep
                + 0.02 * interest
                + 0.01 * somatic
                + 0.05 * hopeless
                + 0.05 * guilt
                + 0.02 * social
                + 0.01 * cognition
                + 0.02 * appetite,
            )
            if self_harm > 0:
                cue_bonus += 0.25
            rule_risk = _clamp(base + cue_bonus)

        if not language_cues:
            return rule_risk

        cue_weights = {
            "自杀意念": 0.25,
            "无望感": 0.15,
            "自责/负担": 0.10,
            "情绪低落": 0.08,
            "睡眠差": 0.05,
            "兴趣减退": 0.05,
            "社交孤立": 0.05,
            "躯体不适": 0.03,
            "认知下降": 0.03,
        }
        cue_risk = 0.5
        for cue in language_cues:
            weight = cue_weights.get(cue.get("cue", ""), 0.0)
            confidence = _clamp(float(cue.get("confidence", 0.5)))
            cue_risk += weight * confidence
        cue_risk = _clamp(cue_risk, 0.0, 0.9)
        return _clamp(0.6 * rule_risk + 0.4 * cue_risk)

    def _scale_risk(self, scale_answers: dict | None, scale_name: str) -> float:
        if not scale_answers:
            return 0.5
        score = self._raw_scale_score(scale_answers, scale_name)
        if score is None:
            return 0.5
        max_scores = {"cesd10": 30, "phq9": 27, "gds15": 15, "hamd24": 76}
        return _clamp(score / max_scores.get(scale_name, 30) * 1.15)

    def _raw_scale_score(self, scale_answers: dict | None, scale_name: str):
        if not scale_answers:
            return None
        from ..scales import score_cesd10, score_gds15, score_hamd24, score_phq9

        items = list(scale_answers.get(scale_name, []))
        score_funcs = {
            "cesd10": score_cesd10,
            "phq9": score_phq9,
            "gds15": score_gds15,
            "hamd24": score_hamd24,
        }
        return score_funcs.get(scale_name, score_cesd10)(items)

    def _context_risk(self, demographics: dict) -> float:
        age = float(demographics.get("age", 70))
        gender = demographics.get("gender")
        chronic = int(demographics.get("comorbidity_count", 0) or 0)
        married = demographics.get("married")
        sleep = demographics.get("sleep_quality")
        base = 0.35
        if age >= 75:
            base += 0.05
        if gender == "female":
            base += 0.05
        if chronic >= 3:
            base += 0.12
        elif chronic >= 1:
            base += 0.06
        if married is False:
            base += 0.08
        if sleep is not None and int(sleep) >= 3:
            base += 0.05
        return self._calibrate_context(base)

    @staticmethod
    def _calibrate_context(heuristic: float) -> float:
        try:
            if not CONTEXT_COMPONENT_JSON.exists():
                return _clamp(heuristic)
            params = json.loads(CONTEXT_COMPONENT_JSON.read_text(encoding="utf-8"))
            intercept = float(params["intercept"])
            coefficient = float(params["coefficient"])
            logit = intercept + coefficient * heuristic
            return _clamp(1.0 / (1.0 + float(np.exp(-logit))))
        except Exception:
            return _clamp(heuristic)

    def _risk_level(self, score: float) -> str:
        if score >= 0.7:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    def _llm_summary(
        self, transcript, scale_risk, context_risk, expression_summary, result, mpdd_risk
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是老年抑郁风险筛查助手。请只依据给定信息输出不超过200字的评估摘要，"
                    "指出语言/表情/量表/共病中的风险线索，并以JSON输出："
                    '{"summary": "...", "risk_factors": ["..."], "probability": 0.0-1.0}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "transcript": transcript,
                        "scale_risk": scale_risk,
                        "context_risk": context_risk,
                        "expression": expression_summary,
                        "mpdd_risk": mpdd_risk,
                        "reference_probability": result["depression_probability"],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        raw = self.llm_complete(messages)
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try:
                return json.dumps(json.loads(m.group(0)), ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        return raw.strip()
