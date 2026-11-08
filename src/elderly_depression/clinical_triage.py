"""Rule-based clinical red-flag triage for dialogue safety."""

from __future__ import annotations

from .negation import has_any_unnegated_keyword


EMERGENCY_MESSAGE = (
    "您描述的情况可能需要紧急医疗处理。请立即联系身边家人并拨打 120；"
    "如果存在自伤风险，请同时拨打 110 或心理援助热线 12356。不要一个人等待。"
)

URGENT_MESSAGE = (
    "这些情况建议尽快就医，最好今天联系家人或医生评估。"
    "如果症状加重，请立即拨打 120。"
)


def assess_red_flags(text: str) -> dict | None:
    compact = str(text or "").replace(" ", "")
    if not compact:
        return None
    chest = has_any_unnegated_keyword(compact, ["胸痛", "胸口疼", "胸口压", "胸闷"])
    breathing = has_any_unnegated_keyword(
        compact, ["喘不上", "呼吸困难", "冒汗", "出冷汗", "放射", "左肩"]
    )
    if chest and breathing:
        return {
            "level": "emergency",
            "reason": "胸痛伴呼吸困难或出汗",
            "message": EMERGENCY_MESSAGE,
        }
    stroke_signs = has_any_unnegated_keyword(
        compact,
        ["嘴歪", "口角歪", "半边无力", "一侧无力", "说话不清", "突然意识", "昏倒"],
    )
    if stroke_signs:
        return {
            "level": "emergency",
            "reason": "疑似急性神经系统症状",
            "message": EMERGENCY_MESSAGE,
        }
    fall = has_any_unnegated_keyword(compact, ["摔倒", "跌倒"])
    injury = has_any_unnegated_keyword(
        compact, ["头破", "头部受伤", "出血", "昏迷", "意识不清", "动不了", "骨折"]
    )
    if fall and injury:
        return {
            "level": "emergency",
            "reason": "跌倒伴明显外伤或意识异常",
            "message": EMERGENCY_MESSAGE,
        }
    suicide_intent = has_any_unnegated_keyword(
        compact, ["想死", "自杀", "不想活", "活着没意思", "结束生命"]
    )
    suicide_plan = has_any_unnegated_keyword(
        compact, ["计划", "遗书", "准备了", "刀", "绳子", "药都吃了", "已经吃了", "马上结束"]
    )
    if suicide_intent and suicide_plan:
        return {
            "level": "emergency",
            "reason": "存在自伤想法和计划线索",
            "message": EMERGENCY_MESSAGE,
        }
    urgent_terms = [
        "突然看不清",
        "突然听不见",
        "剧烈头痛",
        "持续呕吐",
        "不能喝水",
        "尿很少",
        "意识模糊",
        "反复跌倒",
        "站不稳加重",
        "胸口闷",
        "心慌",
    ]
    if has_any_unnegated_keyword(compact, urgent_terms):
        return {
            "level": "urgent",
            "reason": "存在需要尽快评估的症状",
            "message": URGENT_MESSAGE,
        }
    return None
