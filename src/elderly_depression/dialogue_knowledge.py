"""Small locally curated knowledge base for retrieval-guided dialogue."""

from __future__ import annotations

import re


GUIDANCE_ITEMS = [
    {
        "id": "mood-connection",
        "topic": "mood",
        "levels": ["mild", "moderate"],
        "title": "情绪支持与现实连接",
        "keywords": ["难过", "心里堵", "想哭", "低落", "情绪"],
        "validation": "持续或反复的情绪不适值得被认真对待。",
        "action": "鼓励联系一位可信的人，并把感受和睡眠、食欲变化简单记下来。",
        "escalation": "持续两周以上或明显影响日常生活时，建议接受专业评估。",
    },
    {
        "id": "mood-urgent",
        "topic": "mood",
        "levels": ["severe"],
        "title": "情绪困扰升级",
        "keywords": ["每天", "受不了", "越来越重", "绝望", "想哭"],
        "validation": "长期被强烈情绪困住会非常消耗人。",
        "action": "尽快联系家人、医生或心理专业人员，不要求老人独自处理。",
        "escalation": "出现自伤想法或不安全行为时，立即联系急救或心理援助热线。",
    },
    {
        "id": "sleep-routine",
        "topic": "sleep",
        "levels": ["mild", "moderate"],
        "title": "睡眠作息支持",
        "keywords": ["睡不着", "睡不好", "半夜醒", "早醒", "作息", "睡晚"],
        "validation": "睡眠变化会影响白天精力和情绪。",
        "action": "固定起床时间、晨间见光、午睡不超过半小时，逐步提前上床。",
        "escalation": "持续两周以上或白天功能明显受影响时，建议睡眠或精神心理评估。",
    },
    {
        "id": "sleep-exercise-cortisol",
        "topic": "sleep",
        "levels": ["mild", "moderate", "uncertain"],
        "title": "运动、皮质醇与睡眠",
        "keywords": ["健身", "锻炼", "运动", "皮质醇", "入睡晚", "入睡推迟"],
        "validation": "运动后皮质醇短时变化属于正常生理应激，不能只凭主观感受判断皮质醇异常。",
        "action": "记录健身结束时间、入睡时间和第二天精神；可尝试避免高强度运动离睡眠太近，并减少晚间咖啡因。",
        "escalation": "持续失眠、心悸或明显疲劳时，应请医生评估甲状腺、贫血、睡眠呼吸问题和药物影响。",
    },
    {
        "id": "sleep-apnea",
        "topic": "sleep",
        "levels": ["moderate", "severe"],
        "title": "睡眠呼吸问题线索",
        "keywords": ["打鼾", "憋醒", "呼吸暂停", "白天嗜睡", "晨起头痛"],
        "validation": "明显打鼾、憋醒或白天异常嗜睡需要单独考虑睡眠呼吸问题。",
        "action": "记录是否打鼾、被憋醒、晨起头痛和白天嗜睡，并告知医生。",
        "escalation": "怀疑睡眠呼吸暂停时，建议到睡眠专科进行评估。",
    },
    {
        "id": "interest-activation",
        "topic": "interest",
        "levels": ["mild", "moderate"],
        "title": "兴趣与行为激活",
        "keywords": ["没兴趣", "不想做", "没意思", "提不起劲", "不愿意"],
        "validation": "以前喜欢的事变得没劲，会让人失落，也可能让生活范围变小。",
        "action": "从十分钟、低负担的小活动开始，不要求一次恢复到以前的状态。",
        "escalation": "兴趣下降持续两周以上并影响日常活动时，建议专业评估。",
    },
    {
        "id": "social-connection",
        "topic": "social",
        "levels": ["mild", "moderate"],
        "title": "减少孤独与现实连接",
        "keywords": ["孤单", "孤独", "没人说话", "没人陪", "一个人"],
        "validation": "有人陪仍觉得孤单，说明缺少的可能是能说心里话的交流。",
        "action": "先联系一个最信任的人，或安排一次简短见面、通话、楼下散步。",
        "escalation": "长期孤立、完全无人可联系时，建议联系社区服务或专业机构。",
    },
    {
        "id": "cognition-support",
        "topic": "cognition",
        "levels": ["mild", "moderate"],
        "title": "记忆与注意力支持",
        "keywords": ["记不住", "忘事", "注意力", "找不到东西", "反应慢"],
        "validation": "记忆和注意力变化会让人着急，也会影响做事信心。",
        "action": "把钥匙、药、电话放在固定位置，重要事项写下来，并让家人了解变化。",
        "escalation": "突然加重、迷路或明显影响日常生活时，应尽快就医评估。",
    },
    {
        "id": "cognition-acute-change",
        "topic": "cognition",
        "levels": ["moderate", "severe"],
        "title": "突然意识或认知变化",
        "keywords": ["突然糊涂", "意识不清", "急性变化", "发热", "脱水"],
        "validation": "突然出现的意识或认知变化不同于缓慢记忆变化，需要优先排除急症。",
        "action": "尽快联系医生或急救，避免独处，并准备近期用药和感染、脱水、跌倒等信息。",
        "escalation": "伴发热、意识异常、单侧无力或说话不清时立即拨打 120。",
    },
    {
        "id": "negative-cognition-safety",
        "topic": "negative_cognition",
        "levels": ["moderate", "severe"],
        "title": "自责无望与安全支持",
        "keywords": ["拖累", "负担", "没盼头", "绝望", "没用"],
        "validation": "强烈的自责和无望会把人与外界隔开。",
        "action": "先联系可信家人、医生或专业人员，不独自承担这些想法。",
        "escalation": "出现自伤计划、准备行为或无法保证安全时，立即联系急救并保证有人陪伴。",
    },
    {
        "id": "medication-adherence",
        "topic": "medication",
        "levels": ["mild", "moderate", "severe"],
        "title": "安全用药支持",
        "keywords": ["忘记", "漏服", "停药", "换药", "副作用", "头晕"],
        "validation": "漏服或不良反应会影响治疗效果，也可能增加风险。",
        "action": "使用分格药盒或提醒，记录漏服和不适；不要自行加倍或停药。",
        "escalation": "连续漏服、明显不良反应或需要调整剂量时，联系开药医生或药师。",
    },
    {
        "id": "medication-polypharmacy",
        "topic": "medication",
        "levels": ["mild", "moderate", "severe"],
        "title": "多重用药与头晕跌倒",
        "keywords": ["多种药", "头晕", "站起来晕", "跌倒", "镇静药", "降压药"],
        "validation": "多种药物、体位性头晕和跌倒需要结合用药时间一起评估。",
        "action": "整理药名、剂量、服用时间和症状，不要自行停药。",
        "escalation": "晕厥、反复跌倒或意识变化时尽快联系医生。",
    },
    {
        "id": "falls-prevention",
        "topic": "falls",
        "levels": ["mild", "moderate", "severe"],
        "title": "跌倒预防",
        "keywords": ["站不稳", "跌倒", "摔倒", "头晕", "走路不稳"],
        "validation": "站不稳和担心跌倒会限制活动，也会增加受伤风险。",
        "action": "增加照明、清理障碍、使用防滑鞋和扶手，起身转身放慢。",
        "escalation": "跌倒伴头部受伤、意识异常或无法活动时，立即呼叫急救。",
    },
    {
        "id": "general-hypertension",
        "topic": "general_medical",
        "levels": ["mild", "moderate", "severe"],
        "title": "血压异常线索",
        "keywords": ["高血压", "血压高", "后脑勺胀", "血压计"],
        "validation": "血压波动需要结合测量值和当时症状判断。",
        "action": "安静休息后按规范复测并记录时间和数值，不要自行加药。",
        "escalation": "血压明显异常并伴胸痛、呼吸困难、意识或肢体变化时立即拨打 120。",
        "question": "最近一次测得的血压大概是多少？",
    },
    {
        "id": "general-diabetes",
        "topic": "general_medical",
        "levels": ["mild", "moderate", "severe"],
        "title": "血糖异常线索",
        "keywords": ["糖尿病", "血糖", "口渴", "尿多", "体重下降"],
        "validation": "明显口渴、尿多和体重下降需要关注血糖情况。",
        "action": "记录饮水、排尿、体重和近期血糖值，并联系医生评估。",
        "escalation": "伴意识改变、呼吸异常或无法进食饮水时立即就医。",
        "question": "最近有没有测过血糖，数值大概多少？",
    },
    {
        "id": "general-respiratory",
        "topic": "general_medical",
        "levels": ["mild", "moderate", "severe"],
        "title": "呼吸道感染线索",
        "keywords": ["咳嗽", "发热", "痰", "喘", "感冒"],
        "validation": "老年人呼吸道感染可能表现不典型，需要关注精神、进食和呼吸变化。",
        "action": "记录体温、呼吸情况和饮水量；持续发热或气促应联系医生。",
        "escalation": "呼吸困难、口唇发紫、意识变化或血氧明显下降时立即拨打 120。",
        "question": "咳嗽和发热大概持续几天了？",
    },
    {
        "id": "general-gastrointestinal",
        "topic": "general_medical",
        "levels": ["mild", "moderate", "severe"],
        "title": "胃肠道症状线索",
        "keywords": ["胃痛", "腹痛", "反酸", "腹泻", "便秘", "恶心", "呕吐"],
        "validation": "胃肠道症状需要结合饮食、排便、发热和体重变化判断。",
        "action": "记录疼痛位置、排便次数和是否伴发热呕吐，注意补充水分。",
        "escalation": "剧烈腹痛、便血、黑便、持续呕吐或明显脱水时尽快就医。",
        "question": "最明显的是胃痛、腹泻、便秘，还是反酸恶心？",
    },
    {
        "id": "general-urinary",
        "topic": "general_medical",
        "levels": ["mild", "moderate", "severe"],
        "title": "泌尿系统症状线索",
        "keywords": ["尿频", "尿痛", "尿急", "血尿", "尿不出"],
        "validation": "尿频、尿痛和排尿困难在老年人中需要排查感染或排尿问题。",
        "action": "记录排尿次数、疼痛、发热和饮水情况，不要憋尿。",
        "escalation": "完全尿不出、血尿伴血块、高热或腰背剧痛时尽快就医。",
        "question": "有没有尿痛、发热，或者突然尿不出来？",
    },
    {
        "id": "general-pain-arthritis",
        "topic": "general_medical",
        "levels": ["mild", "moderate", "severe"],
        "title": "慢性疼痛与关节症状",
        "keywords": ["关节痛", "腰痛", "膝盖痛", "背痛", "麻木"],
        "validation": "持续疼痛会影响睡眠、活动和情绪，也需要评估功能和安全。",
        "action": "记录疼痛部位、诱发动作和持续时间，活动时使用扶手并避免硬撑。",
        "escalation": "突发剧痛、肢体无力麻木、跌倒受伤或大小便异常时尽快就医。",
        "question": "疼痛最影响走路、睡觉，还是穿衣洗澡？",
    },
    {
        "id": "general-thyroid-anemia",
        "topic": "general_medical",
        "levels": ["mild", "moderate", "severe"],
        "title": "甲状腺与贫血相关线索",
        "keywords": ["怕冷", "消瘦", "脸色白", "心慌", "乏力", "甲状腺", "贫血"],
        "validation": "怕冷、乏力、心慌、脸色苍白或体重变化可能与甲状腺、贫血等多种原因有关。",
        "action": "把症状持续时间和近期检查结果整理好，交给医生结合化验判断。",
        "escalation": "晕厥、明显气促、胸痛或持续心悸时及时就医。",
        "question": "最近有没有查过血常规、甲状腺或血糖？",
    },
]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def retrieve_guidance(
    topic: str,
    text: str,
    severity: str = "moderate",
    max_results: int = 2,
) -> list[dict]:
    """Return locally curated guidance ranked by topic and lexical overlap."""

    compact = _compact(text)
    scored = []
    for item in GUIDANCE_ITEMS:
        score = 0
        if item["topic"] == topic:
            score += 4
        elif any(keyword in compact for keyword in item["keywords"]):
            score += 2
        if severity in item["levels"]:
            score += 2
        score += sum(
            1 for keyword in item["keywords"] if keyword and keyword in compact
        )
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [item for _, item in scored[:max_results]]


def format_guidance_for_prompt(items: list[dict]) -> str:
    if not items:
        return ""
    parts = []
    for item in items:
        parts.append(
            f"{item['title']}：{item['validation']}"
            f"{item['action']} {item['escalation']}"
        )
    return " ".join(parts)


def find_general_medical_guidance(
    text: str,
    max_results: int = 1,
) -> list[dict]:
    compact = _compact(text)
    scored = []
    for item in GUIDANCE_ITEMS:
        if item["topic"] != "general_medical":
            continue
        score = sum(
            1 for keyword in item["keywords"] if keyword and keyword in compact
        )
        if score >= 2 or (
            score >= 1
            and any(
                marker in compact
                for marker in ["最近", "总是", "经常", "两天", "几天", "严重"]
            )
        ):
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [item for _, item in scored[:max_results]]
