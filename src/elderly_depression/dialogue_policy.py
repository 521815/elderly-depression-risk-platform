"""Question-answer policy matrix for humanized dialogue responses."""

from __future__ import annotations

import hashlib

from .negation import has_any_unnegated_keyword


ANSWER_ARCHETYPES = [
    "positive",
    "mild",
    "moderate",
    "severe",
    "uncertain",
    "avoidant",
]


HUMAN_OPENERS = {
    "positive": ["那就好。", "听起来还不错。", "这件事还能让您有点期待，很好。"],
    "mild": ["偶尔出现也会让人不舒服。", "有一点也值得留意。", "这种情况慢慢说就好。"],
    "moderate": [
        "听起来这段时间确实不太好受。",
        "这已经影响到日常生活了。",
        "这种反复出现的情况，会消耗不少精力。",
    ],
    "severe": [
        "这种情况听起来已经很辛苦了。",
        "这不是小事，我们慢慢梳理。",
        "这种感受一直压着您，我先陪您把眼前最要紧的一步理清。",
    ],
    "uncertain": ["一时说不清也很正常。", "没关系，我们可以换个角度想。", "不用急着给准确答案。"],
    "avoidant": ["没关系，不想聊这个也可以。", "谢谢您告诉我您的边界。", "不勉强，我们换个轻松点的。"],
}


TOPIC_RESPONSES = {
    "sleep": {
        "positive": "睡眠稳一些，白天也更有精神。最近白天精神怎么样？",
        "mild": "偶尔醒会让人觉得白天有点累。大概一周会有几次？",
        "moderate": "经常睡不好，身体和心情都会受影响。更明显的是入睡难，还是半夜醒？",
        "severe": "睡不好到这种程度，确实很熬人。这种情况持续多久了，白天能补觉吗？",
        "uncertain": "睡眠有时不太好判断。大概一周有几天觉得没睡够？",
        "avoidant": "那我们不聊睡眠了。最近有没有一件让您稍微放松一点的小事？",
    },
    "interest": {
        "positive": "还有喜欢做的事，生活就有一点亮光。最近做这件事时心情怎么样？",
        "mild": "偶尔提不起兴趣，也会让人觉得少了点劲头。以前喜欢做什么？",
        "moderate": "最近很多事都不想做，听起来挺让人难受的。最明显的是哪一类活动？",
        "severe": "如果大部分事都不想做了，这种变化需要认真看看。大概持续多久了？",
        "uncertain": "兴趣的变化有时候不太好说清。有没有哪件事比以前少做了？",
        "avoidant": "没关系，这个话题可以先放下。我们聊聊今天让您觉得舒服的一件事。",
    },
    "mood": {
        "positive": "心情平稳一些很重要。最近有没有什么让您觉得轻松的时刻？",
        "mild": "偶尔心里不舒服，也会影响一天的状态。这种感觉通常什么时候出现？",
        "moderate": "最近经常难受，确实会让人很累。这种心情大概持续多久了？",
        "severe": "如果这种难受持续得很久，先别一个人扛着。有没有能马上联系到的家人或医生？",
        "uncertain": "情绪有时很难用一句话说清。最近有没有想哭或者心里堵的时候？",
        "avoidant": "不想说也没关系。我们可以先聊点别的，您今天想从哪里说起？",
    },
    "somatic": {
        "positive": "身体舒服一点，人就更有精神。最近有没有哪里还会隐隐不舒服？",
        "mild": "偶尔疼痛或乏力也会影响心情。主要是哪个部位不舒服？",
        "moderate": "身体不适经常出现，确实会让人烦。它有没有影响您走路、睡觉或吃饭？",
        "severe": "身体不舒服到这种程度，需要认真处理。最近有没有看过医生或按时吃药？",
        "uncertain": "身体不舒服有时很难说清。有没有一个最明显的地方？",
        "avoidant": "好，我们先不聊身体。换个轻松点的话题，最近睡得还可以吗？",
    },
    "cognition": {
        "positive": "记性和注意力还算稳定，这很好。最近有没有偶尔忘事的时候？",
        "mild": "偶尔忘事会让人有点担心。是最近才出现，还是已经有一阵子了？",
        "moderate": "经常记不住或注意力不集中，确实会影响生活。做事情时最明显的是什么？",
        "severe": "记性和注意力下降得明显，家里人有没有注意到？这种情况持续多久了？",
        "uncertain": "记忆变化有时不太好判断。和以前比，最近忘事是不是更多了？",
        "avoidant": "不想聊记性也没关系。我们换个话题，最近有没有出去走走？",
    },
    "social": {
        "positive": "身边有人说话是很好的支持。最近和谁聊得比较多？",
        "mild": "偶尔觉得孤单，也会让人心里空落落的。最近有没有想找人说话的时候？",
        "moderate": "经常一个人待着，孤单感会更明显。最近一周和谁见过面或通过电话？",
        "severe": "如果长时间缺少交流，这种孤单会很重。遇到急事时有没有能联系的人？",
        "uncertain": "社交这件事有时不好说清。最近有没有参加过什么活动？",
        "avoidant": "好，我们不勉强聊这个。最近有没有一件让您觉得温暖的小事？",
    },
    "negative_cognition": {
        "positive": "您还能看到生活里的价值，这很重要。最近有没有让您觉得自己有用的时候？",
        "mild": "偶尔觉得自己没用或拖累家人，这种感受也会压人。最近一次这样想是什么时候？",
        "moderate": "经常觉得自己是负担，听起来一定很沉重。这种念头出现时，您通常会怎么做？",
        "severe": "如果这种想法反复出现，需要尽快和可信的家人或医生说说。您现在安全吗？",
        "uncertain": "这些想法有时不容易说清。有没有哪一刻让您觉得自己不被需要？",
        "avoidant": "不想说也没关系。我们先停一停，您现在安全吗？",
    },
    "medication": {
        "positive": "能按时吃药很好。最近有没有漏服，或者吃完哪里不舒服？",
        "mild": "偶尔忘记一顿也常见。您平时用什么办法提醒自己？",
        "moderate": "经常漏服会影响效果。是记不住时间，还是担心副作用？",
        "severe": "如果经常停药或漏服，需要尽快和医生确认。现在有没有不舒服？",
        "uncertain": "用药情况有时不好说。最近一周大概漏过几次？",
        "avoidant": "不想聊吃药也可以，不过按时用药很重要。我们换个话题，最近身体怎么样？",
    },
    "medical_history": {
        "positive": "慢性病稳定是好事。最近复诊和用药还规律吗？",
        "mild": "偶尔有波动也要留意。最近血压或血糖有没有变化？",
        "moderate": "最近指标不太稳定，需要关注。有没有复诊或联系医生？",
        "severe": "如果出现明显不适或指标失控，要尽快就医。现在有没有胸痛、头晕等急症？",
        "uncertain": "病情变化有时难判断。最近有没有测量或复诊？",
        "avoidant": "不想聊病也可以。我们说说最近精神和睡眠怎么样？",
    },
    "function": {
        "positive": "日常活动还顺手很好。有没有哪些事情需要家人帮忙？",
        "mild": "偶尔费劲也值得留意。主要是走路、洗澡还是做饭？",
        "moderate": "日常活动经常困难会影响独立性。现在最需要帮助的是哪一项？",
        "severe": "如果基本生活需要照顾，要尽快安排支持。现在谁在帮您？",
        "uncertain": "功能变化有时不好说。和以前比，哪件事明显变慢了？",
        "avoidant": "不想聊这个也没关系。我们换个轻松点的话题，最近有出去走走吗？",
    },
    "falls": {
        "positive": "走路稳是好消息。最近有没有头晕或差点摔倒？",
        "mild": "偶尔不稳也要当心。是在起床、洗澡还是走路时明显？",
        "moderate": "经常站不稳会增加跌倒风险。家里有没有防滑或扶手？",
        "severe": "如果已经跌倒或明显站不稳，需要尽快检查。现在有没有受伤？",
        "uncertain": "偶尔说不清也很正常。最近有没有差点摔倒的情况？",
        "avoidant": "不想聊跌倒也可以，但安全最重要。现在走路需要人扶吗？",
    },
    "diet": {
        "positive": "吃得下是好事。最近体重和胃口有没有变化？",
        "mild": "偶尔吃少一点也要留意。是没胃口，还是吃东西不方便？",
        "moderate": "经常吃不下会影响体力。最近体重有没有下降？",
        "severe": "如果持续吃不下或体重明显下降，需要尽快就医。现在有没有明显乏力？",
        "uncertain": "胃口变化有时不好判断。最近一周大概有几天吃不下？",
        "avoidant": "不想聊吃饭也可以。我们说说最近精神和睡眠怎么样？",
    },
    "sensory": {
        "positive": "看听都还够用很好。最近有没有看不清或听不见的时候？",
        "mild": "偶尔看不清或听不清会影响交流。是光线、噪音还是本身变化？",
        "moderate": "看听经常受影响会给生活带来困难。有没有检查过视力或听力？",
        "severe": "如果突然看不清、听不见或伴随头晕，需要尽快就医。现在有没有这些情况？",
        "uncertain": "感官变化有时难判断。最近看电视或听人说话需要更用力吗？",
        "avoidant": "不想聊也可以。我们换个话题，最近和邻居聊过天吗？",
    },
    "care_support": {
        "positive": "有人照顾和支持很重要。最近谁帮您最多？",
        "mild": "偶尔需要帮忙很正常。遇到急事能马上联系到人吗？",
        "moderate": "经常需要照护会增加负担。现在主要靠谁帮忙？",
        "severe": "如果缺少照护又独居，需要尽快安排支持。有没有紧急联系人？",
        "uncertain": "照护情况有时不好说。平时一个人住还是有人陪？",
        "avoidant": "不想聊这个也没关系。最近有没有让您觉得安心的人或事？",
    },
}

PROFESSIONAL_TOPICS = {
    "medication",
    "medical_history",
    "function",
    "falls",
    "diet",
    "sensory",
    "care_support",
}

SLEEP_SUBTYPE_PATTERNS = {
    "rhythm": [
        "作息",
        "难调",
        "作息乱",
        "晚睡",
        "睡晚",
        "睡得晚",
        "睡得比较晚",
        "睡得很晚",
        "睡得挺晚",
        "睡得有点晚",
        "睡得偏晚",
        "睡迟",
        "熬夜",
        "熬到",
        "看晚了",
        "聊晚了",
        "早睡",
        "昼夜",
        "倒时差",
        "白天补觉",
        "午睡太久",
        "睡得太晚",
        "起得太晚",
    ],
    "maintenance": [
        "半夜醒",
        "醒好几次",
        "容易醒",
        "睡不踏实",
        "总醒",
        "老醒",
        "夜醒",
    ],
    "early": ["早醒", "醒得早", "醒太早", "天没亮", "凌晨醒"],
    "onset": ["入睡难", "睡不着", "翻来覆去", "躺很久", "入睡困难"],
    "daytime": ["白天没精神", "白天困", "打瞌睡", "白天疲惫"],
}

SLEEP_RHYTHM_RESPONSES = {
    "positive": "作息比较规律，对睡眠和白天精神都有帮助。最近有没有需要临时调整的时候？",
    "mild": "作息有点难调整，确实会让人白天不太舒服。最近是睡得晚、起得晚，还是白天补觉比较多？",
    "moderate": "作息经常乱，调整起来会更费劲。最近晚上大概几点睡、早上几点起？",
    "severe": "作息长期难以调整，需要慢慢找回固定节奏。最近有没有固定起床时间，白天会不会补觉很久？",
    "uncertain": "作息变化有时不太好判断。最近一周大概有几天睡得比平时晚？",
    "avoidant": "不想聊作息也没关系。我们换个话题，最近白天精神怎么样？",
}

SLEEP_SUBTYPE_RESPONSES = {
    "maintenance": (
        "夜里反复醒来属于常见的睡不好，会把睡眠切得很碎，白天也容易累。"
        "一晚大概醒几次，醒了以后还能很快睡着吗？"
    ),
    "early": (
        "比平时醒得太早，常常会让人担心自己没睡够。"
        "一般比计划起床早多久，醒了以后还能再睡着吗？"
    ),
    "onset": (
        "躺下后迟迟睡不着确实让人着急。"
        "通常要躺多久才能睡着，越想睡会不会越清醒？"
    ),
    "daytime": (
        "白天困倦明显，可能和夜间睡眠质量或作息都有关。"
        "白天最困的是上午、午饭后，还是傍晚？"
    ),
}

SOLUTION_SUGGESTIONS = {
    "sleep": "可以先固定起床时间，早上见见光，白天午睡尽量不超过半小时。",
    "interest": "可以先选一件以前喜欢的小事，只做十分钟也算开始。",
    "mood": "可以先每天安排一件很小、能完成的事，并找一个信任的人说说。",
    "somatic": "可以先记录不舒服的部位、时间和诱因，方便后续给医生看。",
    "cognition": "可以把重要事项写在固定位置，必要时让家人一起帮忙留意。",
    "social": "可以先给熟悉的人打个电话，或者到楼下坐一会儿。",
    "negative_cognition": "先把这些想法告诉一个信任的人，不要一个人反复扛着。",
    "medication": "可以用分格药盒或手机提醒，漏服后不要自行加倍。",
    "medical_history": "可以把最近的测量值和用药记下来，复诊时带给医生。",
    "function": "可以把困难的动作拆小，必要时使用扶手或请家人帮忙。",
    "falls": "先清理地面障碍、增加照明，起身和转身慢一点。",
    "diet": "可以少量多餐，优先保证蛋白质和水分，持续下降要及时就医。",
    "sensory": "先检查眼镜或助听设备，突然看不清、听不见要及时就医。",
    "care_support": "可以把紧急联系人写在电话旁，必要时联系社区服务。",
}

SLEEP_POSITIVE_TERMS = [
    "睡得还行",
    "睡的还行",
    "睡得还可以",
    "睡的还可以",
    "睡得挺好",
    "睡的挺好",
    "睡得还不错",
    "睡的还不错",
    "睡得不错",
    "睡的不错",
    "睡眠还不错",
    "睡眠不错",
    "睡眠还可以",
    "睡眠还行",
]

DAYTIME_POSITIVE_TERMS = [
    "白天还行",
    "白天也还好",
    "白天还好",
    "白天挺好",
    "白天精神还行",
    "白天精神还可以",
]

LATE_SLEEP_EVENT_TERMS = [
    "电影",
    "追剧",
    "看节目",
    "看电视",
    "聊天",
    "聊到",
    "聊得",
    "聊晚",
    "刷手机",
    "打牌",
    "聚会",
    "健身",
    "锻炼",
    "运动",
    "喝茶",
    "咖啡",
]

ONE_OFF_TERMS = ["昨天", "昨晚", "前天", "有一晚", "偶尔一次", "只晚了一次"]


def classify_answer_archetype(text: str) -> str:
    compact = str(text or "").replace(" ", "")
    if any(
        phrase in compact
        for phrase in ["不想说", "别问", "不想聊", "算了", "没什么好说", "不想提"]
    ):
        return "avoidant"
    severe_terms = [
        "每天",
        "天天都",
        "一直",
        "整夜",
        "严重",
        "受不了",
        "根本",
        "完全",
    ]
    moderate_terms = [
        "经常",
        "常常",
        "老是",
        "老醒",
        "老忘",
        "很多",
        "好几次",
        "越来越",
        "总是",
        "总睡不着",
        "总忘记",
        "还是",
        "仍然",
        "难受",
        "不好",
        "低落",
        "不开心",
        "烦",
        "孤单",
        "孤独",
        "没用",
        "负担",
        "不愿意",
        "不想做",
        "提不起",
    ]
    mild_terms = ["有点", "偶尔", "有时候", "一阵", "稍微", "一点点"]
    positive_terms = [
        "挺好",
        "很好",
        "不错",
        "稳定",
        "有人陪",
        "有精神",
        "吃得下",
        "睡得着",
        "开心",
        "愿意",
    ]
    uncertain_terms = ["说不清", "不知道", "不确定", "不一定", "还好吧", "一般"]
    if has_any_unnegated_keyword(compact, severe_terms):
        return "severe"
    if has_any_unnegated_keyword(compact, moderate_terms):
        return "moderate"
    if has_any_unnegated_keyword(compact, mild_terms):
        return "mild"
    if has_any_unnegated_keyword(compact, positive_terms):
        return "positive"
    if has_any_unnegated_keyword(compact, uncertain_terms):
        return "uncertain"
    symptom_terms = [
        "失眠",
        "睡不着",
        "睡不好",
        "醒太早",
        "早醒",
        "半夜醒",
        "老醒",
        "难受",
        "不舒服",
        "疼痛",
        "低落",
        "孤独",
        "孤单",
        "没力气",
        "没有力气",
        "忘记",
        "漏服",
        "记不住",
        "站不稳",
    ]
    if has_any_unnegated_keyword(compact, symptom_terms):
        return "mild"
    if any(term in compact for term in symptom_terms):
        return "positive"
    return "unknown"


def classify_sleep_subtype(text: str) -> str:
    compact = str(text or "").replace(" ", "")
    for subtype, keywords in SLEEP_SUBTYPE_PATTERNS.items():
        if any(keyword in compact for keyword in keywords):
            return subtype
    return "unknown"


def is_situational_late_sleep(text: str) -> bool:
    compact = str(text or "").replace(" ", "")
    has_event = any(term in compact for term in LATE_SLEEP_EVENT_TERMS)
    late_sleep = any(
        term in compact
        for term in [
            "睡晚",
            "睡得晚",
            "睡得比较晚",
            "睡得很晚",
            "睡得挺晚",
            "睡得有点晚",
            "睡得偏晚",
            "较晚",
            "偏晚",
            "晚睡",
            "睡迟",
            "熬夜",
            "熬到",
            "看晚",
            "聊晚",
        ]
    ) or (
        has_event
        and any(
            term in compact
            for term in ["到很晚", "到半夜", "到凌晨", "很晚才睡", "才睡"]
        )
    )
    if not late_sleep:
        return False
    recurring_late = any(
        term in compact
        for term in [
            "经常",
            "总是",
            "最近总",
            "每天",
            "连续",
            "最近一直",
            "好几晚",
        ]
    )
    return has_event or recurring_late or any(term in compact for term in ONE_OFF_TERMS)


def build_sleep_response(
    topic: str,
    text: str,
    archetype: str,
    variant_key: int = 0,
    focus_text: str = "",
    positive_context: str = "",
) -> str | None:
    if topic != "sleep":
        return None
    full_text = str(text or "")
    focus = focus_text or full_text
    context = positive_context or text
    if "皮质醇" in full_text:
        return (
            "运动后皮质醇短时升高是常见生理反应，不能只凭这个判断皮质醇异常，"
            "也不能把入睡晚完全归因于它。更实用的是先记录健身结束到上床的时间，"
            "再看间隔。"
            "您通常晚上几点结束健身？"
        )
    exercise_related = any(
        term in full_text for term in ["健身", "锻炼", "运动"]
    ) and any(term in full_text for term in ["晚", "迟", "推迟"])
    if exercise_related:
        sleep_ok = any(term in context for term in SLEEP_POSITIVE_TERMS)
        if sleep_ok:
            return (
                "睡眠本身不错，当前更像是健身时间让入睡推迟。"
                "运动后短时兴奋或皮质醇变化不能单凭感觉判断，先看健身结束到上床相隔多久。"
                "您通常晚上几点结束健身？"
            )
        return (
            "健身时间离睡觉太近，身体仍处在兴奋状态时确实可能推迟入睡。"
            "先不急着归因，看看健身结束到上床相隔多久更实用。"
            "您通常晚上几点结束健身？"
        )
    situational_late_sleep = is_situational_late_sleep(full_text)
    if situational_late_sleep:
        from .dialogue_response_policy import (
            build_strategy_response,
            predict_sleep_strategy,
        )

        strategy = predict_sleep_strategy(full_text)["strategy"]
        trained_response = build_strategy_response(strategy)
        if trained_response:
            return trained_response
    subtype = "rhythm" if situational_late_sleep else classify_sleep_subtype(focus)
    if subtype == "unknown":
        return None
    if subtype == "rhythm" and situational_late_sleep:
        recurring = any(
            term in focus
            for term in [
                "经常",
                "总是",
                "最近总",
                "每天",
                "连续",
                "最近一直",
                "好几晚",
            ]
        )
        if recurring:
            return (
                "如果最近连续因为看电影、聊天或别的事情熬到很晚，作息会越来越难拉回来。"
                "可以先固定早上起床时间，白天午睡不超过半小时，"
                "再每两三天把上床时间提前十五分钟。"
            )
        return (
            "偶尔因为看电影、追剧或聊天晚睡很常见，单次晚睡不等于失眠。"
            "更值得看的是之后会不会补觉很多、白天没精神。"
            "这种情况一周大概几次，第二天精神怎么样？"
        )
    sleep_ok = any(term in context for term in SLEEP_POSITIVE_TERMS)
    daytime_ok = any(term in context for term in DAYTIME_POSITIVE_TERMS)
    if subtype != "rhythm":
        response = SLEEP_SUBTYPE_RESPONSES[subtype]
        if archetype in {"mild", "moderate"}:
            response = f"{response} {SOLUTION_SUGGESTIONS['sleep']}"
        if sleep_ok:
            response = f"睡眠总体还能维持，不过{response}"
        return response
    if sleep_ok and daytime_ok:
        return (
            "睡眠本身和白天状态都还可以，主要困扰是作息调整。"
            "通常晚上几点上床、早上几点起？"
        )
    if sleep_ok:
        return (
            "睡眠本身还可以，主要困扰是作息调整。"
            "通常晚上几点上床、早上几点起？"
        )
    if daytime_ok:
        return (
            "白天状态还可以，说明目前影响还不算大。"
            "晚上一般几点睡、早上几点起？"
        )
    if any(
        word in text
        for word in ["乱", "难", "晚", "补觉", "没调整好", "还没调整"]
    ):
        resolved = "moderate"
    elif archetype in SLEEP_RHYTHM_RESPONSES and archetype != "unknown":
        resolved = archetype
    else:
        resolved = "uncertain"
    openers = HUMAN_OPENERS.get(resolved, [""])
    digest = hashlib.md5(f"sleep:rhythm:{resolved}:{variant_key}".encode("utf-8")).hexdigest()
    opener = openers[int(digest[:8], 16) % len(openers)]
    response = SLEEP_RHYTHM_RESPONSES[resolved]
    if resolved in {"mild", "moderate"}:
        response = _insert_suggestion(response, SOLUTION_SUGGESTIONS["sleep"])
    return f"{opener} {response}"


def _insert_suggestion(response: str, suggestion: str) -> str:
    question_index = response.find("？")
    if question_index >= 0 and question_index < len(response) - 1:
        return (
            f"{response[: question_index + 1]} "
            f"{suggestion} "
            f"{response[question_index + 1 :].strip()}"
        )
    return f"{response} {suggestion}"


def build_policy_response(
    topic: str,
    archetype: str,
    variant_key: int = 0,
    focus_text: str = "",
    positive_context: str = "",
) -> str | None:
    responses = TOPIC_RESPONSES.get(topic)
    if not responses or archetype not in responses:
        return None
    compact = str(focus_text or "").replace(" ", "")
    if topic == "social" and any(
        has_any_unnegated_keyword(compact, [term])
        for term in ["孤单", "孤独", "没人说话", "没人陪"]
    ):
        prefix = (
            "身边有人仍觉得孤单，说明缺少的可能是能说心里话的交流。"
            if positive_context
            else ""
        )
        return (
            f"{prefix}可以先给一个信得过的人打个电话，约一次见面或散步。"
            "最近有没有哪个人是您愿意多聊几句的？"
        )
    if topic == "medication" and has_any_unnegated_keyword(
        compact, ["头晕", "恶心", "心慌", "皮疹", "副作用", "难受"]
    ):
        return (
            "如果吃药后反复头晕、恶心或心慌，先不要自行停药。"
            "可以把药名、服用时间和不适记下来，尽快联系开药医生确认。"
            "这种不适是每次吃完都出现，还是偶尔才有？"
        )
    if topic == "medication" and has_any_unnegated_keyword(
        compact, ["忘", "漏服", "没吃", "停药"]
    ):
        frequency = "偶尔" if archetype == "mild" else "经常"
        severe_note = (
            "如果已经连续停药或漏服很多天，请尽快和医生确认怎么补服。"
            if archetype == "severe"
            else ""
        )
        return (
            f"{frequency}忘记服药也值得调整。可以把药放在固定位置，"
            "用分格药盒或手机提醒，漏服后不要自行加倍。"
            f"{severe_note}"
            "您更容易忘记早上还是晚上那一顿？"
        )
    if topic == "medical_history" and any(
        term in compact
        for term in ["血压", "血糖", "高血压", "糖尿病", "心脏病", "慢病"]
    ):
        return (
            "既然最近血压、血糖等指标不稳定，可以把测量时间和数值连续记几天，"
            "复诊时带给医生看。最近有没有测得特别高或特别低的数值？"
        )
    if topic == "function" and any(
        term in compact
        for term in ["洗澡", "穿衣", "上厕所", "做饭", "走路", "需要帮忙"]
    ):
        return (
            "洗澡、穿衣这些日常活动需要帮忙，说明生活里确实有些不方便。"
            "现在最费劲的是洗澡、穿衣、上厕所，还是走路？"
        )
    if topic == "falls" and any(
        term in compact
        for term in ["站不稳", "跌倒", "摔倒", "头晕", "怕摔"]
    ):
        return (
            "站不稳又担心跌倒，先不要硬撑。是在起床、洗澡还是走路时更明显？"
            "先清理地面障碍、增加照明，起身和转身慢一点。"
        )
    if topic == "diet" and any(
        term in compact
        for term in ["吃不下", "没胃口", "体重", "瘦了", "消瘦"]
    ):
        return (
            "已经吃不下又伴随体重下降，需要比单纯胃口差更重视。"
            "体重大概下降了多少、持续多久了？如果继续下降，建议尽快就医。"
        )
    if topic == "sensory" and any(
        term in compact
        for term in [
            "听不清",
            "听东西不清楚",
            "看不清",
            "看东西不清楚",
            "听力",
            "视力",
            "耳鸣",
        ]
    ):
        return (
            "看或听不清楚会影响交流，也容易让人减少出门。"
            "这是逐渐出现的，还是突然发生的？如果突然变化，要尽快就医。"
        )
    if topic == "care_support" and any(
        term in compact
        for term in ["老伴", "子女", "孩子", "照顾", "有人陪", "家人"]
    ):
        return (
            "身边有人照顾是重要的支持。遇到急事时，照护您的家人能马上联系到吗？"
            "也建议把紧急联系人和常用药放在固定位置。"
        )
    if topic in PROFESSIONAL_TOPICS:
        response = responses[archetype]
        if archetype in {"mild", "moderate"} and topic in SOLUTION_SUGGESTIONS:
            response = _insert_suggestion(response, SOLUTION_SUGGESTIONS[topic])
        return response
    openers = HUMAN_OPENERS.get(archetype, [""])
    digest = hashlib.md5(f"{topic}:{archetype}:{variant_key}".encode("utf-8")).hexdigest()
    opener = openers[int(digest[:8], 16) % len(openers)]
    response = responses[archetype]
    if archetype in {"mild", "moderate"} and topic in SOLUTION_SUGGESTIONS:
        response = _insert_suggestion(response, SOLUTION_SUGGESTIONS[topic])
    return f"{opener} {response}"
