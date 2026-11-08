"""Adaptive empathetic interview agent for elderly depression screening.

The agent is intentionally independent from any single LLM provider:

* without an API key it runs a local, topic-aware interviewer;
* with ``OPENAI_API_KEY`` (or an api_key in context) it uses the LLM to
  generate warmer replies and falls back to the local interviewer on error.

The interface remains compatible with the original ``AgentAdapter`` contract.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import wraps
from typing import Any

from elderly_depression.dialogue_understanding import (
    UtteranceAnalysis,
    analyze_utterance,
)
from elderly_depression.negation import has_any_unnegated_keyword


SAFETY_WORDS = [
    "想死",
    "自杀",
    "不想活",
    "不想活了",
    "活不下去",
    "活着没意思",
    "活着没意义",
    "没有活下去的意义",
    "不如走了",
    "走了算了",
    "想一了百了",
    "结束生命",
    "伤害自己",
    "自残",
    "死了算了",
    "离开这个世界",
    "不想醒来",
    "希望睡过去",
    "消失就好了",
    "没有我更好",
    "活够了",
    "真的撑不住了",
    "快撑不住了",
    "想解脱",
]

BODY_PARTS = [
    "肩膀",
    "肩",
    "脖子",
    "颈椎",
    "腰",
    "背",
    "膝盖",
    "关节",
    "髋",
    "肚子",
    "胃",
    "胸口",
    "头",
    "手臂",
    "手指",
    "腿",
    "脚踝",
    "脚",
    "牙",
]

BODY_PART_FOLLOWUPS = {
    "肩膀": "抬手、穿衣或睡觉时会不会更明显？",
    "肩": "抬手、穿衣或睡觉时会不会更明显？",
    "脖子": "转头或低头时会不会更明显？",
    "颈椎": "转头或低头时会不会更明显？",
    "腰": "弯腰、久坐或起床时会不会更明显？",
    "背": "弯腰、久坐或起床时会不会更明显？",
    "膝盖": "走路、上下楼或起身时会不会更明显？",
    "关节": "活动关节时会不会更疼或更僵？",
    "髋": "走路、坐下或翻身时会不会更明显？",
    "肚子": "吃饭前后会不会更明显，有没有恶心或腹泻？",
    "胃": "吃饭前后会不会更明显，有没有恶心或反酸？",
    "胸口": "活动或紧张时会不会更明显，有没有心慌？",
    "头": "是持续疼还是一阵一阵，有没有头晕或看东西不清？",
    "手臂": "抬手或拿东西时会不会更明显？",
    "手指": "弯曲手指或拿东西时会不会更明显？",
    "腿": "走路或上下楼时会不会更明显？",
    "脚踝": "走路或站立时会不会更明显？",
    "脚": "走路或站立时会不会更明显？",
    "牙": "吃东西或碰到冷热时会不会更疼？",
}

PHQ9_EVIDENCE_MAP = {
    "兴趣减退": [1],
    "情绪低落": [2],
    "睡眠差": [3],
    "躯体不适": [4, 5],
    "自责/负担": [6],
    "认知下降": [7],
    "自杀意念": [9],
    "无望感": [2, 6],
}

PHQ9_ITEM_LABELS = {
    1: "兴趣或愉悦感减退",
    2: "情绪低落、沮丧或绝望",
    3: "入睡困难、睡眠不安或睡眠过多",
    4: "疲倦或没有活力",
    5: "食欲不振或进食过多",
    6: "自责、失败感或让家人失望",
    7: "注意力难以集中",
    8: "动作或说话迟缓、烦躁不安",
    9: "自伤或死亡相关念头",
}

TOPIC_REFLECTIONS = {
    "sleep": [
        "睡眠变化确实会影响一整天的状态。",
        "睡不好会让人白天明显更累。",
        "睡眠这件事反复起来，确实挺折腾人。",
    ],
    "interest": [
        "以前喜欢的事变得不想做，会让人有些失落。",
        "这种提不起劲的变化，确实值得慢慢说说。",
        "听起来这些事现在做起来少了些乐趣。",
    ],
    "mood": [
        "这段时间心里不好受，确实不容易。",
        "情绪压在心里，人会很累。",
        "这种难受的感觉，慢慢说就好。",
    ],
    "somatic": [
        "身体不舒服会拖累心情和日常活动。",
        "长期不舒服确实会让人烦躁、没力气。",
        "身体上的难受和情绪常常会互相影响。",
    ],
    "cognition": [
        "记性和注意力变化会让人担心。",
        "这种变化如果持续，确实值得留意。",
        "记不住事情时，人容易着急，也会影响做事的信心。",
    ],
    "social": [
        "缺少说话的人，孤单感会很明显。",
        "一个人待久了，心里会空落落的。",
        "有人能说说话，对心情确实很重要。",
    ],
    "negative_cognition": [
        "这些想法听起来很沉重。",
        "把这样的感受说出来并不容易。",
        "这种自责和没盼头的感觉，确实会压得人很累。",
    ],
}

PROGRESS_QUESTIONS = {
    "sleep": "这种睡眠变化持续多久了，白天精神受影响吗？",
    "interest": "这种变化持续多久了，最近还愿意做哪些以前喜欢的事？",
    "mood": "这种感受大概持续多久了，有没有影响吃饭、睡眠或出门？",
    "somatic": "这个不舒服大概持续多久了，做什么事情时更明显？",
    "cognition": "这种记性或注意力变化，是最近才明显，还是已经持续一阵子？",
    "social": "最近一周，您和谁见过面或通过电话？",
    "negative_cognition": "这种想法最近大概多久出现一次？",
    "medication": "您更容易忘记哪一顿药，最近有没有头晕、心慌或皮疹？",
    "medical_history": "最近一次测量或复查是什么时候，结果大概怎么样？",
    "function": "现在最影响日常生活的，是洗澡、穿衣、做饭，还是走路？",
    "falls": "最近跌倒或差点跌倒是在什么场景，有没有受伤？",
    "diet": "最近体重有没有明显变化，持续多久了？",
    "sensory": "这个变化是慢慢出现还是突然出现，影响日常交流了吗？",
    "care_support": "遇到急事时，身边有能马上联系到的人吗？",
}

GENERIC_REFLECTIONS = [
    "我在听，您可以慢慢说。",
    "您可以继续说说，不用急着组织语言。",
    "嗯，我明白您的意思了。",
]

POSITIVE_REFLECTIONS = [
    "这些能让您舒服一点的事，很重要。",
    "听起来这些小事给了您一些力量。",
    "有这样的支撑，日子会好过一些。",
]


CUE_RULES: list[tuple[str, list[str]]] = [
    ("自杀意念", SAFETY_WORDS),
    ("无望感", ["没盼头", "绝望", "没希望", "没有出路", "熬不下去", "熬不住"]),
    ("自责/负担", ["拖累", "负担", "对不起", "累赘", "没用", "废物", "拖后腿"]),
    ("睡眠差", ["失眠", "睡不着", "睡不好", "半夜醒", "醒得早", "多梦", "没精神", "入睡难"]),
    ("兴趣减退", ["没意思", "没兴趣", "提不起兴趣", "什么都不想做", "不想做事", "提不起劲", "不想动", "不想出门", "不爱说话", "没劲"]),
    ("情绪低落", ["难过", "伤心", "低落", "烦", "心里堵", "难受", "哭", "压抑", "开心不起来"]),
    ("躯体不适", ["疼", "痛", "乏力", "没力气", "没有力气", "吃不下", "没胃口", "胃口", "头晕", "心慌", "胸闷", "身体不舒服", "肩膀", "肩", "脖子", "颈椎", "腰", "背", "膝盖", "关节", "髋", "肚子", "胃", "胸口", "手臂", "手指", "腿", "脚踝", "脚", "牙"]),
    ("认知下降", ["记不住", "记性", "注意力", "脑子", "忘事", "糊涂", "反应慢"]),
    ("社交孤立", ["孤独", "孤单", "没人说话", "没人陪", "不出门", "见不到人", "一个人"]),
    ("保护因素", ["有人陪", "子女", "老伴", "朋友", "邻居", "散步", "活动", "有人说话", "医生", "吃药"]),
]


GREETING_WORDS = [
    "你好",
    "您好",
    "你好呀",
    "早上好",
    "下午好",
    "晚上好",
    "嗨",
    "在吗",
    "在不在",
    "hello",
    "hi",
]


COMPANION_FASTPATH_WORDS = [
    "开心",
    "高兴",
    "不错",
    "喜欢",
    "出去",
    "散步",
    "锻炼",
    "下棋",
    "听戏",
    "养花",
    "串门",
    "天气",
    "下雨",
    "下雪",
    "太阳",
    "热",
    "冷",
    "晴",
    "风",
    "吃",
    "饭",
    "菜",
    "胃口",
    "做饭",
    "早餐",
    "午饭",
    "晚饭",
    "孩子",
    "孙子",
    "老伴",
    "家人",
    "儿子",
    "女儿",
    "身体",
    "疼",
    "痛",
    "累",
    "不舒服",
    "病",
    "药",
    "难过",
    "伤心",
    "孤单",
    "孤独",
    "烦",
    "心情",
    "没意思",
    "不好",
    "不行",
    "难受",
    "糟糕",
    "没胃口",
    "吃不下",
    "睡不着",
]


TOPIC_MAP = {
    "sleep": [
        "睡眠",
        "睡不着",
        "睡不好",
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
        "半夜",
        "醒",
        "梦",
        "困",
        "没精神",
        "入睡",
        "失眠",
        "作息",
        "难调",
    ],
    "interest": ["没意思", "没兴趣", "提不起兴趣", "什么都不想做", "不想做事", "提不起劲", "不想动", "不想出门", "爱好", "喜欢", "开心不起来", "没劲"],
    "mood": ["难过", "伤心", "低落", "烦", "心里堵", "难受", "哭", "压抑", "情绪", "心情"],
    "somatic": ["疼", "痛", "乏力", "没力气", "没有力气", "吃不下", "没胃口", "胃口", "头晕", "心慌", "胸闷", "身体", "累", "病", "肩膀", "肩", "脖子", "颈椎", "腰", "背", "膝盖", "关节", "髋", "肚子", "胃", "胸口", "手臂", "手指", "腿", "脚踝", "脚", "牙"],
    "cognition": ["记不住", "记性", "注意力", "脑子", "忘事", "糊涂", "反应慢", "想不起来"],
    "social": ["孤独", "孤单", "没人说话", "没人陪", "不出门", "朋友", "邻居", "家人", "子女", "一个人"],
    "negative_cognition": ["没盼头", "绝望", "没希望", "拖累", "负担", "对不起", "没用", "累赘", "活着没意思", "没有出路"],
    "safety": SAFETY_WORDS,
}

PROFESSIONAL_PARENT_TOPICS = {
    "medication": "somatic",
    "medical_history": "somatic",
    "function": "somatic",
    "falls": "somatic",
    "diet": "somatic",
    "sensory": "somatic",
    "care_support": "social",
}

PROFESSIONAL_CUE_MAP = {
    "medication": "用药管理",
    "medical_history": "慢病管理",
    "function": "日常功能变化",
    "falls": "跌倒风险",
    "diet": "饮食变化",
    "sensory": "视听变化",
    "care_support": "照护支持",
}

ADDRESSED_ISSUE_LABELS = {
    "sleep_rhythm": "睡眠作息",
    "sleep_maintenance": "夜间反复醒",
    "sleep_early": "早醒",
    "sleep_onset": "入睡困难",
    "sleep_daytime": "白天困倦",
    "sleep_quality": "夜间睡眠质量",
    "sleep_impact": "睡眠对白天的影响",
    "medication": "用药管理",
    "medical_history": "慢病管理",
    "function": "日常功能",
    "falls": "跌倒风险",
    "diet": "饮食与体重",
    "sensory": "视力与听力",
    "care_support": "照护支持",
}

PROFESSIONAL_FOLLOWUP_KEYWORDS = {
    "medication": [
        "早上",
        "中午",
        "晚上",
        "睡前",
        "两顿",
        "都忘",
        "记不住",
        "不确定",
    ],
    "medical_history": [
        "高",
        "低",
        "正常",
        "没测",
        "没有测",
        "不知道",
        "复诊",
        "医生",
    ],
    "function": ["洗澡", "穿衣", "上厕所", "做饭", "走路", "都需要", "还能"],
    "falls": ["起床", "洗澡", "走路", "转身", "都有", "偶尔", "经常"],
    "diet": ["斤", "公斤", "千克", "天", "周", "月", "不知道", "没称"],
    "sensory": ["逐渐", "慢慢", "突然", "一边", "两边", "一只", "两只"],
    "care_support": ["能", "不能", "有人", "没人", "电话", "号码", "独居"],
}

PROFESSIONAL_FOLLOWUP_CLOSERS = {
    "medication": (
        "容易忘的那一顿可以单独放一板药或设一个固定提醒。"
        "如果连续漏服，先联系医生确认，不要自行加倍。"
    ),
    "medication_side_effect": (
        "请把不适出现的时间、持续多久和当时服用的药记下来。"
        "如果头晕加重、晕倒或心慌明显，先联系医生或及时就医。"
    ),
    "medical_history": (
        "把数值和测量时间记在同一个本子上，复诊时按时间顺序给医生看。"
        "如果数值明显异常并伴随胸闷、头晕等不适，要及时就医。"
    ),
    "function": (
        "最费力的动作可以拆小来做，洗澡和起身时先保证旁边有扶手或家人在场。"
    ),
    "falls": (
        "先处理地面障碍、照明和拖鞋防滑。起身、转身慢一点，"
        "近期反复不稳时尽量让人陪同。"
    ),
    "diet": (
        "先少量多餐并保证饮水，记录体重变化。持续下降或伴吞咽困难时要尽快就医。"
    ),
    "sensory": (
        "如果是逐渐变化，可以安排视力或听力检查；突然变化要尽快就医。"
    ),
    "care_support": (
        "把紧急联系人、常用药和就医信息放在固定位置，遇到急事会更快得到帮助。"
    ),
}

CORE_FOLLOWUP_CLOSERS = {
    "sleep": "这些睡眠细节我已经记下，可以和后面的量表结果一起看。",
    "interest": "能说出具体场景，有助于区分短暂没劲和持续的兴趣减退。",
    "mood": "这些情绪细节有助于判断持续时间和影响程度。",
    "somatic": "把部位、时间和诱因记下来，之后给医生看会更清楚。",
    "cognition": "这种具体变化比笼统地说记性差更有参考价值。",
    "social": "家人愿意常来是重要的支持，可以留意见面后心情是否轻松一点。",
    "negative_cognition": "出现这些想法时，有一个能信任的人可以说一说很重要。",
}

STRONG_CARE_RESPONSES = {
    "mood": (
        "听到您最近经常被这些情绪困住，我有些担心，也很感谢您愿意说出来。"
        "这不是您太脆弱，一个人扛着会很累。"
        "您愿意的话，可以先联系一位信任的家人、朋友或医生，让对方知道您最近的状态；"
        "如果低落已经持续两周以上，或影响了吃饭、睡眠和日常活动，建议尽快做专业评估。"
        "这种状态有没有影响吃饭、睡觉、出门或日常事务？"
    ),
    "cognition": (
        "经常记不住或集中不了注意力，确实会让人着急，也会打击做事的信心。"
        "先不要责怪自己，您愿意的话，可以把钥匙、药和电话放在固定位置，并让家人知道最近的变化。"
        "如果变化突然加重或已经影响日常生活，建议尽快做专业评估。"
        "现在最影响您的是忘事、找不到东西，还是注意力难以集中？"
    ),
    "negative_cognition": (
        "听到您这样看自己，我很担心，也很感谢您愿意把这些话说出来。"
        "这种自责和没盼头的感觉会把人压得很累，但不是您的错，也不需要一个人扛。"
        "您愿意的话，请先告诉一位信任的家人、朋友或医生，让对方陪您一会儿；"
        "如果出现伤害自己的想法，请立即联系家人并拨打 120、110 或心理援助热线 12356。"
        "您现在安全吗，身边有没有能马上联系到的人？"
    ),
}

AFFIRMATIVE_SHORT_ANSWERS = {
    "愿意",
    "愿意的",
    "愿意啊",
    "当然愿意",
    "是的",
    "是",
    "对",
    "对的",
    "会的",
    "会",
    "有的",
    "有",
    "能",
    "可以",
    "嗯",
    "好",
    "好的",
}

NEGATIVE_SHORT_ANSWERS = {
    "不愿意",
    "不想",
    "不想做",
    "不会",
    "不会的",
    "没有",
    "没",
    "不能",
    "不可以",
    "不是",
    "否",
}

PARTIAL_AFFIRMATIVE_SHORT_ANSWERS = {
    "偶尔会",
    "有时候会",
    "有时会",
    "会一点",
    "有一点",
    "有点",
    "偶尔",
    "有时候",
    "有时",
    "偶尔有",
    "偶尔出现",
}

UNCERTAIN_SHORT_ANSWERS = {
    "还行吧",
    "一般",
    "说不清",
    "不知道",
    "不确定",
    "不好说",
    "可能吧",
}

EXPLICIT_CORRECTION_PHRASES = [
    "不是这个意思",
    "我不是这个意思",
    "我没这么说",
    "我没有这么说",
    "你理解错了",
    "你误会了",
    "听错了",
]

BOT_DISSATISFACTION_PHRASES = [
    "太机械",
    "太人机",
    "像机器人",
    "答非所问",
    "你听不懂",
    "你没听懂",
    "太啰嗦",
    "别问了",
    "不要一直问",
    "不想回答",
    "不想说这个",
    "换个话题",
]

ASSUMPTION_TERMS = [
    "乐趣",
    "兴趣",
    "没意思",
    "难受",
    "孤单",
    "孤独",
    "睡不好",
    "失眠",
    "疼痛",
    "不舒服",
    "忘记吃药",
    "记不住",
]

DIAGNOSIS_REQUEST_PATTERNS = [
    "我是不是得了",
    "是不是抑郁症",
    "是不是痴呆",
    "确诊一下",
    "直接告诉我是",
]

MEDICATION_CHANGE_PATTERNS = [
    "停药",
    "换药",
    "加量",
    "减量",
    "不想吃药",
    "不想继续吃药",
    "药不吃了",
    "把药停了",
    "自行停药",
    "不用看医生",
    "不用去医院",
    "直接吃什么药",
    "吃什么药能好",
]

HELP_REQUEST_PATTERNS = [
    "帮帮我",
    "我该怎么办",
    "不知道怎么办",
    "该怎么处理",
    "怎么做才好",
    "我需要帮助",
    "救命",
]


@dataclass
class Topic:
    key: str
    label: str
    opening: str
    followups: list[str]


TOPICS: list[Topic] = [
    Topic(
        "sleep",
        "睡眠与精力",
        "最近这一周，您晚上睡得好吗？会不会半夜醒、醒得早，或者白天总没精神？",
        [
            "是入睡困难，还是半夜容易醒、醒得早？",
            "如果没睡好，第二天身体和精神会感觉怎么样？",
        ],
    ),
    Topic(
        "interest",
        "兴趣与愉悦",
        "以前让您高兴的事，比如下棋、听戏、养花、串门，现在还愿意做吗？",
        [
            "最近一次做喜欢的事，感觉还和以前一样开心吗？",
            "有没有因为提不起兴趣，就不太愿意出门或见人？",
        ],
    ),
    Topic(
        "mood",
        "情绪与感受",
        "最近是不是经常觉得心里堵、难过，或者会莫名其妙想掉眼泪？",
        [
            "这种难受的感觉，大概一天里会出现多久？",
            "有没有什么时候会觉得稍微轻松一点？",
        ],
    ),
    Topic(
        "somatic",
        "身体与共病",
        "身体上有没有不舒服，比如疼痛、乏力、胃口不好？这些会不会反过来影响您的心情？",
        [
            "哪个地方最不舒服，持续多久了？",
            "这些不舒服有没有让您更难出门、睡觉或者做事情？",
        ],
    ),
    Topic(
        "cognition",
        "认知与注意",
        "最近记事情、集中注意力，会不会觉得比以前吃力？",
        [
            "这种记性变差或注意力不集中的情况，是最近几天才明显，还是已经持续一阵子了？",
            "这些记忆或注意力变化，对您平时做事情有没有造成困扰？",
        ],
    ),
    Topic(
        "social",
        "社交与孤独",
        "平时有人跟您说说话吗？会不会觉得孤单，或者没人真正理解自己？",
        [
            "最近一周，有没有跟家人、朋友或邻居见面或通电话？",
            "遇到烦心事的时候，身边有没有能说上话的人？",
        ],
    ),
    Topic(
        "negative_cognition",
        "无望与自责",
        "会不会觉得生活没什么盼头，或者觉得自己成了家里人的负担？",
        [
            "这种想法是偶尔有，还是最近经常出现？",
            "最难受的时候，您通常会怎么熬过去？",
        ],
    ),
]

VALID_TOPIC_KEYS = {topic.key for topic in TOPICS}
INTENT_FASTPATH_THRESHOLD = 0.55
TOPIC_PRIORITY = [
    "safety",
    "negative_cognition",
    "cognition",
    "sleep",
    "interest",
    "mood",
    "social",
    "somatic",
]
PROFESSIONAL_KEYWORDS = {
    "falls": ["跌倒", "摔倒", "站不稳", "走路不稳"],
    "medication": ["吃药", "服药", "漏服", "停药", "药效", "药", "忘记吃"],
    "medical_history": ["高血压", "糖尿病", "心脏病", "复诊", "住院"],
    "function": ["洗澡", "穿衣", "做饭", "上楼", "走路费劲"],
    "diet": ["体重下降", "体重减轻", "胃口", "吃不下", "吃得少"],
    "sensory": [
        "听力",
        "视力",
        "看不清",
        "看东西不清楚",
        "听不清",
        "听东西不清楚",
        "听不见",
        "耳鸣",
    ],
    "care_support": ["照顾", "护工", "独居", "一个人住", "需要人帮"],
}
COMPANION_KEYWORDS = {
    "greeting": ["你好", "您好", "在吗", "好久不见", "刚坐下"],
    "weather": ["天气", "下雨", "下雪", "太阳", "风", "热", "冷", "晴"],
    "food": ["吃", "饭", "菜", "饺子", "粥", "胃口", "做饭", "面条"],
    "family": ["孩子", "儿子", "女儿", "孙子", "老伴", "家人"],
    "hobby": ["下棋", "听戏", "养花", "散步", "锻炼", "爱好"],
    "health": ["身体", "腿", "疼", "累", "没力气", "没有力气", "精神", "睡得", "作息", "肩膀", "肩", "脖子", "颈椎", "腰", "背", "膝盖", "关节", "髋", "肚子", "胃", "胸口", "头", "手臂", "手指", "脚踝", "脚", "牙"],
    "memory": ["以前", "年轻", "小时候", "想起", "老房子"],
    "life_review": ["一辈子", "人生", "这一生", "最骄傲", "成就", "遗憾", "当年"],
    "dignity": ["没用", "废物", "负担", "累赘", "不中用", "没有价值", "不如以前"],
    "negative_event": ["烦", "难受", "委屈", "不开心", "心情不好"],
    "positive_event": ["开心", "高兴", "好事", "来看我", "打电话", "收到"],
    "loneliness": ["孤单", "孤独", "一个人", "没人说话", "冷清"],
    "festival": ["过节", "春节", "中秋", "端午", "年货", "团聚"],
    "pet": ["猫", "狗", "小鸟", "宠物"],
    "media": ["电视", "新闻", "电影", "听歌", "老歌"],
    "neighbor": ["邻居", "楼下", "小区", "串门"],
    "farewell": ["先聊到", "下次再聊", "谢谢你陪", "去休息", "再见"],
}


SAFETY_REPLY = (
    "谢谢您愿意跟我说这些，这一定很不容易。您现在的安全最重要。"
    "如果有伤害自己的想法，请先不要一个人扛着，马上告诉身边可信的家人或医生；"
    "紧急时请拨打当地精神卫生热线或急救电话。我在这里陪您，但无法替代专业急救。"
    "您现在安全吗？身边有没有能马上联系到的人？"
)

CLOSING = (
    "谢谢您今天跟我聊了这么多。听起来您在睡眠、情绪和身体感受上都有一些负担，"
    "这些信息我会整理成风险线索。您最希望先解决、或者最让您放心不下的是哪一件事？"
)

COMPANION_GREETING = (
    "您好呀，今天过得怎么样？"
    "咱们可以随便聊聊，说说天气、家里的事、吃的饭，或者心里想说的话。"
)

COMPANION_SAFETY_REPLY = (
    "谢谢您愿意跟我说这些。您对我来说很重要，安全最要紧。"
    "如果有伤害自己的想法，请先告诉身边可信的家人或医生，紧急时拨打当地精神卫生热线。"
    "我会继续在这里陪着您。您现在安全吗？身边有人能马上陪着您吗？"
)


@dataclass
class InterviewState:
    covered: set[str] = field(default_factory=set)
    current_topic: str = "sleep"
    attempts: dict[str, int] = field(default_factory=dict)
    risk_cues: list[str] = field(default_factory=list)
    cue_confidence: dict[str, float] = field(default_factory=dict)
    safety_flagged: bool = False
    last_reply: str = ""
    llm_next_question: str = ""
    user_facts: dict[str, str] = field(default_factory=dict)
    memory_summary: str = ""
    turn_count: int = 0
    clinical_alert: dict | None = None
    pending_topic: str | None = None
    pending_slot: str | None = None
    last_focus_text: str = ""
    last_focus_reason: str = ""
    last_positive_context: str = ""
    previous_focus_text: str = ""
    previous_topic: str = ""
    addressed_issues: set[str] = field(default_factory=set)
    suggested_topics: set[str] = field(default_factory=set)
    protective_factors: list[str] = field(default_factory=list)
    alliance_repair_count: int = 0
    last_alliance_repair: str = ""
    accepted_actions: list[str] = field(default_factory=list)
    last_guidance_ids: list[str] = field(default_factory=list)
    last_guidance_titles: list[str] = field(default_factory=list)
    last_guidance_prompt: str = ""
    safety_plan_status: str = ""
    clinical_findings: dict[str, str] = field(default_factory=dict)
    clinical_summary: str = ""
    support_preference: str = ""
    support_preference_topic: str = ""
    general_medical_id: str = ""
    general_medical_title: str = ""
    general_medical_question: str = ""
    medical_kb_sources: list[dict] = field(default_factory=list)
    medical_kb_prompt: str = ""


CHINESE_CLOCK_NUMBERS = {
    "零": 0,
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}

SLEEP_EXERCISE_PENDING_SLOTS = {
    "sleep_exercise_timing",
    "sleep_exercise_bedtime",
    "sleep_exercise_plan_choice",
}


def _parse_clock_times(text: str) -> list[int]:
    values = []
    for match in re.finditer(
        r"(?<!\d)([01]?\d|2[0-3])[:：]([0-5]\d)(?!\d)",
        text,
    ):
        values.append(int(match.group(1)))
    for match in re.finditer(r"([0-9]{1,2}|[零一二两三四五六七八九十]{1,3})点", text):
        token = match.group(1)
        if token.isdigit():
            value = int(token)
        elif token in CHINESE_CLOCK_NUMBERS:
            value = CHINESE_CLOCK_NUMBERS[token]
        elif token.startswith("十"):
            value = 10 + CHINESE_CLOCK_NUMBERS.get(token[-1], 0)
        else:
            value = CHINESE_CLOCK_NUMBERS.get(token[0], 0)
        if 0 <= value <= 23:
            values.append(value)
    return values


def _normalise_evening_hour(hour: int) -> int:
    """Map an exercise or bedtime hour to a comparable evening clock time."""
    if hour == 12:
        return 0
    if 1 <= hour <= 11:
        return hour + 12
    return hour


def _format_clock_hour(hour: int) -> str:
    return f"{hour % 24}点"


def _short_answer_polarity(text: str) -> str | None:
    raw = re.sub(r"[\s，,。！!？?]", "", str(text or ""))
    if raw in UNCERTAIN_SHORT_ANSWERS:
        return "uncertain"
    compact = raw
    compact = compact.replace("呢", "").replace("吧", "").replace("呀", "")
    if compact in NEGATIVE_SHORT_ANSWERS:
        return "negative"
    if compact in AFFIRMATIVE_SHORT_ANSWERS:
        return "affirmative"
    if compact in PARTIAL_AFFIRMATIVE_SHORT_ANSWERS:
        return "partial"
    if compact in UNCERTAIN_SHORT_ANSWERS:
        return "uncertain"
    return None


def _timed_chat(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            self.last_latency_ms = round((time.perf_counter() - started) * 1000, 2)

    return wrapper


def _last_user_text(messages: list[dict[str, str]]) -> str | None:
    for item in reversed(messages or []):
        if item.get("role") == "user" and item.get("text"):
            return item["text"]
    if messages and messages[-1].get("text"):
        return messages[-1]["text"]
    return None


def _as_openai_messages(messages: list[dict[str, str]], system: str) -> list[dict[str, str]]:
    result = [{"role": "system", "content": system}]
    for item in messages[-8:]:
        role = item.get("role", "user")
        role = role if role in {"user", "assistant"} else "user"
        text = item.get("text", "")
        if text:
            result.append({"role": role, "content": text})
    return result


class AgentAdapter:
    """Stateful adaptive interviewer with an optional OpenAI enhancement."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        base_url: str | None = os.getenv("OPENAI_BASE_URL"),
        temperature: float = 0.45,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url or None
        self.temperature = temperature
        self._sessions: dict[str, InterviewState] = {}
        self._client = None
        self._client_error: str | None = None
        self._reply_cache: dict[tuple[str, str, str], str] = {}
        self._intent_cache: dict[str, dict | None] = {}
        self.last_engine = "local"
        self.last_llm_error: str | None = None
        self.last_latency_ms = 0.0
        self.last_fastpath_reason = ""

    def set_api_key(self, api_key: str | None) -> None:
        self.api_key = api_key or self.api_key
        self._client = None
        self._client_error = None
        self.last_llm_error = None
        self._reply_cache.clear()

    def warmup(self) -> dict:
        """Preload local models so the first chat turn has no model-loading delay."""
        self._intent_prediction("最近睡得怎么样", "interview")
        try:
            from elderly_depression.dialogue_response_policy import (
                warmup_response_policy,
            )

            response_policy = warmup_response_policy()
        except Exception as exc:
            response_policy = {"ready": False, "error": str(exc)}
        return {
            "intent_cache": bool(self._intent_cache),
            "response_policy": response_policy,
        }

    def _session(self, session_id: str | None, context: dict[str, Any] | None) -> InterviewState:
        sid = session_id or (context or {}).get("session_id") or "default"
        mode = (context or {}).get("mode", "interview")
        key = f"{mode}:{sid}"
        return self._sessions.setdefault(key, InterviewState())

    @staticmethod
    def greeting(mode: str = "interview") -> str:
        if mode == "companion":
            return COMPANION_GREETING
        topic = TOPICS[0]
        return "您好，我先陪您慢慢聊一会儿。不用着急，想到哪说到哪。\n\n" + topic.opening

    @_timed_chat
    def chat(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> str:
        self.last_llm_error = None
        self.last_fastpath_reason = ""
        context = context or {}
        mode = str(context.get("mode", "interview"))
        cloud_consent = bool(context.get("cloud_consent", False))
        if context.get("api_key"):
            self.set_api_key(str(context["api_key"]))

        state = self._session(session_id, context)
        text = _last_user_text(messages)

        if not text:
            reply = self.greeting(mode)
            state.last_reply = reply
            self.last_fastpath_reason = "greeting"
            return reply

        analysis = analyze_utterance(text)
        state.previous_focus_text = state.last_focus_text
        state.previous_topic = state.current_topic
        state.last_focus_text = analysis.focus_text
        state.last_focus_reason = analysis.focus_reason
        state.last_positive_context = analysis.positive_context
        self._record_cues(state, text, analysis)
        self._retrieve_guidance(state, text, analysis)
        self._retrieve_medlineplus(state, text, analysis)
        self._update_clinical_findings(state, text)
        self._extract_facts(state, text)
        self._update_memory_summary(state, messages)
        state.turn_count += 1
        try:
            from elderly_depression.clinical_triage import assess_red_flags

            clinical_alert = assess_red_flags(text)
        except Exception:
            clinical_alert = None
        if clinical_alert:
            state.clinical_alert = clinical_alert
            if clinical_alert.get("level") == "emergency" and "自伤" in clinical_alert.get("reason", ""):
                state.safety_flagged = True
            self.last_engine = "fastpath"
            self.last_fastpath_reason = "clinical_red_flag"
            reply = str(clinical_alert.get("message") or "请尽快联系家人或医生评估。")
            state.last_reply = reply
            return reply
        if self._contains_safety(text):
            state.safety_flagged = True
            if state.pending_slot == "safety_followup":
                state.pending_slot = None
                state.safety_plan_status = "emergency_referral"
                reply = (
                    "您刚才的话让我很担心，这需要马上联系真人帮助。"
                    "请立即拨打 120、110 或心理援助热线 12356，并让身边的人陪着您。"
                    "先把可能伤害自己的物品移开，不要独处。"
                )
            else:
                state.pending_slot = "safety_followup"
                state.pending_topic = "safety"
                state.safety_plan_status = "checking"
                reply = SAFETY_REPLY if mode != "companion" else COMPANION_SAFETY_REPLY
            state.last_reply = reply
            self.last_fastpath_reason = "safety_rule"
            return reply

        boundary_reply = self._medical_boundary_reply(text)
        if boundary_reply:
            self.last_engine = "fastpath"
            self.last_fastpath_reason = "medical_boundary"
            state.last_reply = boundary_reply
            return boundary_reply

        help_reply = self._handle_help_request(state, text)
        if help_reply:
            self.last_engine = "fastpath"
            self.last_fastpath_reason = "help_request"
            state.last_reply = help_reply
            return help_reply

        general_medical_reply = (
            None
            if state.pending_slot == "general_medical_followup"
            else self._handle_general_medical(state, text)
        )
        if general_medical_reply:
            self.last_engine = "fastpath"
            self.last_fastpath_reason = "medical_knowledge"
            state.last_reply = general_medical_reply
            return general_medical_reply

        repair_reply = self._handle_alliance_repair(state, text)
        if repair_reply:
            self.last_engine = "fastpath"
            self.last_fastpath_reason = "alliance_repair"
            state.last_reply = repair_reply
            return repair_reply

        correction_reply = self._handle_user_correction(state, text)
        if correction_reply:
            self.last_engine = "fastpath"
            self.last_fastpath_reason = "user_correction"
            state.last_reply = correction_reply
            return correction_reply

        pending_reply = self._handle_pending_slot(state, text, mode, analysis)
        if pending_reply:
            pending_reply = self._ensure_interview_progress(
                state,
                pending_reply,
                mode,
                attach_existing_question=True,
            )
            self.last_engine = "fastpath"
            self.last_fastpath_reason = "slot_followup"
            state.last_reply = pending_reply
            return pending_reply

        cache_key = (
            str(mode),
            str(session_id or context.get("session_id") or "default"),
            re.sub(r"\s+", "", text),
        )
        if cache_key in self._reply_cache:
            self.last_engine = "cache"
            reply = self._reply_cache[cache_key]
            state.last_reply = reply
            self.last_fastpath_reason = "response_cache"
            return reply

        model_reply = self._model_fast_reply(state, text, mode, analysis)
        if model_reply:
            model_reply = self._ensure_interview_progress(
                state,
                model_reply,
                mode,
            )
            if mode == "companion" and any(
                phrase in text
                for phrase in ["难受", "烦", "孤单", "孤独", "没用", "负担"]
            ):
                state.pending_slot = "support_preference"
                state.support_preference_topic = (
                    self._companion_keyword_intent(text)
                    or state.current_topic
                )
            self.last_engine = "fastpath"
            state.last_reply = model_reply
            self.last_fastpath_reason = "intent_model"
            return model_reply

        use_fastpath = (
            not self.api_key
            or not cloud_consent
            or self._should_use_fastpath(
                mode,
                text,
                analysis.focus_text,
            )
        )
        if self.api_key and cloud_consent and not use_fastpath:
            llm_reply = self._try_llm_reply(messages, context, state, text, mode)
            if llm_reply:
                llm_reply = self._ensure_interview_progress(
                    state,
                    llm_reply,
                    mode,
                )
                self.last_engine = "llm"
                self.last_fastpath_reason = "llm"
                self._reply_cache[cache_key] = llm_reply
                if len(self._reply_cache) > 200:
                    self._reply_cache.pop(next(iter(self._reply_cache)))
                state.last_reply = llm_reply
                return llm_reply
            self.last_llm_error = self._client_error
            self.last_fastpath_reason = "llm_fallback"

        self.last_engine = "fastpath" if use_fastpath else "local"
        if not self.last_llm_error:
            self.last_fastpath_reason = (
                "keyword_or_rule" if use_fastpath else "local_fallback"
            )
        try:
            reply = (
                self._rule_reply(state, text, analysis)
                if mode != "companion"
                else self._companion_rule_reply(state, text, analysis)
            )
            reply = self._ensure_interview_progress(
                state,
                reply,
                mode,
                attach_existing_question=True,
            )
        except Exception:
            reply = "我刚才没接好话头。您别着急，再说一遍就好，我会继续在这里听着。"
        state.last_reply = reply
        return reply

    @staticmethod
    def _is_greeting(text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        return any(word in compact for word in GREETING_WORDS)

    def _intent_prediction(self, text: str, mode: str = "interview") -> dict | None:
        normalized = re.sub(r"\s+", "", text)
        cache_key = f"{mode}:{normalized}"
        if cache_key in self._intent_cache:
            return self._intent_cache[cache_key]
        try:
            from elderly_depression.dialogue_intent import predict_intent

            prediction = predict_intent(text, mode=mode)
        except Exception:
            prediction = None
        self._intent_cache[cache_key] = prediction
        if len(self._intent_cache) > 256:
            self._intent_cache.pop(next(iter(self._intent_cache)))
        return prediction

    def _handle_alliance_repair(
        self,
        state: InterviewState,
        text: str,
    ) -> str | None:
        compact = re.sub(r"\s+", "", text)
        if not any(phrase in compact for phrase in BOT_DISSATISFACTION_PHRASES):
            return None
        state.alliance_repair_count += 1
        if any(
            phrase in compact
            for phrase in ["别问了", "不要一直问", "不想回答", "不想说这个"]
        ):
            state.last_alliance_repair = "boundary"
            state.pending_slot = "alliance_repair"
            state.pending_topic = state.current_topic
            return (
                "好，这一题不问了，您有权利停下来。"
                "我们可以换个轻松话题，或者先安静一会儿。您想怎么来都可以。"
            )
        if any(
            phrase in compact
            for phrase in ["太机械", "太人机", "像机器人", "太啰嗦"]
        ):
            state.last_alliance_repair = "style"
            state.pending_slot = "alliance_repair"
            state.pending_topic = state.current_topic
            return (
                "抱歉，刚才说得太绕，让您觉得不像在听您说话。"
                "我会说短一点，也不急着追问。您想接着聊刚才的事，还是先换轻松话题？"
            )
        state.last_alliance_repair = "misunderstanding"
        state.pending_slot = "alliance_repair"
        state.pending_topic = state.current_topic
        previous = state.previous_focus_text or "您刚才说的话"
        return (
            f"是我刚才没接准，不是您表达得不好。您刚才的重点是“{previous}”。"
            "我会按这个来。您愿意先接着这件事说，还是换个话题？"
        )

    @staticmethod
    def _handle_help_request(
        state: InterviewState,
        text: str,
    ) -> str | None:
        compact = re.sub(r"\s+", "", text)
        if not any(phrase in compact for phrase in HELP_REQUEST_PATTERNS):
            return None
        state.pending_slot = "help_triage"
        state.pending_topic = state.current_topic
        return (
            "您愿意主动求助很重要。我们先只处理眼前最急的一件事："
            "现在最需要帮助的是自身安全、身体急症，还是情绪和生活问题？"
        )

    def _handle_help_triage(
        self,
        state: InterviewState,
        text: str,
    ) -> str:
        compact = re.sub(r"\s+", "", text)
        state.pending_slot = None
        state.pending_topic = None
        if any(
            phrase in compact
            for phrase in ["安全", "想死", "伤害自己", "不想活", "活不下去"]
        ):
            state.safety_flagged = True
            state.pending_slot = "safety_followup"
            state.pending_topic = "safety"
            state.safety_plan_status = "checking"
            return SAFETY_REPLY
        if any(
            phrase in compact
            for phrase in [
                "胸痛",
                "喘不上气",
                "呼吸困难",
                "晕倒",
                "意识不清",
                "大出血",
                "身体急症",
            ]
        ):
            return (
                "这可能是需要立即处理的急症。请马上拨打 120，"
                "不要独自驾车，并让身边的人陪着您等待急救。"
            )
        if any(
            phrase in compact
            for phrase in ["身体", "疼", "头晕", "不舒服", "没力气"]
        ):
            return (
                "身体不舒服时先把安全放在第一位。请告诉我最明显的症状和开始时间；"
                "如果出现胸痛、呼吸困难、晕倒或意识变化，请立即拨打 120。"
            )
        return (
            "好，我们先把问题缩小到一个现在能做的动作："
            "给一位信任的人发消息或打电话，只说“我现在需要有人陪我说几句”，不必一次解释清楚。"
            "如果您愿意，再告诉我现在最压心的是哪件事。"
        )

    def _handle_support_preference(
        self,
        state: InterviewState,
        text: str,
    ) -> str | None:
        compact = re.sub(r"\s+", "", text)
        if any(
            word in compact
            for word in ["听我", "听您", "陪我说", "先听", "只想说"]
        ):
            preference = "listen"
        elif any(
            word in compact
            for word in ["想办法", "建议", "怎么做", "办法"]
        ):
            preference = "action"
        elif any(word in compact for word in ["都行", "都要", "一起", "都需要"]):
            preference = "both"
        elif any(word in compact for word in ["不知道", "随便", "说不清"]):
            preference = "listen"
        else:
            state.pending_slot = None
            state.pending_topic = None
            return None
        state.support_preference = preference
        state.pending_slot = None
        state.pending_topic = None
        if preference == "listen":
            return (
                "好，我先不急着给建议。您从最想说的地方开始，"
                "我在这儿听着，不打断您。"
            )
        try:
            from elderly_depression.dialogue_knowledge import retrieve_guidance

            topic = {
                "negative_event": "mood",
                "loneliness": "social",
                "dignity": "negative_cognition",
            }.get(state.support_preference_topic, state.current_topic)
            guidance = retrieve_guidance(topic, text, "moderate", max_results=1)
        except Exception:
            guidance = []
        if guidance:
            item = guidance[0]
            action = (
                f"{item['validation']} {item['action']} "
                f"{item['escalation']}"
            )
        else:
            action = (
                "可以先选一件十分钟内能完成的小事，或者联系一个信任的人。"
            )
        if preference == "both":
            return (
                "好，我先接住您的感受，再一起想办法。"
                f"{action} 您更愿意先从哪一步开始？"
            )
        return f"{action} 您更愿意先从哪一步开始？"

    @staticmethod
    def _handle_general_medical(
        state: InterviewState,
        text: str,
    ) -> str | None:
        try:
            from elderly_depression.dialogue_knowledge import (
                find_general_medical_guidance,
            )

            matches = find_general_medical_guidance(text, max_results=1)
        except Exception:
            matches = []
        if not matches:
            return None
        item = matches[0]
        state.general_medical_id = str(item["id"])
        state.general_medical_title = str(item["title"])
        question = str(item.get("question") or "")
        state.general_medical_question = question
        state.pending_slot = "general_medical_followup"
        state.pending_topic = state.current_topic
        response = (
            f"{item['validation']} {item['action']} {item['escalation']}"
        )
        return f"{response} {question}".strip()

    def _handle_general_medical_followup(
        self,
        state: InterviewState,
        text: str,
    ) -> str | None:
        compact = re.sub(r"\s+", "", text)
        if self._mentioned_topics(text) or self._professional_keyword_intent(text):
            state.pending_slot = None
            state.general_medical_title = ""
            state.general_medical_id = ""
            state.general_medical_question = ""
            return None
        state.pending_slot = None
        state.general_medical_id = ""
        state.general_medical_title = ""
        state.general_medical_question = ""
        return (
            "这些信息我记下了。先按前面说的记录症状和数值；"
            "如果出现胸痛、呼吸困难、意识变化、晕厥、便血或完全尿不出，请及时就医。"
        )

    def _handle_clinical_mood_slot(
        self,
        state: InterviewState,
        text: str,
        analysis: UtteranceAnalysis,
    ) -> str | None:
        from elderly_depression.clinical_interview import (
            meets_two_week_threshold,
        )

        focus = analysis.focus_text or text
        compact = re.sub(r"\s+", "", focus)
        related_impact_answer = any(
            term in compact
            for term in [
                "影响",
                "不影响",
                "吃饭",
                "睡觉",
                "出门",
                "做事",
                "家务",
                "工作",
                "会",
                "不会",
            ]
        )
        mentioned = self._mentioned_topics(focus)
        if mentioned and "mood" not in mentioned and not related_impact_answer:
            state.pending_slot = None
            state.pending_topic = None
            return None

        impact = state.clinical_findings.get("functional_impact", "")
        if state.pending_slot == "clinical_mood_impact":
            polarity = _short_answer_polarity(focus)
            if not impact and polarity == "affirmative":
                impact = "有"
                state.clinical_findings["functional_impact"] = impact
            elif not impact and polarity == "negative":
                impact = "无"
                state.clinical_findings["functional_impact"] = impact
            if not impact:
                return (
                    "不用急着说得很细。最近这种情绪有没有让您少吃饭、睡不好，"
                    "或者连出门、做家务都比以前费劲？"
                )
            duration = state.clinical_findings.get("duration", "")
            if not duration:
                state.pending_slot = "clinical_mood_duration"
                if impact == "有":
                    return (
                        "这已经影响到日常生活，值得认真对待。"
                        "这种情况大概持续多久了？"
                    )
                return "明白，目前日常生活还能维持。这样的情绪大概持续多久了？"

        duration = state.clinical_findings.get("duration", "")
        state.pending_slot = None
        state.pending_topic = None
        if not duration:
            return (
                "持续时间暂时说不清也可以。先按您能回忆到的范围记录，"
                "如果这种状态达到两周或明显影响生活，建议尽快做专业评估。"
            )
        summary = state.clinical_summary or f"持续时间：{duration}"
        threshold = meets_two_week_threshold(duration)
        advice = (
            "持续达到两周以上，建议尽快联系医生或心理专业人员做专业评估。"
            if threshold
            else "如果继续持续或影响日常生活，建议尽快做专业评估。"
        )
        conclusion = f"我先记下：{summary}。{advice}"
        return conclusion

    def _handle_user_correction(
        self,
        state: InterviewState,
        text: str,
    ) -> str | None:
        compact = re.sub(r"\s+", "", text)
        explicit = any(phrase in compact for phrase in EXPLICIT_CORRECTION_PHRASES)
        positive_interest_correction = any(
            phrase in compact
            for phrase in [
                "并没有少点乐趣",
                "没少乐趣",
                "并没有少乐趣",
                "还是挺喜欢",
                "还是有兴趣",
                "照样喜欢",
            ]
        )
        strong_denial = any(
            phrase in compact
            for phrase in ["并没有觉得", "并没有说", "没有说", "我不是说"]
        )
        last_reply_is_question = state.last_reply.rstrip().endswith(("？", "?"))
        denied_assumption = (
            any(term in compact for term in ["并没有", "没有", "不是", "没少"])
            and any(
                term in compact and term in state.last_reply
                for term in ASSUMPTION_TERMS
            )
            and (strong_denial or not last_reply_is_question)
        ) or positive_interest_correction
        if not explicit and not denied_assumption:
            return None

        previous_focus = state.previous_focus_text
        current_analysis = analyze_utterance(text)
        current_focus = current_analysis.focus_text or text
        current_topics = self._mentioned_topics(current_focus)
        prior_topics = self._mentioned_topics(previous_focus)
        target_topic = (
            current_topics[0]
            if current_topics
            else (prior_topics[0] if prior_topics else "")
        )
        if current_topics and current_focus:
            state.current_topic = target_topic
            state.pending_topic = target_topic
            state.pending_slot = f"core_{target_topic}"
            questions = {
                "sleep": "是入睡困难、半夜容易醒，还是醒得太早？",
                "interest": "最近是提不起兴趣，还是身体不方便？",
                "mood": "最近这种感受大概持续多久了？",
                "social": "最近最想找人说话是在什么时候？",
                "cognition": "更明显的是忘事，还是注意力难以集中？",
                "somatic": "是吃不下、胃里不舒服，还是整个人没力气？",
                "negative_cognition": "这种想法最近大概多久出现一次？",
            }
            return (
                f"明白了，是我刚才理解偏了。您真正说的是“{current_focus}”。"
                f"我们按这个来。{questions.get(target_topic, '您愿意再具体说说吗？')}"
            )
        if any(
            term in compact
            for term in ["乐趣", "兴趣", "没意思", "开心"]
        ):
            state.covered.add("interest")
            state.risk_cues = [
                cue for cue in state.risk_cues if cue != "兴趣减退"
            ]
            state.addressed_issues.add("interest_positive")
            if "兴趣保持" not in state.protective_factors:
                state.protective_factors.append("兴趣保持")
        if target_topic in VALID_TOPIC_KEYS:
            state.current_topic = target_topic
            state.pending_topic = target_topic
            from elderly_depression.dialogue_policy import (
                is_situational_late_sleep,
            )

            if target_topic == "sleep" and is_situational_late_sleep(previous_focus):
                state.pending_slot = "sleep_late_pattern"
                return (
                    "明白了，是我刚才理解偏了，您并没有说兴趣或乐趣减少。"
                    f"我们回到刚才的情况：{previous_focus}。"
                    "这次晚睡后第二天精神怎么样？"
                )
            state.pending_slot = f"core_{target_topic}"
            return (
                "明白了，是我刚才理解偏了，不该把这句话当成那个结论。"
                "谢谢您纠正，我们按您的原意继续。"
            )
        return (
            "明白了，是我理解偏了，刚才那个判断不成立。"
            "您愿意再说说，真正想让我记住的是哪一点？"
        )

    @staticmethod
    def _medical_boundary_reply(text: str) -> str | None:
        compact = re.sub(r"\s+", "", text)
        doctor_directed = any(
            phrase in compact
            for phrase in ["医生说", "医生让", "医嘱", "药师说", "药师建议"]
        )
        if (
            any(pattern in compact for pattern in MEDICATION_CHANGE_PATTERNS)
            and not doctor_directed
        ):
            return (
                "我不能替您决定停药、换药或调整剂量，自行改动可能有风险。"
                "请先联系开药医生或药师确认；如果出现胸痛、呼吸困难、意识不清等急症，请立即拨打 120。"
                "您愿意告诉我现在吃的是什么药、为什么想调整吗？"
            )
        if any(pattern in compact for pattern in DIAGNOSIS_REQUEST_PATTERNS):
            return (
                "仅凭聊天不能确诊抑郁症或痴呆，您的感受仍然值得认真评估，"
                "但需要医生结合问诊、量表和其他检查判断。"
                "我可以先帮您整理持续时间和对生活的影响，也建议预约专业评估。最近这些变化持续多久了？"
            )
        return None

    def _handle_pending_slot(
        self,
        state: InterviewState,
        text: str,
        mode: str,
        analysis: UtteranceAnalysis,
    ) -> str | None:
        if not state.pending_slot:
            return None
        if state.pending_slot == "safety_followup":
            return self._handle_safety_followup(state, text)
        if state.pending_slot == "help_triage":
            return self._handle_help_triage(state, text)
        if state.pending_slot.startswith("clinical_mood_"):
            return self._handle_clinical_mood_slot(state, text, analysis)
        if state.pending_slot == "support_preference":
            return self._handle_support_preference(state, text)
        if state.pending_slot == "general_medical_followup":
            return self._handle_general_medical_followup(state, text)
        if mode != "interview":
            return None
        if state.pending_slot.startswith("generic_followup_"):
            return self._handle_generic_followup(state, text, analysis)
        if state.pending_slot in SLEEP_EXERCISE_PENDING_SLOTS:
            return self._handle_sleep_exercise_slot(state, text, analysis)
        if state.pending_slot.startswith("professional_"):
            return self._handle_professional_slot(state, text, analysis)
        if state.pending_slot.startswith("core_"):
            return self._handle_core_slot(state, text, analysis)
        if state.pending_slot in {
            "sleep_schedule",
            "sleep_quality",
            "sleep_impact",
            "sleep_late_pattern",
        }:
            focus = analysis.focus_text or text
            mentioned = self._mentioned_topics(focus)
            professional = self._professional_keyword_intent(focus)
            if professional or (mentioned and "sleep" not in mentioned):
                state.pending_slot = None
                state.pending_topic = None
                return None
        if state.pending_slot == "alliance_repair":
            compact = re.sub(r"\s+", "", text)
            topic = state.pending_topic or state.current_topic
            state.pending_slot = None
            state.pending_topic = None
            if any(word in compact for word in ["换", "轻松", "别的"]):
                state.covered.add(topic)
                next_topic = self._next_uncovered(state, after=topic)
                return (
                    self._ask_topic(state, next_topic)
                    if next_topic
                    else "好，我们先换个轻松话题。您今天想聊点什么？"
                )
            if any(word in compact for word in ["停", "安静", "不想", "算了"]):
                return "好，我们先停一停，不继续问了。您想说的时候再告诉我。"
            if any(word in compact for word in ["继续", "接着", "好", "可以"]):
                state.pending_topic = topic
                state.pending_slot = f"core_{topic}"
                return "好，我们慢一点来，不急着一次说清楚。您接着说就好。"
            return None
        if state.pending_slot == "sleep_late_pattern":
            focus = analysis.focus_text or text
            if (
                "sleep-exercise-cortisol" in state.last_guidance_ids
                or "皮质醇" in focus
            ):
                from elderly_depression.dialogue_policy import (
                    build_sleep_response,
                    classify_answer_archetype,
                )

                combined = (
                    f"{state.previous_focus_text}，{focus}"
                    if state.previous_focus_text
                    else focus
                )
                response = build_sleep_response(
                    "sleep",
                    combined,
                    classify_answer_archetype(combined),
                    0,
                    focus_text=combined,
                    positive_context=combined,
                )
                if response:
                    state.pending_slot = None
                    state.pending_topic = None
                    state.current_topic = "sleep"
                    state.covered.add("sleep")
                    state.addressed_issues.add("sleep_rhythm")
                    self._mark_sleep_exercise_timing(state)
                    return response
            mentioned = self._mentioned_topics(focus)
            if mentioned and "sleep" not in mentioned:
                state.pending_slot = None
                state.pending_topic = None
                return None
            negative = has_any_unnegated_keyword(
                focus,
                [
                    "没精神",
                    "白天困",
                    "补觉",
                    "很累",
                    "头疼",
                    "影响",
                    "两三次",
                    "三次",
                    "四次",
                    "每天",
                ],
            )
            positive = has_any_unnegated_keyword(
                focus,
                ["没事", "不影响", "精神不错", "精神还行", "睡够了"],
            )
            state.pending_slot = None
            state.pending_topic = None
            state.addressed_issues.add("sleep_late_pattern")
            next_topic = self._next_uncovered(state, after="sleep")
            if negative and not positive:
                conclusion = (
                    "如果晚睡后第二天明显没精神，或者一周出现两三次，"
                    "就按作息偏移来调整：先固定起床时间、减少白天补觉，"
                    "再逐步把上床时间提前。"
                )
            elif positive:
                conclusion = (
                    "偶尔一次晚睡、第二天精神也不受影响，可以先观察，"
                    "不用把这一次直接当成睡眠问题。"
                )
            else:
                conclusion = (
                    "偶发晚睡先不用紧张，重点看频率和第二天是否明显没精神。"
                    "连续出现时再固定起床时间并减少白天补觉。"
                )
            return (
                f"{conclusion}\n\n{self._ask_topic(state, next_topic)}"
                if next_topic
                else conclusion
            )
        if state.pending_slot == "sleep_schedule":
            times = _parse_clock_times(text)
            if len(times) < 2:
                focus = analysis.focus_text or text
                wants_schedule_help = (
                    analysis.requests_help
                    or analysis.is_correction
                    or any(
                        phrase in focus
                        for phrase in ["调整作息", "作息困难", "作息有困难"]
                    )
                )
                if wants_schedule_help:
                    state.pending_slot = None
                    state.covered.add("sleep")
                    state.suggested_topics.add("sleep")
                    state.addressed_issues.add("sleep_rhythm")
                    return (
                        "明白，您睡眠本身还可以，真正想解决的是作息调整。"
                        "可以先固定起床时间，起床后见见光，白天午睡不超过半小时；"
                        "每两三天把上床时间提前十五分钟。先试一周，"
                        "不用要求自己一次就调整到位。"
                    )
                return "我还没听清具体时间。晚上大概几点上床，早上一般几点起？"
            bedtime, wake = times[0], times[1]
            if wake <= bedtime:
                wake += 24
            duration = wake - bedtime
            wake_display = wake % 24
            state.pending_slot = "sleep_quality"
            if 7 <= duration <= 10:
                return (
                    f"从{bedtime}点到{wake_display}点，大约{duration}小时，睡眠时长不算少。"
                    + ("自然醒也说明睡眠量还可以。" if "自然醒" in text else "")
                    + "现在更主要的可能是入睡时间偏晚。夜里会不会醒，白天精神怎么样？"
                )
            if duration < 7:
                return (
                    f"按这个时间算，大约{duration}小时，可能偏少。"
                    "夜里会不会醒，白天精神怎么样？"
                )
            return (
                f"按这个时间算，大约{duration}小时，睡眠时间偏长。"
                "夜里睡得踏实吗，白天精神怎么样？"
            )
        if state.pending_slot == "sleep_quality":
            focus = analysis.focus_text or text
            negative = has_any_unnegated_keyword(
                focus,
                ["会醒", "醒好几次", "容易醒", "睡不踏实", "没睡好", "做梦多", "白天累", "没精神", "困"],
            )
            positive = has_any_unnegated_keyword(
                focus,
                [
                    "不醒",
                    "不太醒",
                    "没醒",
                    "睡得踏实",
                    "睡得好",
                    "白天还行",
                    "白天也还行",
                    "白天还可以",
                    "白天有精神",
                    "精神不错",
                ],
            )
            if negative and not positive:
                state.pending_slot = "sleep_impact"
                return "夜里容易醒会打断睡眠。白天会不会更累，或者做事没精神？"
            state.covered.add("sleep")
            state.pending_slot = None
            state.addressed_issues.add("sleep_quality")
            next_topic = self._next_uncovered(state, after="sleep")
            if positive and "sleep_rhythm" not in state.addressed_issues:
                conclusion = (
                    "这样看，睡眠和白天精神都还可以，先保持现在的作息就好。"
                )
            elif positive:
                conclusion = (
                    "这样看，睡眠时长和质量还可以，主要需要调整的是入睡时间。"
                    "可以先把起床时间固定下来，起床后见见光，白天午睡尽量不超过半小时；"
                    "每两三天把上床时间提前十五分钟，不用一次提前太多。"
                )
                state.suggested_topics.add("sleep")
            else:
                conclusion = (
                    "睡眠情况我先记下了。可以先固定起床时间、减少白天补觉，"
                    "并观察一两周的变化。"
                )
                state.suggested_topics.add("sleep")
            return (
                f"{conclusion}\n\n{self._ask_topic(state, next_topic)}"
                if next_topic
                else f"{conclusion}\n\n{CLOSING}"
            )
        if state.pending_slot == "sleep_impact":
            state.covered.add("sleep")
            state.pending_slot = None
            state.addressed_issues.add("sleep_impact")
            state.suggested_topics.add("sleep")
            next_topic = self._next_uncovered(state, after="sleep")
            conclusion = (
                "如果夜里醒得多，白天又累，睡眠确实需要认真调整。"
                "可以先固定起床时间、减少白天补觉，睡前少喝浓茶咖啡；"
                "如果持续影响白天，建议到睡眠门诊评估。"
            )
            return (
                f"{conclusion}\n\n{self._ask_topic(state, next_topic)}"
                if next_topic
                else f"{conclusion}\n\n{CLOSING}"
            )
        return None

    def _handle_generic_followup(
        self,
        state: InterviewState,
        text: str,
        analysis: UtteranceAnalysis,
    ) -> str | None:
        topic = (state.pending_slot or "").removeprefix("generic_followup_")
        focus = analysis.focus_text or text
        mentioned = self._mentioned_topics(focus)
        professional = self._professional_keyword_intent(focus)
        if professional:
            if PROFESSIONAL_PARENT_TOPICS.get(professional) != topic:
                state.pending_slot = None
                state.pending_topic = None
                return None
        elif mentioned and topic not in mentioned:
            state.pending_slot = None
            state.pending_topic = None
            return None

        compact = re.sub(r"\s+", "", text)
        if any(
            phrase in compact
            for phrase in ["说不清", "不知道", "不确定", "不一定"]
        ):
            attempt_key = f"generic_followup_{topic}"
            attempts = state.attempts.get(attempt_key, 0)
            state.attempts[attempt_key] = attempts + 1
            if attempts < 1:
                return (
                    "没关系，不用一次说得特别准确，给个大概范围就可以。"
                    + PROGRESS_QUESTIONS.get(topic, "您愿意再具体说一点吗？")
                )

        state.pending_slot = None
        state.pending_topic = None
        state.covered.add(topic)
        state.addressed_issues.add(topic)
        summary = state.clinical_summary
        prefix = f"我记下了：{summary}。" if summary else "这一点我记下了。"
        action = ""
        try:
            from elderly_depression.dialogue_knowledge import retrieve_guidance
            from elderly_depression.dialogue_policy import (
                classify_answer_archetype,
            )

            items = retrieve_guidance(
                topic,
                focus,
                classify_answer_archetype(focus),
                max_results=1,
            )
            if items:
                action = str(items[0].get("action") or "").strip()
        except Exception:
            action = ""
        if not action:
            action = (
                "可以先记录变化出现的时间、诱因和对日常活动的影响，"
                "方便后续判断。"
            )
        next_topic = self._next_uncovered(state, after=topic)
        if next_topic:
            return (
                f"{prefix}{action}\n\n"
                f"{self._ask_progress_topic(state, next_topic.key)}"
            )
        return (
            f"{prefix}{action}\n\n"
            "聊到这里，您最希望先处理的是哪一部分？"
        )

    def _handle_sleep_exercise_slot(
        self,
        state: InterviewState,
        text: str,
        analysis: UtteranceAnalysis,
    ) -> str | None:
        focus = analysis.focus_text or text
        mentioned = self._mentioned_topics(focus)
        professional = self._professional_keyword_intent(focus)
        if professional or (mentioned and "sleep" not in mentioned):
            state.pending_slot = None
            state.pending_topic = None
            return None

        if state.pending_slot == "sleep_exercise_timing":
            times = _parse_clock_times(text)
            if len(times) >= 2:
                exercise_hour = _normalise_evening_hour(times[0])
                bedtime_hour = _normalise_evening_hour(times[1])
                return self._build_sleep_exercise_interval_reply(
                    state,
                    exercise_hour,
                    bedtime_hour,
                )
            if not times:
                compact = re.sub(r"\s+", "", text)
                unknown_time = any(
                    term in compact
                    for term in [
                        "不知道",
                        "说不清",
                        "不确定",
                        "没准",
                        "不一定",
                        "记不清",
                    ]
                )
                if unknown_time:
                    attempt_key = "sleep_exercise_timing"
                    attempts = state.attempts.get(attempt_key, 0)
                    state.attempts[attempt_key] = attempts + 1
                    if attempts < 1:
                        return (
                            "没关系，可以先说个大概，比如九点或十点。"
                            "您通常晚上几点结束健身？"
                        )
                    state.pending_slot = "sleep_quality"
                    state.pending_topic = "sleep"
                    state.addressed_issues.add("sleep_exercise_observation")
                    return (
                        "那先不用估时间，连续记三天健身结束时间、上床时间和第二天精神就够。"
                        "如果三天的间隔都不到两小时，再把健身提前或降低晚间强度。"
                        "您白天现在会不会明显犯困？"
                    )
                if self._is_sleep_exercise_context(state, text):
                    return (
                        "运动后皮质醇短时升高是常见反应，不能只凭这个判断异常。"
                        "我们先只看间隔：您通常晚上几点结束健身？"
                    )
                return (
                    "我还没听清具体时间。您可以说“九点”或“九点半”，"
                    "大概是晚上几点结束健身？"
                )
            exercise_hour = _normalise_evening_hour(times[0])
            display = _format_clock_hour(exercise_hour)
            self._set_clinical_finding(state, "exercise_end", display)
            state.pending_topic = "sleep"
            state.pending_slot = "sleep_exercise_bedtime"
            return f"如果健身在{display}结束，您一般几点上床？"

        if state.pending_slot == "sleep_exercise_bedtime":
            times = _parse_clock_times(text)
            if not times:
                compact = re.sub(r"\s+", "", text)
                if any(
                    term in compact
                    for term in ["不知道", "说不清", "不确定", "没准", "记不清"]
                ):
                    state.pending_slot = "sleep_quality"
                    state.pending_topic = "sleep"
                    state.addressed_issues.add("sleep_exercise_observation")
                    return (
                        "没关系，那先连续记三天上床时间，不用要求精确到分钟。"
                        "如果健身结束到上床不到两小时，再考虑提前或降低晚间强度。"
                        "您白天现在会不会明显犯困？"
                    )
                return (
                    "上床时间我还没对上。您可以说“十一点”或“十一点半”，"
                    "一般大概几点上床？"
                )
            exercise_display = state.clinical_findings.get("exercise_end", "")
            exercise_times = _parse_clock_times(exercise_display)
            if not exercise_times:
                state.pending_slot = "sleep_exercise_timing"
                state.pending_topic = "sleep"
                return "刚才的健身时间我没记清，您通常晚上几点结束健身？"
            exercise_hour = _normalise_evening_hour(exercise_times[0])
            bedtime_hour = _normalise_evening_hour(times[0])
            return self._build_sleep_exercise_interval_reply(
                state,
                exercise_hour,
                bedtime_hour,
            )

        if state.pending_slot == "sleep_exercise_plan_choice":
            compact = re.sub(r"\s+", "", text)
            advance = any(term in compact for term in ["提前", "早点", "前移"])
            reduce_intensity = any(
                term in compact
                for term in ["降低强度", "强度低", "轻松", "拉伸", "慢走"]
            )
            keep = any(
                term in compact
                for term in ["保持", "不用改", "不想改", "维持"]
            )
            if advance and reduce_intensity:
                plan = (
                    "那就先把健身整体提前三十分钟，并把晚间最后一组高强度动作减下来；"
                    "连续记录一周，重点看入睡是否更顺。"
                )
            elif advance:
                plan = (
                    "那就分步来：先把健身结束时间提前三十分钟，"
                    "最终尽量留出上床前三小时；连续记录一周，再看入睡变化。"
                )
            elif reduce_intensity:
                plan = (
                    "晚间健身可以保留，但把高强度部分放在前面，"
                    "最后改成低强度拉伸或慢走，连续观察一周。"
                )
            elif keep:
                plan = (
                    "可以先保持现在的运动安排，重点固定起床时间、"
                    "睡前减少屏幕和浓茶咖啡，同时记录一周的入睡变化。"
                )
            else:
                plan = (
                    "不用一次改很多，先从把健身提前三十分钟开始，"
                    "同时把晚间最后一段强度降下来，观察一周。"
                )
            state.pending_slot = None
            state.pending_topic = None
            state.covered.add("sleep")
            state.current_topic = "sleep"
            state.addressed_issues.add("sleep_exercise_routine")
            state.accepted_actions.append(plan)
            state.user_facts["睡眠行动计划"] = plan
            next_topic = self._next_uncovered(state, after="sleep")
            return (
                f"{plan}\n\n{self._ask_topic(state, next_topic)}"
                if next_topic
                else plan
            )
        return None

    def _build_sleep_exercise_interval_reply(
        self,
        state: InterviewState,
        exercise_hour: int,
        bedtime_hour: int,
    ) -> str:
        if bedtime_hour <= exercise_hour:
            bedtime_hour += 24
        interval = bedtime_hour - exercise_hour
        exercise_display = _format_clock_hour(exercise_hour)
        bedtime_display = _format_clock_hour(bedtime_hour)
        self._set_clinical_finding(state, "exercise_end", exercise_display)
        self._set_clinical_finding(state, "sleep_time", bedtime_display)
        state.current_topic = "sleep"
        state.pending_topic = "sleep"
        state.pending_slot = "sleep_exercise_plan_choice"
        state.addressed_issues.add("sleep_rhythm")

        lead = (
            f"从{exercise_display}健身结束到{bedtime_display}上床，"
            f"大约间隔{interval}小时。"
        )
        if interval >= 3:
            assessment = (
                "这个间隔基本合适，不必再为皮质醇过分担心。"
                "如果仍然入睡晚，先把固定起床和睡前放松做好。"
            )
        elif interval >= 2:
            assessment = (
                "这个间隔稍近，运动后的兴奋可能让身体没那么快进入睡眠状态。"
                "可以考虑提前一点，或把晚间强度降下来。"
            )
        else:
            assessment = (
                "这个间隔偏近，运动后的兴奋可能推迟入睡，但不等于皮质醇异常。"
                "先调整运动结束时间或晚间强度，会比只关注皮质醇更实用。"
            )
        return (
            f"{lead}{assessment}"
            "您更愿意把健身提前，还是晚间降低强度？"
        )

    def _handle_safety_followup(
        self,
        state: InterviewState,
        text: str,
    ) -> str:
        compact = re.sub(r"\s+", "", text)
        unsafe = any(
            phrase in compact
            for phrase in ["不安全", "控制不住", "没人", "一个人", "有计划", "准备了"]
        )
        safe = any(
            phrase in compact
            for phrase in ["安全", "有人陪", "家人陪", "朋友陪", "联系到", "在一起"]
        )
        state.pending_slot = None
        state.pending_topic = None
        if unsafe and not safe:
            state.safety_plan_status = "emergency_referral"
            return (
                "听到您现在可能不安全，我很担心。请立即拨打 120、110 或 12356，"
                "并请身边可信的人陪着您，先移开可能伤害自己的物品，不要独处。"
            )
        if safe:
            state.safety_plan_status = "support_confirmed"
            return (
                "好，您现在安全并有可联系的人，这很重要。请先不要独处，"
                "把家人、医生或 12356 放在方便联系的位置。"
                "今天尽快把刚才的感受告诉家人或专业人员，让他们陪您一起处理。"
            )
        state.pending_slot = "safety_followup"
        state.pending_topic = "safety"
        return (
            "我还是想先确认安全：您现在是否安全，身边有没有能马上联系到的人？"
        )

    def _handle_core_slot(
        self,
        state: InterviewState,
        text: str,
        analysis: UtteranceAnalysis,
    ) -> str | None:
        topic = state.pending_topic or (state.pending_slot or "").removeprefix("core_")
        focus = analysis.focus_text or text
        mentioned = self._mentioned_topics(focus)
        professional = self._professional_keyword_intent(focus)
        topic_detail_present = has_any_unnegated_keyword(
            focus,
            TOPIC_MAP.get(topic, []),
        )
        answer_polarity = _short_answer_polarity(focus)
        if topic == "interest" and answer_polarity == "affirmative":
            state.pending_slot = None
            state.pending_topic = None
            state.covered.add("interest")
            state.addressed_issues.add("interest_positive")
            if "兴趣保持" not in state.protective_factors:
                state.protective_factors.append("兴趣保持")
            next_topic = self._next_uncovered(state, after="interest")
            hobby = state.user_facts.get("爱好", "").strip()
            if hobby:
                conclusion = (
                    f"那就好，您现在还愿意继续{hobby}，说明这件事仍然能给您带来一些乐趣。"
                    "继续保持规律的爱好，对情绪是很好的支持。"
                )
            else:
                conclusion = (
                    "那就好，您现在还愿意做以前喜欢的事，说明这些活动仍然能给您带来一些乐趣。"
                    "继续保持规律的爱好，对情绪是很好的支持。"
                )
            return (
                f"{conclusion}\n\n{self._ask_topic(state, next_topic)}"
                if next_topic
                else conclusion
            )
        if topic == "interest" and answer_polarity == "negative":
            state.addressed_issues.add("interest_reduced")
            if "兴趣减退" not in state.risk_cues:
                state.risk_cues.append("兴趣减退")
            state.pending_slot = None
            state.pending_topic = None
            hobby = state.user_facts.get("爱好", "").strip()
            hobby_text = f"{hobby}现在" if hobby else "以前喜欢的事现在"
            return (
                f"{hobby_text}不太愿意做，确实会让人失落，也可能让生活变得没劲。"
                "先不用强迫自己一次做很多。您愿意的话，可以从十分钟的小活动开始，"
                "也让家人知道最近的变化；如果持续两周以上或影响日常生活，建议做专业评估。"
                "您觉得更多是提不起兴趣，还是身体不舒服、出门不方便？"
            )
        if topic == "interest" and answer_polarity == "partial":
            state.addressed_issues.add("interest_fluctuating")
            state.pending_topic = "interest"
            state.pending_slot = "core_interest"
            return (
                "有时愿意、有时提不起劲，这种变化本身也会让人困惑。"
                "通常是在身体累、心情低，还是一个人待着时更明显？"
            )
        if answer_polarity == "uncertain":
            state.pending_topic = topic
            state.pending_slot = f"core_{topic}"
            return self._uncertain_clarification(topic)
        if answer_polarity:
            contextual = self._build_contextual_polarity_reply(
                state,
                topic,
                answer_polarity,
            )
            if contextual:
                return contextual
        if topic == "sleep":
            from elderly_depression.dialogue_policy import (
                build_sleep_response,
                build_policy_response,
                classify_answer_archetype,
                is_situational_late_sleep,
            )

            if is_situational_late_sleep(text):
                response = build_sleep_response(
                    "sleep",
                    text,
                    classify_answer_archetype(text),
                    0,
                    focus_text=text,
                )
                if response:
                    state.pending_slot = "sleep_late_pattern"
                    state.pending_topic = "sleep"
                    state.current_topic = "sleep"
                    state.covered.add("sleep")
                    state.addressed_issues.add("sleep_rhythm")
                    return response

            if "sleep-exercise-cortisol" in state.last_guidance_ids or "皮质醇" in text:
                combined = (
                    f"{state.previous_focus_text}，{text}"
                    if state.previous_focus_text
                    else text
                )
                response = build_sleep_response(
                    "sleep",
                    combined,
                    classify_answer_archetype(combined),
                    0,
                    focus_text=combined,
                    positive_context=combined,
                )
                if response:
                    state.pending_slot = None
                    state.pending_topic = None
                    state.current_topic = "sleep"
                    state.covered.add("sleep")
                    state.addressed_issues.add("sleep_rhythm")
                    state.suggested_topics.add("sleep")
                    self._mark_sleep_exercise_timing(state)
                    return response

            answer_archetype = classify_answer_archetype(focus)
            if answer_archetype != "unknown":
                if answer_archetype == "positive" and "白天" in focus:
                    state.pending_slot = None
                    state.pending_topic = None
                    state.covered.add("sleep")
                    state.addressed_issues.add("sleep_quality")
                    next_topic = self._next_uncovered(state, after="sleep")
                    conclusion = (
                        "白天精神也还可以，说明这次睡眠没有明显影响白天的状态。"
                    )
                    return (
                        f"{conclusion}\n\n{self._ask_topic(state, next_topic)}"
                        if next_topic
                        else conclusion
                    )
                response = build_sleep_response(
                    "sleep",
                    text,
                    answer_archetype,
                    0,
                    focus_text=text,
                    positive_context=analysis.positive_context,
                ) or build_policy_response(
                    "sleep",
                    answer_archetype,
                    0,
                    focus_text=focus,
                    positive_context=analysis.positive_context,
                )
                if response:
                    state.covered.add("sleep")
                    state.addressed_issues.add("sleep")
                    state.pending_topic = "sleep"
                    state.pending_slot = (
                        "sleep_quality" if "白天" in response else None
                    )
                    return response

        if (
            analysis.has_focused_issue
            or topic_detail_present
            or (mentioned and topic not in mentioned)
            or (
                professional
                and PROFESSIONAL_PARENT_TOPICS.get(professional) != topic
            )
        ):
            state.pending_slot = None
            state.pending_topic = None
            return None

        state.pending_slot = None
        state.pending_topic = None
        state.addressed_issues.add(topic)
        conclusion = CORE_FOLLOWUP_CLOSERS.get(topic, "这一点我已经记下。")
        next_topic = self._next_uncovered(state, after=topic)
        if next_topic:
            return f"{conclusion}\n\n{self._ask_topic(state, next_topic)}"
        return conclusion

    def _build_contextual_polarity_reply(
        self,
        state: InterviewState,
        topic: str,
        polarity: str,
    ) -> str | None:
        prompt = state.last_reply
        affirmative = polarity in {"affirmative", "partial"}
        partial = polarity == "partial"
        if topic == "sleep":
            state.covered.add("sleep")
            state.addressed_issues.add("sleep")
            if affirmative:
                if "睡眠差" not in state.risk_cues:
                    state.risk_cues.append("睡眠差")
                state.pending_topic = "sleep"
                state.pending_slot = "core_sleep"
                if partial:
                    return (
                        "偶尔睡不好也会影响第二天的精神。"
                        "您愿意的话，可以先留意一周大概有几天，以及是不是在压力大或心里有事时更明显？"
                    )
                return (
                    "最近经常睡不好一定很熬人，白天也会跟着受影响。"
                    "您愿意的话，可以先固定早上起床时间、白天午睡不超过半小时，"
                    "睡前少喝浓茶咖啡，也不要在床上反复看时间。"
                    "如果持续两周以上或白天明显受影响，建议到睡眠门诊评估。"
                    "现在更明显的是入睡困难、半夜容易醒，还是醒得太早？"
                )
            next_topic = self._next_uncovered(state, after="sleep")
            conclusion = "睡眠目前没有明显困扰，先保持现在的作息就好。"
            return (
                f"{conclusion}\n\n{self._ask_topic(state, next_topic)}"
                if next_topic
                else conclusion
            )
        if topic == "mood" and any(
            term in prompt for term in ["心里堵", "难过", "掉眼泪", "情绪", "低落"]
        ):
            state.covered.add("mood")
            if affirmative:
                if "情绪低落" not in state.risk_cues:
                    state.risk_cues.append("情绪低落")
                state.pending_topic = "mood"
                state.pending_slot = "clinical_mood_impact"
                if partial:
                    return (
                        "偶尔出现这些情绪也会让人不好受，您愿意说出来很重要。"
                        "先不用责怪自己。如果您愿意，难受时可以给信任的人打个电话，别一个人扛着。"
                        "这种情绪出现时，会影响吃饭、睡觉或出门吗？"
                    )
                return STRONG_CARE_RESPONSES["mood"]
            state.addressed_issues.add("mood_positive")
            next_topic = self._next_uncovered(state, after="mood")
            conclusion = "最近情绪没有明显低落，这是很好的保护因素。"
            return (
                f"{conclusion}\n\n{self._ask_topic(state, next_topic)}"
                if next_topic
                else conclusion
            )
        if topic == "cognition" and any(
            term in prompt for term in ["记性", "忘事", "注意力", "记不住"]
        ):
            state.covered.add("cognition")
            if affirmative:
                if "认知下降" not in state.risk_cues:
                    state.risk_cues.append("认知下降")
                state.pending_topic = "cognition"
                state.pending_slot = "core_cognition"
                if partial:
                    return (
                        "偶尔忘事会让人有点担心，不过偶尔发生并不等于记性出了大问题。"
                        "您愿意的话，可以先留意是在疲劳、着急还是同时做几件事时更容易出现？"
                    )
                return STRONG_CARE_RESPONSES["cognition"]
            state.addressed_issues.add("cognition_positive")
            next_topic = self._next_uncovered(state, after="cognition")
            conclusion = "记性和注意力目前没有明显变化，可以继续观察。"
            return (
                f"{conclusion}\n\n{self._ask_topic(state, next_topic)}"
                if next_topic
                else conclusion
            )
        if topic == "negative_cognition" and any(
            term in prompt for term in ["没盼头", "负担", "拖累", "没用"]
        ):
            if affirmative:
                if "无望感" not in state.risk_cues:
                    state.risk_cues.append("无望感")
                state.pending_topic = "negative_cognition"
                state.pending_slot = "core_negative_cognition"
                if partial:
                    return (
                        "偶尔冒出这些想法也会很沉，您愿意说出来很不容易。"
                        "先别一个人反复想。您愿意的话，可以先告诉身边信任的人。"
                        "最近一次出现这种感受时，有人能陪您说说话吗？"
                    )
                return STRONG_CARE_RESPONSES["negative_cognition"]
            state.addressed_issues.add("negative_cognition_positive")
            next_topic = self._next_uncovered(
                state, after="negative_cognition"
            )
            conclusion = "最近没有明显的无望或自责感受，这是重要的保护因素。"
            return (
                f"{conclusion}\n\n{self._ask_topic(state, next_topic)}"
                if next_topic
                else conclusion
            )
        return None

    @staticmethod
    def _uncertain_clarification(topic: str) -> str:
        responses = {
            "mood": (
                "一时说不清很正常，不用急着给自己下结论。"
                "换个容易回答的：最近一周大概有几天会觉得心里不舒服？"
            ),
            "sleep": (
                "睡眠有时确实不好判断。可以先回想最近一周，"
                "大概有几天觉得没睡够或白天没精神？"
            ),
            "interest": (
                "可以慢慢想。最近一周有没有哪件以前喜欢的事，"
                "哪怕只做了一小会儿？"
            ),
            "social": (
                "不用急着定义孤单。最近一周有没有和熟悉的人见过面或通过电话？"
            ),
            "cognition": (
                "一时难判断也没关系。和以前比，最近有没有更常忘事或找不到东西？"
            ),
            "somatic": (
                "可以先说最具体的一点：最近有没有哪个身体部位持续不舒服？"
            ),
            "negative_cognition": (
                "可以慢慢想。最近有没有哪一刻觉得自己不被需要或成了负担？"
            ),
        }
        return responses.get(
            topic,
            "不用急着给准确答案，您可以想到哪里说到哪里。",
        )

        state.pending_slot = None
        state.pending_topic = None
        state.addressed_issues.add(topic)
        conclusion = CORE_FOLLOWUP_CLOSERS.get(topic, "这一点我已经记下。")
        next_topic = self._next_uncovered(state, after=topic)
        if next_topic:
            return f"{conclusion}\n\n{self._ask_topic(state, next_topic)}"
        return conclusion

    def _handle_professional_slot(
        self,
        state: InterviewState,
        text: str,
        analysis: UtteranceAnalysis,
    ) -> str | None:
        slot = state.pending_slot or ""
        topic = state.pending_topic or slot.removeprefix("professional_")
        if topic not in PROFESSIONAL_PARENT_TOPICS:
            state.pending_slot = None
            state.pending_topic = None
            return None

        focus = analysis.focus_text or text
        compact = re.sub(r"\s+", "", focus)
        other_professional = self._professional_keyword_intent(compact)
        core_topics = self._mentioned_topics(compact)
        expected_terms = PROFESSIONAL_FOLLOWUP_KEYWORDS.get(topic, [])
        direct_match = any(term in compact for term in expected_terms) or bool(
            re.search(r"\d", compact)
        )
        relevant = direct_match or len(compact) <= 4
        if (
            (other_professional and other_professional != topic)
            or (core_topics and not direct_match)
            or (not direct_match and len(compact) > 4)
        ):
            state.pending_slot = None
            state.pending_topic = None
            return None
        if not relevant:
            state.pending_slot = None
            state.pending_topic = None
            return None

        state.pending_slot = None
        state.pending_topic = None
        state.addressed_issues.add(topic)
        closer_key = (
            "medication_side_effect"
            if slot.endswith("medication_side_effect")
            else topic
        )
        conclusion = PROFESSIONAL_FOLLOWUP_CLOSERS[closer_key]
        next_topic = self._next_uncovered(state, after=str(state.current_topic))
        if next_topic:
            return f"{conclusion}\n\n{self._ask_topic(state, next_topic)}"
        return conclusion

    def _model_fast_reply(
        self,
        state: InterviewState,
        text: str,
        mode: str,
        analysis: UtteranceAnalysis,
    ) -> str | None:
        focus = analysis.focus_text or text
        prediction = self._intent_prediction(focus, mode)
        if prediction is None and focus != text:
            prediction = self._intent_prediction(text, mode)
        intent = ""
        if mode == "interview":
            intent = self._professional_keyword_intent(focus)
            if not intent and focus != text:
                intent = self._professional_keyword_intent(text)
            if (
                not intent
                and prediction
                and prediction.get("fastpath_allowed", True)
                and prediction.get("confidence", 0) >= INTENT_FASTPATH_THRESHOLD
            ):
                intent = str(prediction.get("intent") or "")
        elif (
            prediction
            and prediction.get("fastpath_allowed", True)
            and prediction.get("confidence", 0) >= INTENT_FASTPATH_THRESHOLD
        ):
            intent = str(prediction.get("intent") or "")
        if mode == "companion" and not intent:
            intent = self._companion_keyword_intent(focus)
        if mode == "companion" and not intent:
            try:
                from elderly_depression.dialogue_policy import classify_answer_archetype
                from elderly_depression.dialogue_responses import (
                    get_companion_archetype_response,
                )

                if classify_answer_archetype(focus) == "avoidant":
                    return get_companion_archetype_response("", "avoidant")
            except Exception:
                pass
        if not intent:
            return None
        extended_topics = PROFESSIONAL_PARENT_TOPICS
        if mode == "interview" and intent not in extended_topics:
            return None
        try:
            from elderly_depression.dialogue_policy import (
                build_policy_response,
                classify_answer_archetype,
            )
            from elderly_depression.dialogue_responses import (
                get_companion_archetype_response,
                get_fast_response,
            )

            if mode == "interview":
                archetype = classify_answer_archetype(focus)
                if archetype == "unknown":
                    archetype = "mild"
                response = build_policy_response(
                    intent,
                    archetype,
                    state.turn_count,
                    focus_text=focus,
                    positive_context=analysis.positive_context,
                )
            elif intent == "health":
                response = self._body_part_response(focus)
                if not response and any(
                    word in focus
                    for word in ["不舒服", "难受", "疼", "累", "没力气"]
                ):
                    response = (
                        "听您说身体不舒服，我有点担心。"
                        "是哪里不舒服，还是整个人没力气？"
                    )
            else:
                response = get_companion_archetype_response(
                    intent,
                    classify_answer_archetype(focus),
                )
            if not response:
                response = get_fast_response(mode, intent)
        except Exception:
            return None
        if not response:
            return None
        if mode == "interview":
            topic = extended_topics.get(intent)
            if topic in VALID_TOPIC_KEYS:
                state.current_topic = topic
            state.addressed_issues.add(intent)
            state.pending_topic = intent
            if intent == "medication" and has_any_unnegated_keyword(
                focus,
                ["头晕", "恶心", "心慌", "皮疹", "副作用", "难受"],
            ):
                state.pending_slot = "professional_medication_side_effect"
            else:
                state.pending_slot = f"professional_{intent}"
            cue = PROFESSIONAL_CUE_MAP.get(intent)
            if cue and cue not in state.risk_cues:
                state.risk_cues.append(cue)
        return response

    @staticmethod
    def _professional_keyword_intent(text: str) -> str:
        compact = re.sub(r"\s+", "", text)
        for intent, keywords in PROFESSIONAL_KEYWORDS.items():
            if has_any_unnegated_keyword(compact, keywords):
                return intent
        return ""

    @staticmethod
    def _companion_keyword_intent(text: str) -> str:
        compact = re.sub(r"\s+", "", text)
        if any(word in compact for word in ["过节", "春节", "中秋", "端午", "年货", "团聚"]):
            return "festival"
        if has_any_unnegated_keyword(
            compact, COMPANION_KEYWORDS["dignity"]
        ):
            return "dignity"
        if has_any_unnegated_keyword(
            compact, COMPANION_KEYWORDS["life_review"]
        ):
            return "life_review"
        if has_any_unnegated_keyword(compact, ["开心", "高兴", "好消息", "收到"]):
            return "positive_event"
        if has_any_unnegated_keyword(compact, BODY_PARTS) or has_any_unnegated_keyword(
            compact,
            ["身体", "睡不着", "没胃口", "不舒服", "疼", "累", "没力气", "头晕"],
        ):
            return "health"
        if any(word in compact for word in ["孩子", "儿子", "女儿", "孙子", "老伴", "家人"]):
            return "family"
        if any(word in compact for word in ["邻居", "楼下", "小区", "串门"]):
            return "neighbor"
        if any(word in compact for word in ["猫", "狗", "小鸟", "宠物"]):
            return "pet"
        if any(word in compact for word in ["电视", "新闻", "电影", "听歌", "老歌"]):
            return "media"
        if has_any_unnegated_keyword(
            compact, ["烦", "难受", "委屈", "不开心", "心情不好"]
        ):
            return "negative_event"
        for intent, keywords in COMPANION_KEYWORDS.items():
            if has_any_unnegated_keyword(compact, keywords):
                return intent
        return ""

    def _should_use_fastpath(self, mode: str, text: str, focus_text: str = "") -> bool:
        focus_compact = re.sub(r"\s+", "", focus_text or text)
        if AgentAdapter._is_greeting(text):
            return True
        if mode == "companion":
            return any(word in focus_compact for word in COMPANION_FASTPATH_WORDS)
        if AgentAdapter._mentioned_topics(focus_compact) or AgentAdapter._is_thin(text):
            return True
        prediction = self._intent_prediction(focus_compact, mode)
        return bool(
            prediction
            and prediction.get("fastpath_allowed", True)
            and prediction.get("confidence", 0) >= INTENT_FASTPATH_THRESHOLD
            and prediction.get("topic") in VALID_TOPIC_KEYS
        )

    def next_question(
        self,
        context: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> str:
        state = self._session(session_id, context)
        if (context or {}).get("mode") == "companion":
            if state.safety_flagged:
                return "您现在感觉安全吗？有没有可以马上联系到的家人或医生？"
            return "今天有没有什么开心的小事，或者想跟我聊聊的？"
        if state.safety_flagged:
            return "您现在感觉安全吗？有没有可以马上联系到的家人或医生？"
        if state.llm_next_question and not (context or {}).get("mode") == "companion":
            return state.llm_next_question
        for topic in TOPICS:
            if topic.key not in state.covered:
                if state.attempts.get(topic.key, 0) < len(topic.followups):
                    return topic.followups[state.attempts.get(topic.key, 0)]
                return topic.opening
        return "您最希望先解决、或者最让您放心不下的是哪一件事？"

    def snapshot(
        self,
        context: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        state = self._session(session_id, context)
        evidence: dict[int, set[str]] = {}
        for cue in state.risk_cues:
            for item in PHQ9_EVIDENCE_MAP.get(cue, []):
                evidence.setdefault(item, set()).add(cue)
        scale_evidence = [
            {
                "item": item,
                "label": PHQ9_ITEM_LABELS.get(item, ""),
                "cues": sorted(cues),
                "status": "需量表确认",
            }
            for item, cues in sorted(evidence.items())
        ]
        return {
            "mode": (context or {}).get("mode", "interview"),
            "topic": state.current_topic,
            "covered_topics": sorted(state.covered),
            "risk_cues": list(state.risk_cues),
            "cue_confidence": dict(state.cue_confidence),
            "next_question": state.llm_next_question,
            "safety_alert": state.safety_flagged,
            "engine": self.last_engine,
            "llm_error": self.last_llm_error,
            "memory": dict(state.user_facts),
            "memory_summary": state.memory_summary,
            "latency_ms": self.last_latency_ms,
            "fastpath_reason": self.last_fastpath_reason,
            "turn_count": state.turn_count,
            "clinical_alert": state.clinical_alert,
            "scale_evidence": scale_evidence,
            "focus": state.last_focus_text,
            "focus_reason": state.last_focus_reason,
            "positive_context": state.last_positive_context,
            "addressed_issues": sorted(state.addressed_issues),
            "suggested_topics": sorted(state.suggested_topics),
            "protective_factors": list(state.protective_factors),
            "alliance_repair_count": state.alliance_repair_count,
            "last_alliance_repair": state.last_alliance_repair,
            "accepted_actions": list(state.accepted_actions),
            "last_guidance_ids": list(state.last_guidance_ids),
            "last_guidance_titles": list(state.last_guidance_titles),
            "safety_plan_status": state.safety_plan_status,
            "clinical_findings": dict(state.clinical_findings),
            "clinical_summary": state.clinical_summary,
            "support_preference": state.support_preference,
            "general_medical_id": state.general_medical_id,
            "general_medical_title": state.general_medical_title,
            "general_medical_question": state.general_medical_question,
            "medical_sources": [
                {
                    "id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "title_zh": item.get("title_zh", ""),
                    "organization": item.get("organization", ""),
                    "url": item.get("url", ""),
                }
                for item in state.medical_kb_sources
            ],
        }

    def _record_cues(
        self,
        state: InterviewState,
        text: str,
        analysis: UtteranceAnalysis,
    ) -> None:
        focus = analysis.focus_text or text
        for label, words in CUE_RULES:
            if label == "保护因素":
                if (
                    has_any_unnegated_keyword(
                        f"{analysis.positive_context}{focus}",
                        words,
                    )
                    and label not in state.protective_factors
                ):
                    state.protective_factors.append(label)
                continue
            if label in state.risk_cues:
                continue
            if has_any_unnegated_keyword(focus, words):
                state.risk_cues.append(label)

    @staticmethod
    def _retrieve_medlineplus(
        state: InterviewState,
        text: str,
        analysis: UtteranceAnalysis,
    ) -> None:
        try:
            from elderly_depression.medical_kb import (
                format_medical_kb_for_prompt,
                search_medical_knowledge,
            )

            combined = (
                f"{state.previous_focus_text}，{analysis.focus_text or text}"
                if state.previous_focus_text
                else analysis.focus_text or text
            )
            items = search_medical_knowledge(combined, max_results=2)
            state.medical_kb_sources = items
            state.medical_kb_prompt = format_medical_kb_for_prompt(items)
        except Exception:
            state.medical_kb_sources = []
            state.medical_kb_prompt = ""

    @staticmethod
    def _update_clinical_findings(state: InterviewState, text: str) -> None:
        try:
            from elderly_depression.clinical_interview import (
                build_clinical_summary,
                update_clinical_findings,
            )

            state.clinical_findings = update_clinical_findings(
                state.clinical_findings,
                text,
            )
            state.clinical_summary = build_clinical_summary(
                state.clinical_findings
            )
        except Exception:
            pass

    @staticmethod
    def _set_clinical_finding(
        state: InterviewState,
        key: str,
        value: str,
    ) -> None:
        state.clinical_findings = {
            **state.clinical_findings,
            key: value,
        }
        try:
            from elderly_depression.clinical_interview import (
                build_clinical_summary,
            )

            state.clinical_summary = build_clinical_summary(
                state.clinical_findings
            )
        except Exception:
            pass

    @staticmethod
    def _is_sleep_exercise_context(
        state: InterviewState,
        text: str,
    ) -> bool:
        compact = re.sub(r"\s+", "", text)
        direct_cortisol = "皮质醇" in compact
        exercise = any(
            term in compact
            for term in ["健身", "锻炼", "运动"]
        )
        sleep_or_late = any(
            term in compact
            or term in re.sub(r"\s+", "", state.previous_focus_text)
            for term in ["睡", "晚", "迟", "入睡", "作息"]
        )
        return direct_cortisol or (exercise and sleep_or_late)

    @staticmethod
    def _mark_sleep_exercise_timing(state: InterviewState) -> None:
        state.current_topic = "sleep"
        state.pending_topic = "sleep"
        state.pending_slot = "sleep_exercise_timing"
        state.covered.add("sleep")
        state.addressed_issues.add("sleep_rhythm")
        state.suggested_topics.add("sleep")

    @staticmethod
    def _retrieve_guidance(
        state: InterviewState,
        text: str,
        analysis: UtteranceAnalysis,
    ) -> None:
        items = []
        formatted = ""
        try:
            from elderly_depression.dialogue_knowledge import (
                format_guidance_for_prompt,
                retrieve_guidance,
            )
            from elderly_depression.dialogue_policy import classify_answer_archetype

            focus = analysis.focus_text or text
            mentioned = AgentAdapter._mentioned_topics(focus)
            professional = AgentAdapter._professional_keyword_intent(text)
            topic = professional or (
                mentioned[0] if mentioned else state.current_topic
            )
            severity = classify_answer_archetype(focus)
            items = retrieve_guidance(topic, text, severity)
            formatted = format_guidance_for_prompt(items) if items else ""
        except Exception:
            items = []
            formatted = ""
        state.last_guidance_ids = [str(item["id"]) for item in items]
        state.last_guidance_titles = [str(item["title"]) for item in items]
        state.last_guidance_prompt = formatted

    @staticmethod
    def _extract_facts(state: InterviewState, text: str) -> None:
        patterns = {
            "姓名": r"(?:我叫|我的名字是)([\u4e00-\u9fa5A-Za-z0-9]{1,12})",
            "年龄": r"(?:我)?(?:今年)?(\d{2,3})岁",
            "居住地": r"(?:住在|家住)([^，。；\s]{2,20})",
            "爱好": r"(?:喜欢|爱好是|爱)([^，。；]{1,18})",
            "健康状况": r"(?:有|得了)([^，。；]{1,18}(?:病|高血压|糖尿病|心脏病|关节炎))",
            "近期目标": r"(?:最想先|想先|最希望|希望先|想解决|想改善)([^，。；]{2,24})",
        }
        for label, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip()
                if value:
                    state.user_facts[label] = value
        family = next(
            (word for word in ["老伴", "儿子", "女儿", "孙子", "孙女", "孩子"] if word in text),
            None,
        )
        if family:
            state.user_facts["家人"] = family
        action_match = re.search(
            r"(?:我愿意|我会|我打算|我准备)(?:先)?"
            r"((?:联系|记录|预约|试试|给|散步|睡觉|起床|告诉)[^，。；]{0,24})",
            text,
        )
        if action_match:
            action = action_match.group(1).strip()
            state.user_facts["行动计划"] = action
            if action not in state.accepted_actions:
                state.accepted_actions.append(action)
        while len(state.user_facts) > 8:
            state.user_facts.pop(next(iter(state.user_facts)))

    @staticmethod
    def _update_memory_summary(
        state: InterviewState,
        messages: list[dict[str, str]],
    ) -> None:
        older_user_texts = [
            str(item.get("text") or "")
            for item in messages[:-8]
            if item.get("role") == "user"
        ]
        if not older_user_texts:
            return
        joined = " ".join(older_user_texts)
        cues = [
            label
            for label, keywords in CUE_RULES
            if has_any_unnegated_keyword(joined, keywords)
        ]
        parts = []
        if cues:
            parts.append("早前提到：" + "、".join(cues[:5]))
        if state.covered:
            labels = [
                topic.label for topic in TOPICS if topic.key in state.covered
            ]
            if labels:
                parts.append("已聊过：" + "、".join(labels))
        if state.user_facts:
            parts.append(
                "已知信息：" + "、".join(state.user_facts.keys())
            )
        if parts:
            state.memory_summary = "；".join(parts)[:260]

    def reset_session(self, session_id: str | None = None, mode: str | None = None) -> int:
        sid = str(session_id or "default")
        if mode in {"interview", "companion"}:
            self._sessions.pop(f"{mode}:{sid}", None)
            return 1
        removed = 0
        for key in [f"interview:{sid}", f"companion:{sid}"]:
            if self._sessions.pop(key, None) is not None:
                removed += 1
        return removed

    @staticmethod
    def _contains_safety(text: str) -> bool:
        return has_any_unnegated_keyword(text, SAFETY_WORDS)

    @staticmethod
    def _mentioned_topics(text: str) -> list[str]:
        found: list[str] = []
        for key in TOPIC_PRIORITY:
            if key == "safety":
                if AgentAdapter._contains_safety(text):
                    found.append(key)
            elif has_any_unnegated_keyword(text, TOPIC_MAP.get(key, [])):
                found.append(key)
        return found

    @staticmethod
    def _is_thin(text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        return len(compact) <= 5

    def _rule_reply(
        self,
        state: InterviewState,
        text: str,
        analysis: UtteranceAnalysis,
    ) -> str:
        focus = analysis.focus_text or text
        if self._is_greeting(text):
            next_topic = self._next_uncovered(state) or TOPICS[0]
            state.current_topic = next_topic.key
            return (
                "您好，我在这儿呢。您不用着急，想到什么都可以说。"
                "我们先从最近的身体和心情聊起。\n\n"
                f"{self._ask_topic(state, next_topic)}"
            )

        body_reply = self._body_part_response(focus)
        if body_reply:
            state.current_topic = "somatic"
            if "躯体不适" not in state.risk_cues:
                state.risk_cues.append("躯体不适")
            if not self._is_thin(focus):
                state.covered.add("somatic")
            state.last_focus_text = focus
            return body_reply

        from elderly_depression.dialogue_policy import (
            build_sleep_response,
            classify_answer_archetype,
            is_situational_late_sleep,
        )

        if is_situational_late_sleep(text):
            response = build_sleep_response(
                "sleep",
                text,
                classify_answer_archetype(text),
                state.turn_count,
                focus_text=text,
                positive_context=analysis.positive_context,
            )
            if response:
                state.current_topic = "sleep"
                state.covered.add("sleep")
                state.pending_topic = "sleep"
                state.pending_slot = "sleep_late_pattern"
                state.addressed_issues.add("sleep_rhythm")
                if self._is_sleep_exercise_context(state, text):
                    self._mark_sleep_exercise_timing(state)
                return response

        mentioned = self._mentioned_topics(focus)
        if not mentioned:
            prediction = self._intent_prediction(focus, "interview")
            if (
                prediction
                and prediction.get("confidence", 0) >= INTENT_FASTPATH_THRESHOLD
                and prediction.get("topic") in VALID_TOPIC_KEYS
            ):
                mentioned = [prediction["topic"]]
        active = self._select_topic(
            state,
            mentioned,
            prefer_mentioned=analysis.has_focused_issue or bool(mentioned),
        )

        if active is None:
            state.covered.update({t.key for t in TOPICS})
            return CLOSING

        reflection = self._reflection(focus, active, state.turn_count)
        topic_map = {item.key: item for item in TOPICS}
        topic = topic_map.get(active)
        if topic is None:
            active = self._next_uncovered(state)
            topic = topic_map.get(active) if active else None
        if topic is None:
            state.covered.update({item.key for item in TOPICS})
            return f"{reflection}\n\n{CLOSING}"
        attempts = state.attempts.get(active, 0)
        state.current_topic = active
        topic_detail_present = has_any_unnegated_keyword(
            focus,
            TOPIC_MAP.get(active, []),
        )

        try:
            from elderly_depression.dialogue_policy import (
                build_policy_response,
                build_sleep_response,
                classify_answer_archetype,
                classify_sleep_subtype,
                is_situational_late_sleep,
            )

            archetype = classify_answer_archetype(focus)
            sleep_subtype = classify_sleep_subtype(text)
            specialized_sleep = build_sleep_response(
                active,
                text,
                archetype,
                state.turn_count,
                focus_text=text,
                positive_context=analysis.positive_context,
            )
            situational_late_sleep = is_situational_late_sleep(text)
        except Exception:
            build_policy_response = None
            specialized_sleep = None
            archetype = "unknown"
            sleep_subtype = "unknown"
            situational_late_sleep = False

        if archetype == "avoidant" and build_policy_response is not None:
            state.covered.add(active)
            policy = build_policy_response(
                active,
                archetype,
                state.turn_count,
                focus_text=focus,
                positive_context=analysis.positive_context,
            )
            if policy:
                state.addressed_issues.add(active)
                return policy

        if (
            active in state.covered
            and not analysis.has_focused_issue
            and not topic_detail_present
        ) or attempts >= len(topic.followups):
            state.covered.add(active)
            next_topic = self._next_uncovered(state, after=active)
            if next_topic is None:
                return f"{reflection}\n\n{CLOSING}"
            return f"{reflection}\n\n{self._ask_topic(state, next_topic)}"

        if not self._is_thin(focus) or archetype in {"positive", "moderate", "severe"}:
            state.covered.add(active)
        state.attempts[active] = attempts + 1
        if specialized_sleep:
            state.addressed_issues.add(f"sleep_{sleep_subtype}")
            if self._is_sleep_exercise_context(state, text):
                self._mark_sleep_exercise_timing(state)
            elif sleep_subtype == "rhythm":
                state.pending_topic = "sleep"
                state.pending_slot = (
                    "sleep_late_pattern"
                    if situational_late_sleep
                    else "sleep_schedule"
                )
            elif active in CORE_FOLLOWUP_CLOSERS and "？" in specialized_sleep:
                state.pending_topic = active
                state.pending_slot = f"core_{active}"
            if archetype in {"mild", "moderate"}:
                state.suggested_topics.add(active)
            return specialized_sleep
        if build_policy_response is not None:
            policy = build_policy_response(
                active,
                archetype,
                state.turn_count,
                focus_text=focus,
                positive_context=analysis.positive_context,
            )
            if policy:
                state.addressed_issues.add(active)
                state.suggested_topics.add(active)
                if active in CORE_FOLLOWUP_CLOSERS and "？" in policy:
                    state.pending_topic = active
                    state.pending_slot = f"core_{active}"
                return policy
        followup = topic.followups[min(attempts, len(topic.followups) - 1)]
        state.pending_topic = active
        state.pending_slot = f"core_{active}"
        return f"{reflection}\n\n{followup}"

    @staticmethod
    def _body_part_response(text: str) -> str:
        compact = re.sub(r"\s+", "", text)
        part = next((item for item in BODY_PARTS if item in compact), "")
        if not part:
            return ""
        if part == "头" and any(word in compact for word in ["晕", "昏", "眩"]):
            return (
                "头晕吗？是天旋地转、昏昏沉沉，还是站起来时更明显？"
                "有没有伴随恶心或看东西不清？"
            )
        if part == "胸口" and "闷" in compact:
            return (
                "胸口闷吗？是像压着东西、喘不过气，还是活动时更明显？"
                "有没有心慌或出冷汗？"
            )
        if part in {"胃", "肚子"} and any(
            word in compact for word in ["胀", "反酸", "恶心", "拉肚子", "腹泻"]
        ):
            return (
                f"{part}不舒服吗？是胀、反酸、恶心，还是拉肚子？"
                "吃饭前后会不会更明显？"
            )
        followup = BODY_PART_FOLLOWUPS.get(part, "这个不舒服在什么时候会更明显？")
        if any(word in compact for word in ["疼", "痛", "酸", "僵", "麻"]):
            return (
                f"{part}疼起来确实不舒服。是酸痛、刺痛，还是持续胀痛？"
                f"{followup}这种情况大概持续多久了？"
            )
        return (
            f"{part}不舒服吗？是酸痛、僵硬，还是别的感觉？"
            f"{followup}这种情况大概持续多久了？"
        )

    @staticmethod
    def _companion_rule_reply(
        state: InterviewState,
        text: str,
        analysis: UtteranceAnalysis,
    ) -> str:
        focus = re.sub(r"\s+", "", analysis.focus_text or text)
        if (
            has_any_unnegated_keyword(
                focus,
                [
                "不好",
                "不太好",
                "不太",
                "不怎么样",
                "不怎么",
                "不行",
                "难受",
                "糟糕",
                "不舒服",
                "没胃口",
                "吃不下",
                "睡不着",
                "睡不好",
                "老醒",
                "心里堵",
                "孤单",
                "孤独",
                "委屈",
                "记不住",
                "心情不好",
                "很烦",
                "太累",
                ],
            )
            or ("心里" in focus and "堵" in focus)
        ):
            if analysis.positive_context:
                return (
                    f"{analysis.positive_context}，不过您心里还有不痛快的地方。"
                    "您可以先说说，最让您难受的是哪件事？"
                )
            return "听您这么说，最近是不是不太舒服，或者心里有点烦？先慢慢说，我陪着您。"
        if has_any_unnegated_keyword(
            focus, ["开心", "高兴", "好", "不错", "喜欢", "出去", "散步"]
        ):
            return "那真好呀。能说说具体是什么时候、和谁在一起吗？我想听听您开心的事。"
        if any(word in focus for word in ["天气", "热", "冷", "下雨", "晴", "风"]):
            return "是呀，天气会影响心情。最近这边天气怎么样，您还习惯吗？"
        if any(word in focus for word in ["吃", "饭", "菜", "胃口", "做饭"]):
            return "吃饭这件事挺重要的。最近家里都做了什么菜，您最爱吃哪一样？"
        if any(word in focus for word in ["孩子", "孙子", "老伴", "家人", "儿子", "女儿"]):
            return "家里人的事最挂心。他们最近怎么样，有常来看您吗？"
        if any(word in focus for word in ["身体", "疼", "累", "不舒服"]):
            return "听您说到身体，我有点心疼。先不急着想别的，今天有没有舒服一点的时候？"
        return "嗯，我听着呢。再多跟我说一点吧，不管是今天的事，还是以前的故事。"

    def _ensure_interview_progress(
        self,
        state: InterviewState,
        reply: str,
        mode: str,
        attach_existing_question: bool = False,
    ) -> str:
        """Keep every interview turn open unless the user asked to stop."""
        if mode != "interview" or not reply.strip():
            return reply
        if state.pending_slot:
            return reply
        if any(
            phrase in reply
            for phrase in [
                "这个话题可以先放下",
                "先停一停",
                "不继续问了",
                "不想继续问",
                "告诉我",
                "接着说",
                "继续说",
                "说说",
                "我在这儿陪着",
            ]
        ):
            return reply

        topic = state.current_topic
        if topic not in PROGRESS_QUESTIONS:
            next_topic = self._next_uncovered(state)
            if next_topic is None:
                return (
                    f"{reply}\n\n"
                    "您最希望我先记住或继续了解哪一点？"
                )
            topic = next_topic.key
        if "？" in reply:
            if attach_existing_question:
                state.pending_topic = topic
                state.pending_slot = f"generic_followup_{topic}"
            return reply
        topic_addressed = any(
            issue == topic or issue.startswith(f"{topic}_")
            for issue in state.addressed_issues
        )
        if topic in state.covered and topic_addressed:
            next_topic = self._next_uncovered(state, after=topic)
            if next_topic:
                return (
                    f"{reply}\n\n"
                    f"{self._ask_progress_topic(state, next_topic.key)}"
                )
        findings = state.clinical_findings
        if (
            topic == "mood"
            and findings.get("duration")
            and findings.get("functional_impact")
        ):
            state.covered.add(topic)
            next_topic = self._next_uncovered(state, after=topic)
            if next_topic:
                return (
                    f"{reply}\n\n"
                    f"{self._ask_progress_topic(state, next_topic.key)}"
                )
        question = self._progress_question_for(state, topic)
        state.pending_topic = topic
        state.pending_slot = f"generic_followup_{topic}"
        return f"{reply}\n\n{question}"

    @staticmethod
    def _progress_question_for(state: InterviewState, topic: str) -> str:
        findings = state.clinical_findings
        if findings.get("duration"):
            if topic == "mood":
                if findings.get("functional_impact"):
                    return "最近有没有哪一刻，这种难受会稍微轻一点？"
                return "这种感受有没有影响吃饭、睡眠或出门？"
            if topic == "somatic":
                return "这个不舒服有没有影响走路、做事或睡觉？"
            if topic == "cognition":
                return "这些变化有没有影响您平时做事情？"
            if topic == "interest":
                return "这有没有影响您出门或和人来往？"
        return PROGRESS_QUESTIONS.get(
            topic,
            "您愿意再具体说说现在最影响生活的一点吗？",
        )

    @classmethod
    def _ask_progress_topic(
        cls,
        state: InterviewState,
        topic: str,
    ) -> str:
        state.current_topic = topic
        state.pending_topic = topic
        state.pending_slot = f"generic_followup_{topic}"
        return cls._progress_question_for(state, topic)

    @staticmethod
    def _ask_topic(state: InterviewState, topic: Topic) -> str:
        state.current_topic = topic.key
        state.pending_topic = topic.key
        state.pending_slot = f"core_{topic.key}"
        return topic.opening

    @staticmethod
    def _select_topic(
        state: InterviewState,
        mentioned: list[str],
        prefer_mentioned: bool = False,
    ) -> str | None:
        if prefer_mentioned:
            explicit = next((key for key in mentioned if key != "safety"), None)
            if explicit:
                return explicit
        for key in mentioned:
            if key != "safety" and key not in state.covered:
                return key
        if (
            state.current_topic in VALID_TOPIC_KEYS
            and state.current_topic not in state.covered
        ):
            return state.current_topic
        next_topic = AgentAdapter._next_uncovered(state)
        return next_topic.key if next_topic else None

    @staticmethod
    def _next_uncovered(state: InterviewState, after: str | None = None) -> Topic | None:
        if after:
            start = next((i for i, t in enumerate(TOPICS) if t.key == after), -1) + 1
            ordered = TOPICS[start:] + TOPICS[:start]
            candidates = [t for t in ordered if t.key not in state.covered]
        else:
            candidates = [t for t in TOPICS if t.key not in state.covered]
        return candidates[0] if candidates else None

    @staticmethod
    def _reflection(text: str, topic: str = "", turn_count: int = 0) -> str:
        if has_any_unnegated_keyword(
            text, ["还好", "不错", "挺好", "有人陪", "出去", "开心", "稳定"]
        ):
            variants = POSITIVE_REFLECTIONS
        else:
            variants = TOPIC_REFLECTIONS.get(topic, GENERIC_REFLECTIONS)
        digest = sum((index + 1) * ord(char) for index, char in enumerate(text))
        return variants[(digest + turn_count) % len(variants)]

    def _try_llm_reply(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any],
        state: InterviewState,
        text: str,
        mode: str = "interview",
    ) -> str | None:
        if not self.api_key or not context.get("cloud_consent", False):
            return None
        try:
            analysis = analyze_utterance(text)
            focus = analysis.focus_text or text
            if self._client is None:
                if self._client_error:
                    return None
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )

            if mode == "companion":
                system = (
                    "你是一位温柔、健谈、有生活气息的老年陪伴朋友。你的任务是轻松聊天，"
                    "听老人说家常、回忆、天气、饮食、家人和今天的小事。不要问筛查量表，"
                    "不要诊断，不要连续追问情绪问题。多用短句，适当表达关心和共鸣。"
                    "如果出现自伤或自杀表达，优先确认安全并建议联系家人、医生或求助热线。"
                    '只输出 JSON：{"reply":"你的陪伴式回复","topic":"companion","risk_cues":[]}'
                )
            else:
                current = next((t for t in TOPICS if t.key == state.current_topic), TOPICS[0])
                planned = self._next_uncovered(state, after=current.key) or current
                system = (
                    "你是一位温和、耐心的中国老年抑郁风险筛查访谈员。你的目标是通过自然对话"
                    "了解老人的睡眠、兴趣、情绪、躯体、认知、社交、无望感和安全风险。"
                    "请共情但不诊断、不治疗，不贴标签，语言要短句、口语化、尊重老人。"
                    "如果出现自伤或自杀表达，优先确认安全并建议联系家人、医生或求助热线。"
                    f"当前主题：{current.label}。已完成主题："
                    f"{'、'.join([t.label for t in TOPICS if t.key in state.covered]) or '无'}。"
                    f"下一轮倾向主题：{planned.label}。"
                    '只输出 JSON：{"reply":"你的共情回复和提问","topic":"主题英文键",'
                    '"risk_cues":["风险线索"],"cue_confidence":{"风险线索":0.0-1.0},'
                    '"next_question":"下一句要问的问题"}'
                )
            from elderly_depression.privacy import (
                memory_for_cloud,
                redact_messages,
                redact_text,
            )

            cloud_memory = memory_for_cloud(state.user_facts)
            if cloud_memory:
                memory_text = "；".join(
                    f"{key}：{value}" for key, value in cloud_memory.items()
                )
                system += (
                    f" 已知用户信息：{memory_text}。请自然利用这些信息，不要重复询问已经知道的内容。"
                )
            if state.memory_summary:
                system += f" 对话早期摘要：{state.memory_summary}。"
            if state.clinical_summary:
                system += (
                    " 已采集的临床访谈字段："
                    + state.clinical_summary
                    + "。不要重复询问已经明确提供的信息。"
                )
            if state.last_guidance_prompt:
                system += (
                    " 本地整理的低风险支持建议："
                    + state.last_guidance_prompt
                    + " 请按用户当前状态筛选，不要照抄全部内容或替代医生诊疗。"
                )
            if state.medical_kb_prompt:
                system += (
                    " 权威患者教育资料（MedlinePlus, U.S. National Library of Medicine）："
                    + state.medical_kb_prompt
                    + " 只提取与当前问题相关的事实，不直接照搬长篇内容，不把资料当作诊断。"
                )
            addressed = [
                ADDRESSED_ISSUE_LABELS.get(item, item)
                for item in sorted(state.addressed_issues)
            ]
            if addressed:
                system += " 已回应或已给建议：" + "、".join(addressed) + "。不要重复追问这些内容。"
            if analysis.background_text or analysis.positive_context:
                safe_background = redact_text(
                    analysis.background_text or analysis.positive_context
                )
                safe_focus = redact_text(focus)
                system += (
                    f" 用户本轮先说明了背景：{safe_background}；"
                    f"真正需要回应的是：{safe_focus}。先承认背景，再处理焦点，"
                    "不要把背景误当成整句结论。"
                )
            else:
                system += f" 用户本轮真正需要回应的是：{redact_text(focus)}。"
            system += (
                " 回复要求：使用简短口语，最多三句话，只问一个问题；"
                "先回应用户感受和已经给出的具体信息，再自然衔接；"
                "每次回复都应形成可继续的下一轮，追问一个当前问题中尚未获得、"
                "且会改变判断或行动的关键事实；如果已经可以给出行动，"
                "就问用户更愿意选择哪一步。不要复述用户原句，不要重复上一轮问题，"
                "不要输出列表或诊断结论；用户明确想结束时停止追问。"
            )
            known_identifiers = [
                state.user_facts.get(key, "")
                for key in ["姓名", "居住地"]
                if state.user_facts.get(key)
            ]
            prompt = _as_openai_messages(
                redact_messages(messages, known_identifiers),
                system,
            )
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=220,
                timeout=20.0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if not match:
                    return None
                data = json.loads(match.group(0))
            reply = str(data.get("reply", "")).strip()
            reply = self._sanitize_reply(reply)
            if not reply:
                return None
            if self._is_repetitive(state, reply):
                return None
            topic = str(data.get("topic", "")).strip()
            if topic in VALID_TOPIC_KEYS:
                state.current_topic = topic
            if mode != "companion":
                for cue in data.get("risk_cues", []) or []:
                    if cue and cue not in state.risk_cues:
                        state.risk_cues.append(str(cue))
            conf = data.get("cue_confidence") or {}
            if isinstance(conf, dict):
                state.cue_confidence.update({str(k): float(v) for k, v in conf.items()})
            next_q = str(data.get("next_question", "")).strip()
            if next_q:
                state.llm_next_question = next_q
            return reply
        except Exception as exc:  # pragma: no cover - provider/network failure
            self._client_error = str(exc)
            self._client = None
            return None

    @staticmethod
    def _sanitize_reply(reply: str) -> str:
        text = re.sub(r"\s+", " ", str(reply or "")).strip()
        text = text.replace("```json", "").replace("```", "").strip()
        if len(text) > 260:
            cut = max(
                text.rfind("。", 0, 220),
                text.rfind("？", 0, 220),
                text.rfind("！", 0, 220),
            )
            text = text[: cut + 1] if cut >= 80 else text[:220].rstrip() + "。"
        return text

    @staticmethod
    def _is_repetitive(state: InterviewState, reply: str) -> bool:
        if not state.last_reply:
            return False
        left = re.sub(r"\s+", "", state.last_reply)
        right = re.sub(r"\s+", "", reply)
        return SequenceMatcher(None, left, right).ratio() >= 0.90


AGENT_CONTRACT = {
    "purpose": "自适应老年抑郁筛查访谈 Agent；本地意图、响应策略和检索知识优先，用户明确同意后才使用第三方模型。",
    "endpoint": "POST /api/agent/chat",
    "request": {
        "session_id": "string",
        "messages": [{"role": "user", "text": "string"}],
        "context": {
            "mode": "interview | companion",
            "age": "int",
            "gender": "string",
            "scale_name": "string",
            "previous_risk": "float",
            "api_key": "string (optional)",
            "cloud_consent": "boolean; required before any third-party model call",
        },
    },
    "response": {
        "reply": "string",
        "agent": "string",
        "mode": "string",
        "topic": "string",
        "risk_cues": ["string"],
        "cue_confidence": {"风险线索": "0.0-1.0"},
        "covered_topics": ["string"],
        "next_question": "string",
        "safety_alert": "boolean",
        "safety_plan_status": "string",
        "clinical_findings": {
            "frequency": "string",
            "duration": "string",
            "functional_impact": "string",
            "sleep": "string",
            "exercise_end": "string",
            "sleep_time": "string",
            "appetite": "string",
        },
        "clinical_summary": "string",
        "support_preference": "listen | action | both | empty",
        "protective_factors": ["string"],
        "accepted_actions": ["string"],
        "last_guidance_titles": ["string"],
        "engine": "llm | fastpath | cache | local",
    },
}
