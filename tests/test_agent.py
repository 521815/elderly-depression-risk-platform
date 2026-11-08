import unittest

from backend_api.agent_adapter import AgentAdapter, VALID_TOPIC_KEYS


class AgentTests(unittest.TestCase):
    def test_adaptive_topic_transition(self):
        agent = AgentAdapter()
        first = agent.chat(
            [
                {"role": "assistant", "text": agent.greeting()},
                {"role": "user", "text": "最近总睡不着，半夜醒，白天没精神。"},
            ],
            context={},
            session_id="unit-1",
        )
        self.assertIn("睡不好", first)
        snapshot = agent.snapshot(session_id="unit-1")
        self.assertIn("sleep", snapshot["covered_topics"])
        self.assertIn("睡眠差", snapshot["risk_cues"])

    def test_safety_reply(self):
        agent = AgentAdapter()
        reply = agent.chat(
            [{"role": "user", "text": "有时候觉得活着没意思，甚至想死。"}],
            context={},
            session_id="unit-2",
        )
        self.assertIn("安全最重要", reply)
        self.assertIn("您现在安全吗", reply)
        self.assertTrue(agent.snapshot(session_id="unit-2")["safety_alert"])

    def test_safety_followup_confirms_support(self):
        agent = AgentAdapter()
        first = agent.chat(
            [{"role": "user", "text": "有时候觉得活着没意思，甚至想死。"}],
            context={"mode": "interview"},
            session_id="safety-followup-safe",
        )
        second = agent.chat(
            [
                {"role": "user", "text": "有时候觉得活着没意思，甚至想死。"},
                {"role": "assistant", "text": first},
                {"role": "user", "text": "现在安全，女儿陪着我"},
            ],
            context={"mode": "interview"},
            session_id="safety-followup-safe",
        )
        self.assertIn("有可联系的人", second)
        self.assertIn("不要独处", second)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="safety-followup-safe"
        )
        self.assertEqual(snapshot["safety_plan_status"], "support_confirmed")

    def test_safety_followup_escalates_when_unsafe(self):
        agent = AgentAdapter()
        first = agent.chat(
            [{"role": "user", "text": "我觉得活着没意义，想伤害自己"}],
            context={"mode": "interview"},
            session_id="safety-followup-unsafe",
        )
        second = agent.chat(
            [
                {"role": "user", "text": "我觉得活着没意义，想伤害自己"},
                {"role": "assistant", "text": first},
                {"role": "user", "text": "我现在控制不住，也没人"},
            ],
            context={"mode": "interview"},
            session_id="safety-followup-unsafe",
        )
        self.assertIn("120", second)
        self.assertIn("不要独处", second)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="safety-followup-unsafe"
        )
        self.assertEqual(snapshot["safety_plan_status"], "emergency_referral")

    def test_passive_death_wish_enters_safety_followup(self):
        agent = AgentAdapter()
        reply = agent.chat(
            [{"role": "user", "text": "最近总觉得没有我更好，也不想醒来"}],
            context={"mode": "interview"},
            session_id="passive-safety",
        )
        self.assertIn("您现在安全吗", reply)
        self.assertEqual(agent.last_fastpath_reason, "safety_rule")
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="passive-safety"
        )
        self.assertEqual(snapshot["safety_plan_status"], "checking")

    def test_help_request_routes_to_safety_body_and_life_triage(self):
        agent = AgentAdapter(api_key=None)
        first = agent.chat(
            [{"role": "user", "text": "我该怎么办，真的不知道怎么办"}],
            context={"mode": "interview"},
            session_id="help-triage",
        )
        self.assertIn("自身安全", first)
        self.assertIn("身体急症", first)
        self.assertIn("情绪和生活问题", first)
        self.assertEqual(agent.last_fastpath_reason, "help_request")
        second = agent.chat(
            [
                {"role": "user", "text": "我该怎么办，真的不知道怎么办"},
                {"role": "assistant", "text": first},
                {"role": "user", "text": "是情绪和生活问题"},
            ],
            context={"mode": "interview"},
            session_id="help-triage",
        )
        self.assertIn("信任的人", second)
        self.assertIn("发消息或打电话", second)

    def test_help_request_escalates_body_emergency(self):
        agent = AgentAdapter(api_key=None)
        first = agent.chat(
            [{"role": "user", "text": "帮帮我"}],
            context={"mode": "interview"},
            session_id="help-emergency",
        )
        second = agent.chat(
            [
                {"role": "user", "text": "帮帮我"},
                {"role": "assistant", "text": first},
                {"role": "user", "text": "身体急症，胸痛喘不上气"},
            ],
            context={"mode": "interview"},
            session_id="help-emergency",
        )
        self.assertIn("120", second)

    def test_emergency_red_flag_preempts_normal_dialogue(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "胸口疼，喘不上气，还冒汗"}],
            context={"mode": "interview"},
            session_id="red-flag",
        )
        self.assertIn("120", reply)
        snapshot = agent.snapshot({"mode": "interview"}, session_id="red-flag")
        self.assertEqual(snapshot["clinical_alert"]["level"], "emergency")

    def test_scale_evidence_is_a_confirmation_hint(self):
        agent = AgentAdapter(api_key=None)
        agent.chat(
            [{"role": "user", "text": "最近睡不着，什么都不想做"}],
            context={"mode": "interview"},
            session_id="scale-evidence",
        )
        snapshot = agent.snapshot({"mode": "interview"}, session_id="scale-evidence")
        items = {item["item"] for item in snapshot["scale_evidence"]}
        self.assertIn(1, items)
        self.assertIn(3, items)
        self.assertTrue(
            all(item["status"] == "需量表确认" for item in snapshot["scale_evidence"])
        )

    def test_local_retrieval_adds_support_guidance_to_snapshot(self):
        agent = AgentAdapter(api_key=None)
        agent.chat(
            [{"role": "user", "text": "最近心里堵，偶尔想哭"}],
            context={"mode": "interview"},
            session_id="guidance-snapshot",
        )
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="guidance-snapshot"
        )
        self.assertTrue(snapshot["last_guidance_ids"])
        self.assertTrue(
            any("情绪" in title for title in snapshot["last_guidance_titles"])
        )
        self.assertTrue(snapshot["medical_sources"])
        self.assertTrue(
            all(
                source["organization"] == "National Library of Medicine"
                for source in snapshot["medical_sources"]
            )
        )

    def test_general_medical_knowledge_routes_blood_pressure(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "最近总是后脑勺胀，血压高"}],
            context={"mode": "interview"},
            session_id="general-medical-bp",
        )
        self.assertIn("血压", reply)
        self.assertIn("复测", reply)
        self.assertIn("120", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="general-medical-bp"
        )
        self.assertEqual(
            snapshot["general_medical_id"],
            "general-hypertension",
        )

    def test_general_medical_followup_records_answer(self):
        agent = AgentAdapter(api_key=None)
        first = agent.chat(
            [{"role": "user", "text": "最近总是口渴、尿多"}],
            context={"mode": "interview"},
            session_id="general-medical-followup",
        )
        second = agent.chat(
            [
                {"role": "user", "text": "最近总是口渴、尿多"},
                {"role": "assistant", "text": first},
                {"role": "user", "text": "血糖 12，还没看过医生"},
            ],
            context={"mode": "interview"},
            session_id="general-medical-followup",
        )
        self.assertIn("这些信息我记下了", second)
        self.assertIn("及时就医", second)

    def test_negated_symptoms_do_not_create_cues(self):
        agent = AgentAdapter(api_key=None)
        agent.chat(
            [{"role": "user", "text": "我没有失眠，也没有不舒服，不想死"}],
            context={"mode": "interview"},
            session_id="negated-cues",
        )
        snapshot = agent.snapshot({"mode": "interview"}, session_id="negated-cues")
        self.assertNotIn("睡眠差", snapshot["risk_cues"])
        self.assertFalse(snapshot["safety_alert"])

    def test_companion_mode_is_separate_and_casual(self):
        agent = AgentAdapter()
        greeting = agent.chat([], {"mode": "companion"}, session_id="unit-3")
        self.assertIn("随便聊聊", greeting)
        reply = agent.chat(
            [{"role": "user", "text": "今天天气不错，出去走了走。"}],
            {"mode": "companion"},
            session_id="unit-3",
        )
        self.assertIn("天气", reply)
        self.assertNotIn("抑郁", reply)
        interview = agent.snapshot({"mode": "interview"}, session_id="unit-3")
        companion = agent.snapshot({"mode": "companion"}, session_id="unit-3")
        self.assertEqual(interview["mode"], "interview")
        self.assertEqual(companion["mode"], "companion")

    def test_preset_answer_uses_fastpath_instead_of_llm(self):
        agent = AgentAdapter(api_key="invalid", base_url="http://127.0.0.1:1")
        reply = agent.chat(
            [
                {"role": "assistant", "text": agent.greeting()},
                {"role": "user", "text": "最近总睡不着，半夜醒。"},
            ],
            context={},
            session_id="unit-4",
        )
        self.assertIn("睡不好", reply)
        self.assertEqual(agent.last_engine, "fastpath")

    def test_companion_negative_is_not_treated_as_positive(self):
        agent = AgentAdapter()
        reply = agent.chat(
            [{"role": "user", "text": "今天心情不太好。"}],
            context={"mode": "companion"},
            session_id="unit-5",
        )
        self.assertNotIn("那真好呀", reply)
        self.assertIn("陪着您", reply)

    def test_companion_negative_combines_empathy_and_action(self):
        agent = AgentAdapter()
        first = agent.chat(
            [{"role": "user", "text": "今天心里很烦，什么都不想做"}],
            context={"mode": "companion"},
            session_id="companion-negative-care",
        )
        self.assertIn("不痛快", first)
        self.assertIn("陪着您", first)
        self.assertIn("先听", first)
        self.assertIn("想办法", first)
        second = agent.chat(
            [
                {"role": "user", "text": "今天心里很烦，什么都不想做"},
                {"role": "assistant", "text": first},
                {"role": "user", "text": "一起想办法"},
            ],
            context={"mode": "companion"},
            session_id="companion-negative-care",
        )
        self.assertIn("可信的人", second)
        snapshot = agent.snapshot(
            {"mode": "companion"}, session_id="companion-negative-care"
        )
        self.assertEqual(snapshot["support_preference"], "action")

    def test_companion_can_choose_to_be_listened_to_first(self):
        agent = AgentAdapter()
        first = agent.chat(
            [{"role": "user", "text": "一个人有点孤单"}],
            context={"mode": "companion"},
            session_id="companion-listen-first",
        )
        second = agent.chat(
            [
                {"role": "user", "text": "一个人有点孤单"},
                {"role": "assistant", "text": first},
                {"role": "user", "text": "先听我说"},
            ],
            context={"mode": "companion"},
            session_id="companion-listen-first",
        )
        self.assertIn("不急着给建议", second)
        snapshot = agent.snapshot(
            {"mode": "companion"}, session_id="companion-listen-first"
        )
        self.assertEqual(snapshot["support_preference"], "listen")

    def test_companion_dignity_and_life_review_are_not_treated_as_symptoms(self):
        agent = AgentAdapter()
        dignity = agent.chat(
            [{"role": "user", "text": "我老了，没用了，成了家里的负担"}],
            context={"mode": "companion"},
            session_id="companion-dignity",
        )
        self.assertIn("价值", dignity)
        self.assertIn("不好受", dignity)
        self.assertNotIn("那真好呀", dignity)
        life_review = agent.chat(
            [{"role": "user", "text": "我想说说这一辈子最骄傲的事"}],
            context={"mode": "companion"},
            session_id="companion-life-review",
        )
        self.assertTrue(any(word in life_review for word in ["人生", "经历"]))

    def test_topic_advance_wraps_back_to_earlier_topics(self):
        agent = AgentAdapter()
        state = agent._session("unit-6", {})
        state.covered = {"sleep", "interest", "mood", "somatic", "cognition", "social"}
        state.current_topic = "social"
        nxt = agent._next_uncovered(state, after="social")
        self.assertEqual(nxt.key, "negative_cognition")

    def test_topic_progression_returns_topic_key_not_topic_object(self):
        agent = AgentAdapter()
        state = agent._session("topic-key-regression", {})
        state.current_topic = "sleep"
        state.covered = {"sleep"}
        selected = agent._select_topic(state, ["sleep"])
        self.assertIsInstance(selected, str)
        self.assertIn(selected, VALID_TOPIC_KEYS)

    def test_completed_sleep_topic_does_not_trigger_generic_fallback(self):
        agent = AgentAdapter(api_key=None)
        messages = []
        for text in ["您好", "最近总是睡不好", "还好，就是作息有点难调"]:
            messages.append({"role": "user", "text": text})
            reply = agent.chat(
                messages,
                context={"mode": "interview"},
                session_id="sleep-sequence-regression",
            )
            self.assertNotIn("没接好话头", reply)
            messages.append({"role": "assistant", "text": reply})

    def test_negative_cognition_priority_over_somatic_substring(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "觉得自己拖累孩子"}],
            context={"mode": "interview"},
            session_id="negative-cognition-priority",
        )
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="negative-cognition-priority"
        )
        self.assertEqual(snapshot["topic"], "negative_cognition")
        self.assertIn("想法", reply)

    def test_invalid_safety_topic_does_not_crash_rule_fallback(self):
        agent = AgentAdapter()
        state = agent._session("unit-7", {})
        state.current_topic = "safety"
        reply = agent.chat(
            [{"role": "user", "text": "今天就一般，没什么特别的。"}],
            context={},
            session_id="unit-7",
        )
        self.assertTrue(reply)
        self.assertNotEqual(reply.strip(), "")

    def test_repeated_llm_reply_uses_session_cache(self):
        agent = AgentAdapter(api_key="invalid", base_url="http://127.0.0.1:1")
        calls = {"count": 0}

        def fake_llm(messages, context, state, text, mode="interview"):
            calls["count"] += 1
            return "这是缓存回复，您想接着说说吗？"

        agent._try_llm_reply = fake_llm
        message = "请详细说说今天发生的一件让我印象很深的事情"
        consent = {"cloud_consent": True}
        first = agent.chat(
            [{"role": "user", "text": message}],
            consent,
            session_id="unit-8",
        )
        second = agent.chat(
            [{"role": "user", "text": message}],
            consent,
            session_id="unit-8",
        )
        self.assertEqual(first, "这是缓存回复，您想接着说说吗？")
        self.assertEqual(second, "这是缓存回复，您想接着说说吗？")
        self.assertEqual(calls["count"], 1)
        self.assertEqual(agent.last_engine, "cache")

    def test_cloud_model_is_not_called_without_consent(self):
        agent = AgentAdapter(api_key="invalid", base_url="http://127.0.0.1:1")
        calls = {"count": 0}

        def fake_llm(messages, context, state, text, mode="interview"):
            calls["count"] += 1
            return "不应出现"

        agent._try_llm_reply = fake_llm
        reply = agent.chat(
            [{"role": "user", "text": "最近有些说不上来的感受"}],
            {"cloud_consent": False},
            session_id="no-cloud-consent",
        )
        self.assertTrue(reply)
        self.assertNotEqual(reply, "不应出现")
        self.assertEqual(calls["count"], 0)

    def test_learned_intent_model_can_trigger_fastpath(self):
        agent = AgentAdapter(api_key="invalid", base_url="http://127.0.0.1:1")
        reply = agent.chat(
            [{"role": "user", "text": "心里像压着一块石头，老想掉眼泪"}],
            context={},
            session_id="unit-9",
        )
        self.assertTrue(reply)
        self.assertEqual(agent.last_engine, "fastpath")

    def test_daily_and_professional_intents_have_fast_responses(self):
        agent = AgentAdapter(api_key="invalid", base_url="http://127.0.0.1:1")
        daily = agent.chat(
            [{"role": "user", "text": "今天吃了饺子，挺香的"}],
            context={"mode": "companion"},
            session_id="unit-10",
        )
        self.assertEqual(agent.last_engine, "fastpath")
        self.assertIn("吃饭", daily)
        professional = agent.chat(
            [{"role": "user", "text": "最近药有时候忘记吃"}],
            context={"mode": "interview"},
            session_id="unit-11",
        )
        self.assertEqual(agent.last_engine, "fastpath")
        self.assertTrue(
            any(word in professional for word in ["按时", "忘记", "提醒"])
        )

    def test_sleep_answer_archetypes_get_different_responses(self):
        cases = [
            ("最近睡得挺好", ["白天"]),
            ("偶尔会醒一下", ["一周", "几次"]),
            ("最近经常睡不好", ["入睡", "半夜"]),
            ("每天半夜都醒", ["持续", "白天"]),
            ("说不清楚", ["一周", "几天"]),
            ("不想聊这个", ["不聊", "没关系", "换"]),
        ]
        replies = []
        for index, (text, expected_any) in enumerate(cases):
            agent = AgentAdapter(api_key=None)
            reply = agent.chat(
                [{"role": "user", "text": text}],
                context={"mode": "interview"},
                session_id=f"archetype-{index}",
            )
            replies.append(reply)
            self.assertTrue(any(word in reply for word in expected_any))
        self.assertEqual(len(replies), len(set(replies)))

    def test_professional_answer_archetypes(self):
        agent = AgentAdapter(api_key=None)
        mild = agent.chat(
            [{"role": "user", "text": "偶尔忘记吃药"}],
            context={"mode": "interview"},
            session_id="med-mild",
        )
        self.assertIn("偶尔", mild)
        self.assertTrue(any(word in mild for word in ["分格药盒", "提醒"]))
        severe = agent.chat(
            [{"role": "user", "text": "每天都会忘记吃药"}],
            context={"mode": "interview"},
            session_id="med-severe",
        )
        self.assertIn("医生", severe)

    def test_shoulder_mentions_get_somatic_response_not_fallback(self):
        for mode in ["interview", "companion"]:
            agent = AgentAdapter(api_key=None)
            reply = agent.chat(
                [{"role": "user", "text": "肩膀不舒服"}],
                context={"mode": mode},
                session_id=f"shoulder-{mode}",
            )
            self.assertIn("肩膀", reply)
            self.assertNotIn("没接好话头", reply)
            self.assertEqual(agent.last_engine, "fastpath")

    def test_body_specific_questions_are_not_generic(self):
        agent = AgentAdapter(api_key=None)
        dizziness = agent.chat(
            [{"role": "user", "text": "头晕"}],
            context={"mode": "interview"},
            session_id="body-dizzy",
        )
        chest = agent.chat(
            [{"role": "user", "text": "胸口闷"}],
            context={"mode": "interview"},
            session_id="body-chest",
        )
        self.assertIn("天旋地转", dizziness)
        self.assertIn("尽快就医", chest)

    def test_sleep_schedule_answer_does_not_ask_about_waking(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "还好，就是作息有点难调"}],
            context={"mode": "interview"},
            session_id="sleep-rhythm",
        )
        self.assertIn("作息", reply)
        self.assertNotIn("偶尔醒", reply)
        self.assertNotIn("半夜醒", reply)

    def test_situational_late_sleep_stays_on_sleep_topic(self):
        cases = [
            "昨天看了个电影所以睡晚了",
            "昨晚追剧看到很晚",
            "前天和朋友聊到很晚才睡",
            "最近总熬夜",
        ]
        for index, text in enumerate(cases):
            with self.subTest(text=text):
                agent = AgentAdapter(api_key=None)
                reply = agent.chat(
                    [
                        {"role": "user", "text": "最近睡得怎么样"},
                        {
                            "role": "assistant",
                            "text": "是入睡困难，还是半夜容易醒、醒得早？",
                        },
                        {"role": "user", "text": text},
                    ],
                    context={"mode": "interview"},
                    session_id=f"situational-late-sleep-{index}",
                )
                self.assertTrue(
                    any(
                        phrase in reply
                        for phrase in ["晚睡", "熬夜", "上床时间", "作息"]
                    )
                )
                self.assertNotIn("少了些乐趣", reply)
                self.assertNotIn("喜欢的事", reply)
                snapshot = agent.snapshot(
                    {"mode": "interview"},
                    session_id=f"situational-late-sleep-{index}",
                )
                self.assertEqual(snapshot["topic"], "sleep")
                self.assertNotIn("睡眠差", snapshot["risk_cues"])

    def test_exercise_related_late_sleep_keeps_all_disclosed_facts(self):
        agent = AgentAdapter(api_key=None)
        text = (
            "睡的不错，就是作息还没调整好，入睡的相对较晚，"
            "可能是因为健身导致的"
        )
        reply = agent.chat(
            [
                {"role": "user", "text": "您好"},
                {"role": "assistant", "text": agent.greeting()},
                {"role": "user", "text": text},
            ],
            context={"mode": "interview"},
            session_id="exercise-late-sleep",
        )
        self.assertIn("睡眠本身不错", reply)
        self.assertIn("健身时间", reply)
        self.assertIn("几点结束健身", reply)
        self.assertNotIn("一时说不清", reply)
        self.assertNotIn("一周大概有几天", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="exercise-late-sleep"
        )
        self.assertEqual(snapshot["topic"], "sleep")
        self.assertNotIn("睡眠差", snapshot["risk_cues"])

    def test_cortisol_followup_uses_previous_sleep_context(self):
        agent = AgentAdapter(api_key=None)
        messages = [
            {
                "role": "user",
                "text": (
                    "睡的不错，就是作息还没调整好，入睡的相对较晚，"
                    "可能是因为健身导致的"
                ),
            }
        ]
        first = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="cortisol-followup",
        )
        messages.extend(
            [
                {"role": "assistant", "text": first},
                {"role": "user", "text": "可能是因为健身导致的皮质醇"},
            ]
        )
        second = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="cortisol-followup",
        )
        self.assertIn("皮质醇短时升高", second)
        self.assertIn("不能只凭这个判断", second)
        self.assertNotIn("喜欢的事", second)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="cortisol-followup"
        )
        self.assertEqual(snapshot["topic"], "sleep")
        self.assertIn(
            "sleep-exercise-cortisol",
            snapshot["last_guidance_ids"],
        )

    def test_sleep_exercise_timing_flow_reaches_a_plan_and_next_topic(self):
        agent = AgentAdapter(api_key=None)
        messages = [{"role": "user", "text": "最近睡得怎么样"}]
        opening = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-exercise-timing-flow",
        )
        messages.append({"role": "assistant", "text": opening})
        messages.append(
            {"role": "user", "text": "可能是因为健身导致的皮质醇"}
        )
        cortisol = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-exercise-timing-flow",
        )
        self.assertIn("几点结束健身", cortisol)
        self.assertNotIn("以前让您高兴", cortisol)

        messages.append({"role": "assistant", "text": cortisol})
        messages.append({"role": "user", "text": "九点"})
        bedtime_question = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-exercise-timing-flow",
        )
        self.assertIn("21点结束", bedtime_question)
        self.assertIn("几点上床", bedtime_question)

        messages.append({"role": "assistant", "text": bedtime_question})
        messages.append({"role": "user", "text": "十一点"})
        interval_reply = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-exercise-timing-flow",
        )
        self.assertIn("大约间隔2小时", interval_reply)
        self.assertIn("健身提前", interval_reply)
        self.assertIn("晚间降低强度", interval_reply)
        self.assertNotIn("一周有几天", interval_reply)

        messages.append({"role": "assistant", "text": interval_reply})
        messages.append({"role": "user", "text": "把健身提前"})
        plan_reply = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-exercise-timing-flow",
        )
        self.assertIn("提前三十分钟", plan_reply)
        self.assertIn("连续记录一周", plan_reply)
        self.assertIn("以前让您高兴", plan_reply)
        snapshot = agent.snapshot(
            {"mode": "interview"},
            session_id="sleep-exercise-timing-flow",
        )
        self.assertEqual(
            snapshot["clinical_findings"]["exercise_end"],
            "21点",
        )
        self.assertEqual(
            snapshot["clinical_findings"]["sleep_time"],
            "23点",
        )
        self.assertIn("健身结束：21点", snapshot["clinical_summary"])
        self.assertIn("上床时间：23点", snapshot["clinical_summary"])

    def test_sleep_exercise_unknown_timing_switches_to_observation(self):
        agent = AgentAdapter(api_key=None)
        messages = [
            {"role": "user", "text": "可能是因为健身导致的皮质醇"}
        ]
        first = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-exercise-unknown",
        )
        self.assertIn("几点结束健身", first)
        messages.extend(
            [
                {"role": "assistant", "text": first},
                {"role": "user", "text": "说不清"},
            ]
        )
        second = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-exercise-unknown",
        )
        self.assertIn("比如九点或十点", second)
        messages.extend(
            [
                {"role": "assistant", "text": second},
                {"role": "user", "text": "还是记不清"},
            ]
        )
        third = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-exercise-unknown",
        )
        self.assertIn("连续记三天", third)
        self.assertIn("白天", third)
        snapshot = agent.snapshot(
            {"mode": "interview"},
            session_id="sleep-exercise-unknown",
        )
        self.assertNotIn("exercise_end", snapshot["clinical_findings"])

    def test_sleep_exercise_pending_slot_yields_to_a_new_topic(self):
        agent = AgentAdapter(api_key=None)
        messages = [
            {"role": "user", "text": "可能是因为健身导致的皮质醇"}
        ]
        first = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-exercise-topic-switch",
        )
        messages.extend(
            [
                {"role": "assistant", "text": first},
                {"role": "user", "text": "先不聊睡觉了，我胸口有点闷"},
            ]
        )
        switched = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-exercise-topic-switch",
        )
        self.assertNotIn("几点结束健身", switched)
        snapshot = agent.snapshot(
            {"mode": "interview"},
            session_id="sleep-exercise-topic-switch",
        )
        self.assertEqual(snapshot["topic"], "somatic")

    def test_generic_progress_guard_continues_across_clinical_topics(self):
        cases = [
            (
                "sleep",
                "睡眠变化可以先记录下来，再结合白天精神判断。",
                "这种睡眠变化持续多久了",
            ),
            (
                "mood",
                "情绪变化需要结合持续时间和生活影响一起看。",
                "这种感受大概持续多久了",
            ),
            (
                "somatic",
                "疼痛可能与活动方式、睡眠和情绪共同有关。",
                "这个不舒服大概持续多久了",
            ),
            (
                "medication",
                "漏服后不要自行补双倍剂量，可以先联系医生确认。",
                "更容易忘记哪一顿药",
            ),
        ]
        for index, (topic, statement, expected_question) in enumerate(cases):
            with self.subTest(topic=topic):
                agent = AgentAdapter(api_key=None)
                state = agent._session(
                    f"generic-progress-{index}",
                    {"mode": "interview"},
                )
                state.current_topic = topic
                reply = agent._ensure_interview_progress(
                    state,
                    statement,
                    "interview",
                )
                self.assertIn(expected_question, reply)
                self.assertEqual(
                    state.pending_slot,
                    f"generic_followup_{topic}",
                )

    def test_generic_followup_uses_the_answer_and_moves_forward(self):
        agent = AgentAdapter(api_key=None)
        state = agent._session(
            "generic-followup",
            {"mode": "interview"},
        )
        state.current_topic = "somatic"
        first = agent._ensure_interview_progress(
            state,
            "疼痛可能与活动方式、睡眠和情绪共同有关。",
            "interview",
        )
        self.assertIn("持续多久", first)
        reply = agent.chat(
            [{"role": "user", "text": "膝盖上下楼疼，已经两周了"}],
            context={"mode": "interview"},
            session_id="generic-followup",
        )
        self.assertIn("持续：两周", reply)
        self.assertIn("这些变化有没有影响", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"},
            session_id="generic-followup",
        )
        self.assertIn("somatic", snapshot["covered_topics"])

    def test_existing_local_question_is_wired_into_next_turn(self):
        agent = AgentAdapter(api_key=None)
        state = agent._session(
            "generic-question-slot",
            {"mode": "interview"},
        )
        state.current_topic = "somatic"
        first = agent._ensure_interview_progress(
            state,
            "膝盖疼是酸痛、僵硬，还是活动时刺痛？",
            "interview",
            attach_existing_question=True,
        )
        self.assertIn("活动时刺痛", first)
        self.assertEqual(state.pending_slot, "generic_followup_somatic")
        reply = agent.chat(
            [{"role": "user", "text": "酸痛，已经两天了"}],
            context={"mode": "interview"},
            session_id="generic-question-slot",
        )
        self.assertIn("持续：两天", reply)
        self.assertTrue(reply.rstrip().endswith("？"))

    def test_compensated_late_sleep_acknowledges_all_three_facts(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [
                {
                    "role": "user",
                    "text": "最近这一周，您晚上睡得好吗？",
                },
                {
                    "role": "assistant",
                    "text": "会不会半夜醒、醒得早，或者白天总没精神？",
                },
                {
                    "role": "user",
                    "text": "昨晚看电影睡晚了，但是因为时长睡够了所以精神还好",
                },
            ],
            context={"mode": "interview"},
            session_id="compensated-late-sleep",
        )
        self.assertIn("睡得晚", reply)
        self.assertIn("总时长睡够", reply)
        self.assertIn("精神也没受影响", reply)
        self.assertNotIn("入睡困难", reply)
        self.assertNotIn("少了些乐趣", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="compensated-late-sleep"
        )
        self.assertEqual(snapshot["topic"], "sleep")
        self.assertNotIn("睡眠差", snapshot["risk_cues"])

    def test_user_can_reject_an_inferred_loss_of_interest(self):
        agent = AgentAdapter(api_key=None)
        messages = [
            {"role": "user", "text": "最近睡得怎么样"},
            {
                "role": "assistant",
                "text": "是入睡困难，还是半夜容易醒、醒得早？",
            },
            {"role": "user", "text": "昨天看了个电影所以睡晚了"},
        ]
        sleep_reply = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="interest-correction",
        )
        messages.extend(
            [
                {"role": "assistant", "text": sleep_reply},
                {"role": "user", "text": "并没有少点乐趣啊"},
            ]
        )
        reply = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="interest-correction",
        )
        self.assertIn("理解偏了", reply)
        self.assertIn("电影", reply)
        self.assertIn("睡晚", reply)
        self.assertIn("第二天精神", reply)
        self.assertNotIn("喜欢的事", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="interest-correction"
        )
        self.assertIn("interest", snapshot["covered_topics"])
        self.assertIn("兴趣保持", snapshot["protective_factors"])
        self.assertNotIn("兴趣减退", snapshot["risk_cues"])

    def test_sleep_schedule_keeps_positive_daytime_context(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [
                {
                    "role": "user",
                    "text": "睡的还行，白天也还好，就是睡的我觉得有点晚，调整作息有困难",
                }
            ],
            context={"mode": "interview"},
            session_id="sleep-rhythm-positive-daytime",
        )
        self.assertIn("都还可以", reply)
        self.assertIn("几点上床", reply)
        self.assertNotIn("白天不太舒服", reply)

    def test_sleep_schedule_clarification_gets_actionable_advice(self):
        agent = AgentAdapter(api_key=None)
        messages = []
        first = agent.chat(
            [{"role": "user", "text": "睡的还不错，调作息有困难"}],
            context={"mode": "interview"},
            session_id="sleep-clarification",
        )
        self.assertIn("睡眠本身还可以", first)
        self.assertNotIn("作息比较规律", first)
        messages.extend(
            [
                {"role": "user", "text": "睡的还不错，调作息有困难"},
                {"role": "assistant", "text": first},
                {"role": "user", "text": "不是，就是想调整作息"},
            ]
        )
        second = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-clarification",
        )
        self.assertIn("明白", second)
        self.assertIn("固定起床", second)
        self.assertNotIn("没听清", second)

    def test_sleep_schedule_duration_and_quality_flow(self):
        agent = AgentAdapter(api_key=None)
        messages = []
        first = agent.chat(
            [
                {
                    "role": "user",
                    "text": "睡的还行，白天也还好，就是睡的我觉得有点晚，调整作息有困难",
                }
            ],
            context={"mode": "interview"},
            session_id="sleep-full-flow",
        )
        messages.extend(
            [
                {
                    "role": "user",
                    "text": "睡的还行，白天也还好，就是睡的我觉得有点晚，调整作息有困难",
                },
                {"role": "assistant", "text": first},
                {"role": "user", "text": "一两点，早上自然醒的话会到十点"},
            ]
        )
        second = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-full-flow",
        )
        self.assertIn("9小时", second)
        self.assertIn("自然醒", second)
        self.assertNotIn("睡不好", second)
        messages.extend(
            [
                {"role": "assistant", "text": second},
                {"role": "user", "text": "夜里不太醒，白天也还行"},
            ]
        )
        third = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-full-flow",
        )
        self.assertIn("质量还可以", third)
        self.assertIn("固定", third)
        self.assertIn("午睡", third)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="sleep-full-flow"
        )
        self.assertNotIn("睡眠差", snapshot["risk_cues"])
        self.assertEqual(snapshot["scale_evidence"], [])

    def test_composite_social_answer_focuses_on_loneliness(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "平时有人陪，不过还是觉得孤单"}],
            context={"mode": "interview"},
            session_id="composite-social",
        )
        self.assertIn("仍觉得孤单", reply)
        self.assertIn("心里话", reply)
        self.assertNotIn("最近和谁聊得比较多", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="composite-social"
        )
        self.assertEqual(snapshot["focus"], "还是觉得孤单")
        self.assertIn("社交孤立", snapshot["risk_cues"])
        self.assertIn("保护因素", snapshot["protective_factors"])

    def test_short_affirmative_interest_answer_keeps_correct_polarity(self):
        for index, answer in enumerate(["愿意的", "愿意", "当然愿意"]):
            with self.subTest(answer=answer):
                agent = AgentAdapter(api_key=None)
                session_id = f"interest-affirmative-{index}"
                state = agent._session(session_id, {"mode": "interview"})
                state.current_topic = "interest"
                state.pending_topic = "interest"
                state.pending_slot = "core_interest"
                state.last_reply = (
                    "以前让您高兴的事，比如下棋、听戏、养花、串门，"
                    "现在还愿意做吗？"
                )
                reply = agent.chat(
                    [{"role": "user", "text": answer}],
                    context={"mode": "interview"},
                    session_id=session_id,
                )
                self.assertIn("那就好", reply)
                self.assertIn("愿意做", reply)
                self.assertNotIn("不想做", reply)
                self.assertNotIn("失落", reply)
                snapshot = agent.snapshot(
                    {"mode": "interview"}, session_id=session_id
                )
                self.assertNotIn("兴趣减退", snapshot["risk_cues"])
                self.assertIn("兴趣保持", snapshot["protective_factors"])
                self.assertIn("interest", snapshot["covered_topics"])

    def test_interest_response_uses_remembered_hobby(self):
        agent = AgentAdapter(api_key=None)
        state = agent._session("interest-hobby", {"mode": "interview"})
        state.current_topic = "interest"
        state.pending_topic = "interest"
        state.pending_slot = "core_interest"
        state.user_facts["爱好"] = "下棋"
        state.last_reply = "现在还愿意做以前喜欢的事吗？"
        reply = agent.chat(
            [{"role": "user", "text": "愿意的"}],
            context={"mode": "interview"},
            session_id="interest-hobby",
        )
        self.assertIn("下棋", reply)

    def test_short_negative_interest_answer_is_recorded_as_reduced(self):
        agent = AgentAdapter(api_key=None)
        state = agent._session("interest-negative", {"mode": "interview"})
        state.current_topic = "interest"
        state.pending_topic = "interest"
        state.pending_slot = "core_interest"
        state.last_reply = "现在还愿意做以前喜欢的事吗？"
        reply = agent.chat(
            [{"role": "user", "text": "不愿意"}],
            context={"mode": "interview"},
            session_id="interest-negative",
        )
        self.assertIn("不太愿意", reply)
        self.assertIn("身体不舒服", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="interest-negative"
        )
        self.assertIn("兴趣减退", snapshot["risk_cues"])

    def test_short_negative_mood_answer_is_not_treated_as_low_mood(self):
        agent = AgentAdapter(api_key=None)
        state = agent._session("mood-negative", {"mode": "interview"})
        state.current_topic = "mood"
        state.pending_topic = "mood"
        state.pending_slot = "core_mood"
        state.last_reply = "最近是不是经常觉得心里堵、难过，或者想掉眼泪？"
        reply = agent.chat(
            [{"role": "user", "text": "没有"}],
            context={"mode": "interview"},
            session_id="mood-negative",
        )
        self.assertIn("没有明显低落", reply)
        self.assertNotIn("最近确实有这样的情绪", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="mood-negative"
        )
        self.assertNotIn("情绪低落", snapshot["risk_cues"])

    def test_partial_negative_mood_answer_gets_care_before_questions(self):
        agent = AgentAdapter(api_key=None)
        state = agent._session("mood-partial", {"mode": "interview"})
        state.current_topic = "mood"
        state.pending_topic = "mood"
        state.pending_slot = "core_mood"
        state.last_reply = (
            "最近是不是经常觉得心里堵、难过，或者会莫名其妙想掉眼泪？"
        )
        reply = agent.chat(
            [{"role": "user", "text": "偶尔会"}],
            context={"mode": "interview"},
            session_id="mood-partial",
        )
        self.assertIn("不好受", reply)
        self.assertIn("愿意说出来", reply)
        self.assertIn("打个电话", reply)
        self.assertIn("会影响吃饭", reply)
        self.assertNotIn("身体上有没有不舒服", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="mood-partial"
        )
        self.assertIn("情绪低落", snapshot["risk_cues"])
        self.assertEqual(snapshot["topic"], "mood")

    def test_full_affirmative_mood_answer_gets_strong_care_and_solutions(self):
        agent = AgentAdapter(api_key=None)
        state = agent._session("mood-full-affirmative", {"mode": "interview"})
        state.current_topic = "mood"
        state.pending_topic = "mood"
        state.pending_slot = "core_mood"
        state.last_reply = (
            "最近是不是经常觉得心里堵、难过，或者会莫名其妙想掉眼泪？"
        )
        reply = agent.chat(
            [{"role": "user", "text": "会的"}],
            context={"mode": "interview"},
            session_id="mood-full-affirmative",
        )
        self.assertIn("有些担心", reply)
        self.assertIn("不是您太脆弱", reply)
        self.assertIn("您愿意的话", reply)
        self.assertIn("信任的家人", reply)
        self.assertIn("专业评估", reply)
        self.assertIn("吃饭、睡觉", reply)
        self.assertNotIn("身体上有没有不舒服", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="mood-full-affirmative"
        )
        self.assertIn("情绪低落", snapshot["risk_cues"])

    def test_uncertain_mood_answer_uses_easier_frequency_question(self):
        agent = AgentAdapter(api_key=None)
        state = agent._session("mood-uncertain", {"mode": "interview"})
        state.current_topic = "mood"
        state.pending_topic = "mood"
        state.pending_slot = "core_mood"
        state.last_reply = "最近是不是经常觉得心里堵、难过？"
        reply = agent.chat(
            [{"role": "user", "text": "还行吧"}],
            context={"mode": "interview"},
            session_id="mood-uncertain",
        )
        self.assertIn("一时说不清", reply)
        self.assertIn("最近一周", reply)
        self.assertNotIn("最近确实有这样的情绪", reply)

    def test_positive_mood_disclosure_collects_impact_and_duration(self):
        agent = AgentAdapter(api_key=None)
        state = agent._session("mood-clinical-slots", {"mode": "interview"})
        state.current_topic = "mood"
        state.pending_topic = "mood"
        state.pending_slot = "clinical_mood_impact"
        state.last_reply = "这种状态有没有影响吃饭、睡觉或日常事务？"
        first = agent.chat(
            [{"role": "user", "text": "会影响吃饭和睡觉"}],
            context={"mode": "interview"},
            session_id="mood-clinical-slots",
        )
        self.assertIn("影响到日常生活", first)
        self.assertIn("持续多久", first)
        second = agent.chat(
            [
                {"role": "user", "text": "会影响吃饭和睡觉"},
                {"role": "assistant", "text": first},
                {"role": "user", "text": "已经一个月了"},
            ],
            context={"mode": "interview"},
            session_id="mood-clinical-slots",
        )
        self.assertIn("持续：一个月", second)
        self.assertIn("专业评估", second)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="mood-clinical-slots"
        )
        self.assertEqual(snapshot["clinical_findings"]["duration"], "一个月")
        self.assertEqual(
            snapshot["clinical_findings"]["functional_impact"],
            "有",
        )

    def test_medication_change_request_uses_medical_boundary(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "我不想看医生，直接停药可以吗"}],
            context={"mode": "interview"},
            session_id="medical-boundary-medication",
        )
        self.assertIn("不能替您决定", reply)
        self.assertIn("医生或药师", reply)
        self.assertIn("120", reply)
        self.assertEqual(agent.last_fastpath_reason, "medical_boundary")

    def test_medication_refusal_is_not_treated_as_generic_adherence(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "我不想吃药了，可以停吗"}],
            context={"mode": "interview"},
            session_id="medication-refusal-boundary",
        )
        self.assertIn("不能替您决定", reply)
        self.assertEqual(agent.last_fastpath_reason, "medical_boundary")

    def test_appetite_statement_is_not_treated_as_medication_change(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "我胃口不好，不吃了"}],
            context={"mode": "interview"},
            session_id="appetite-not-medication",
        )
        self.assertNotIn("不能替您决定停药", reply)
        self.assertNotEqual(agent.last_fastpath_reason, "medical_boundary")

    def test_diagnosis_request_is_not_treated_as_confirmed_diagnosis(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "我是不是得了抑郁症"}],
            context={"mode": "interview"},
            session_id="medical-boundary-diagnosis",
        )
        self.assertIn("不能确诊", reply)
        self.assertIn("专业评估", reply)
        self.assertEqual(agent.last_fastpath_reason, "medical_boundary")

    def test_explicit_correction_restores_the_replaced_topic(self):
        agent = AgentAdapter(api_key=None)
        state = agent._session("explicit-topic-correction", {"mode": "interview"})
        state.last_focus_text = "最近睡不着"
        reply = agent.chat(
            [
                {
                    "role": "user",
                    "text": "你理解错了，我说的是没胃口，不是睡不着",
                }
            ],
            context={"mode": "interview"},
            session_id="explicit-topic-correction",
        )
        self.assertIn("没胃口", reply)
        self.assertIn("吃不下", reply)
        self.assertNotIn("您愿意再说说，真正想让我记住", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="explicit-topic-correction"
        )
        self.assertEqual(snapshot["topic"], "somatic")

    def test_alliance_repair_acknowledges_misunderstanding(self):
        agent = AgentAdapter(api_key=None)
        state = agent._session("alliance-repair", {"mode": "interview"})
        state.current_topic = "sleep"
        state.last_focus_text = "最近总是睡不好"
        reply = agent.chat(
            [{"role": "user", "text": "你听不懂我的意思"}],
            context={"mode": "interview"},
            session_id="alliance-repair",
        )
        self.assertIn("没接准", reply)
        self.assertIn("最近总是睡不好", reply)
        self.assertIn("换个话题", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="alliance-repair"
        )
        self.assertEqual(snapshot["alliance_repair_count"], 1)
        self.assertEqual(snapshot["last_alliance_repair"], "misunderstanding")

    def test_alliance_repair_respects_stop_questioning_boundary(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "别问了，我不想回答"}],
            context={"mode": "interview"},
            session_id="alliance-boundary",
        )
        self.assertIn("不问了", reply)
        self.assertIn("有权利停下来", reply)
        self.assertNotIn("？", reply)

    def test_goal_and_action_commitment_are_remembered(self):
        agent = AgentAdapter(api_key=None)
        agent.chat(
            [
                {
                    "role": "user",
                    "text": "我最想先调整作息，我会先记录每天的睡眠",
                }
            ],
            context={"mode": "interview"},
            session_id="goal-memory",
        )
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="goal-memory"
        )
        self.assertIn("调整作息", snapshot["memory"].get("近期目标", ""))
        self.assertIn("记录每天的睡眠", snapshot["memory"].get("行动计划", ""))
        self.assertIn("记录每天的睡眠", snapshot["accepted_actions"])

    def test_social_followup_keeps_a_short_relation_answer_in_context(self):
        agent = AgentAdapter(api_key=None)
        messages = [{"role": "user", "text": "平时有人陪，不过还是觉得孤单"}]
        first = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="social-followup",
        )
        messages.extend(
            [
                {"role": "assistant", "text": first},
                {"role": "user", "text": "其实女儿每周都来"},
            ]
        )
        second = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="social-followup",
        )
        self.assertIn("重要", second)
        self.assertNotIn("入睡困难", second)

    def test_core_followup_yields_when_a_new_topic_is_introduced(self):
        agent = AgentAdapter(api_key=None)
        messages = [{"role": "user", "text": "平时有人陪，不过还是觉得孤单"}]
        first = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="social-topic-switch",
        )
        messages.extend(
            [
                {"role": "assistant", "text": first},
                {"role": "user", "text": "最近睡不好，半夜总醒"},
            ]
        )
        second = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="social-topic-switch",
        )
        self.assertIn("睡", second)
        self.assertNotIn("家人愿意常来", second)

    def test_sleep_correction_answers_the_actual_subtype(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "不是睡不着，是早上醒太早"}],
            context={"mode": "interview"},
            session_id="sleep-correction",
        )
        self.assertIn("醒得太早", reply)
        self.assertNotIn("睡眠稳一些", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="sleep-correction"
        )
        self.assertEqual(snapshot["focus"], "早上醒太早")
        self.assertIn("sleep_early", snapshot["addressed_issues"])

    def test_professional_contrast_uses_the_requested_detail(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "吃药基本正常，就是总忘记"}],
            context={"mode": "interview"},
            session_id="medication-contrast",
        )
        self.assertIn("忘记", reply)
        self.assertIn("药盒", reply)
        self.assertNotIn("最近有没有漏服", reply)
        snapshot = agent.snapshot(
            {"mode": "interview"}, session_id="medication-contrast"
        )
        self.assertIn("medication", snapshot["addressed_issues"])
        self.assertNotIn("躯体不适", snapshot["risk_cues"])

    def test_professional_route_uses_disclosed_details(self):
        cases = [
            (
                "function-detail",
                "洗澡穿衣需要帮忙",
                ["需要帮忙", "最费劲"],
                "最近有没有需要家人帮忙",
            ),
            (
                "diet-detail",
                "最近吃不下，体重也轻了",
                ["吃不下", "体重下降", "持续多久"],
                "最近吃得下吗",
            ),
            (
                "sensory-detail",
                "最近听东西不清楚",
                ["听不清楚", "逐渐", "突然"],
                "听声音还清楚吗",
            ),
            (
                "care-detail",
                "平时靠老伴照顾",
                ["照顾", "紧急联系人", "马上联系"],
                "平时谁在照顾您",
            ),
        ]
        for session_id, text, expected, repeated_question in cases:
            with self.subTest(session_id=session_id):
                agent = AgentAdapter(api_key=None)
                reply = agent.chat(
                    [{"role": "user", "text": text}],
                    context={"mode": "interview"},
                    session_id=session_id,
                )
                for phrase in expected:
                    self.assertIn(phrase, reply)
                self.assertNotIn(repeated_question, reply)

    def test_medication_side_effect_takes_priority_over_adherence_template(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "吃药基本正常，就是吃完会头晕"}],
            context={"mode": "interview"},
            session_id="medication-side-effect",
        )
        self.assertIn("头晕", reply)
        self.assertIn("不要自行停药", reply)
        self.assertNotIn("您更容易忘记早上还是晚上", reply)

    def test_professional_followup_accepts_a_short_slot_answer(self):
        agent = AgentAdapter(api_key=None)
        first = agent.chat(
            [{"role": "user", "text": "最近药有时候忘记吃"}],
            context={"mode": "interview"},
            session_id="professional-slot",
        )
        self.assertIn("早上还是晚上", first)
        second = agent.chat(
            [
                {"role": "user", "text": "最近药有时候忘记吃"},
                {"role": "assistant", "text": first},
                {"role": "user", "text": "早上"},
            ],
            context={"mode": "interview"},
            session_id="professional-slot",
        )
        self.assertIn("提醒", second)
        self.assertNotIn("没接好话头", second)

    def test_professional_slot_yields_when_user_switches_topic(self):
        agent = AgentAdapter(api_key=None)
        first = agent.chat(
            [{"role": "user", "text": "最近药有时候忘记吃"}],
            context={"mode": "interview"},
            session_id="professional-slot-switch",
        )
        second = agent.chat(
            [
                {"role": "user", "text": "最近药有时候忘记吃"},
                {"role": "assistant", "text": first},
                {"role": "user", "text": "最近睡不好，半夜总醒"},
            ],
            context={"mode": "interview"},
            session_id="professional-slot-switch",
        )
        self.assertTrue(any(word in second for word in ["睡", "醒"]))
        self.assertNotIn("容易忘的那一顿", second)

    def test_completed_topic_can_be_revisited_after_a_correction(self):
        agent = AgentAdapter(api_key=None)
        messages = [{"role": "user", "text": "最近睡不太好"}]
        first = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-revisit",
        )
        messages.append({"role": "assistant", "text": first})
        messages.append({"role": "user", "text": "不是睡不着，是早上醒太早"})
        second = agent.chat(
            messages,
            context={"mode": "interview"},
            session_id="sleep-revisit",
        )
        self.assertIn("醒得太早", second)
        self.assertNotIn("喜欢做的事", second)

    def test_companion_contrast_keeps_negative_focus(self):
        agent = AgentAdapter(api_key=None)
        reply = agent.chat(
            [{"role": "user", "text": "今天天气不错，不过心里有点堵"}],
            context={"mode": "companion"},
            session_id="companion-contrast",
        )
        self.assertIn("心里", reply)
        self.assertNotIn("穿着还合适吗", reply)

    def test_companion_context_priority(self):
        agent = AgentAdapter(api_key=None)
        family = agent.chat(
            [{"role": "user", "text": "孩子最近让我很烦"}],
            context={"mode": "companion"},
            session_id="companion-family",
        )
        neighbor = agent.chat(
            [{"role": "user", "text": "邻居让我很烦"}],
            context={"mode": "companion"},
            session_id="companion-neighbor",
        )
        shoulder = agent.chat(
            [{"role": "user", "text": "肩膀不舒服"}],
            context={"mode": "companion"},
            session_id="companion-shoulder",
        )
        self.assertIn("家里的事", family)
        self.assertIn("邻居", neighbor)
        self.assertIn("肩膀", shoulder)

    def test_generic_thank_you_phrase_is_not_reused(self):
        for index, text in enumerate(
            ["今天有点说不上来的感觉", "最近心里有些乱", "身体有点怪怪的"]
        ):
            agent = AgentAdapter(api_key=None)
            reply = agent.chat(
                [{"role": "user", "text": text}],
                context={"mode": "interview"},
                session_id=f"reflection-{index}",
            )
            self.assertNotIn("谢谢您愿意说这些", reply)
            self.assertNotIn("我听到了", reply)

    def test_conversation_memory_and_reset(self):
        agent = AgentAdapter(api_key=None)
        agent.chat(
            [
                {
                    "role": "user",
                    "text": "我叫王阿姨，今年75岁，住在上海，我平时喜欢下棋。",
                }
            ],
            context={"mode": "interview"},
            session_id="memory-test",
        )
        snapshot = agent.snapshot({"mode": "interview"}, session_id="memory-test")
        self.assertEqual(snapshot["memory"].get("姓名"), "王阿姨")
        self.assertEqual(snapshot["memory"].get("年龄"), "75")
        self.assertIn("上海", snapshot["memory"].get("居住地", ""))
        self.assertIn("下棋", snapshot["memory"].get("爱好", ""))
        self.assertEqual(agent.reset_session("memory-test", "interview"), 1)
        reset_snapshot = agent.snapshot({"mode": "interview"}, session_id="memory-test")
        self.assertEqual(reset_snapshot["memory"], {})

    def test_reply_sanitizer_and_repetition_guard(self):
        text = "这是一段很长的回复。" * 40
        clean = AgentAdapter._sanitize_reply(text)
        self.assertLessEqual(len(clean), 260)
        self.assertTrue(clean.endswith("。"))
        state = AgentAdapter()._session("rep-test", {})
        state.last_reply = "最近睡得怎么样？"
        self.assertTrue(AgentAdapter._is_repetitive(state, "最近睡得怎么样？"))
        self.assertFalse(AgentAdapter._is_repetitive(state, "今天吃过饭了吗？"))

    def test_chat_latency_and_reason_metadata(self):
        agent = AgentAdapter(api_key=None)
        agent.chat(
            [{"role": "user", "text": "您好"}],
            context={"mode": "interview"},
            session_id="latency-test",
        )
        snapshot = agent.snapshot({"mode": "interview"}, session_id="latency-test")
        self.assertGreaterEqual(snapshot["latency_ms"], 0)
        self.assertIn(
            snapshot["fastpath_reason"],
            {"greeting", "keyword_or_rule", "intent_model"},
        )


if __name__ == "__main__":
    unittest.main()
