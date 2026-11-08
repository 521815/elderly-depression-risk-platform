import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_api.app import app
from elderly_depression import auth as account_auth
from elderly_depression import records, services


class ApiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        self._patchers = [
            mock.patch.object(
                account_auth, "AUTH_DB", self._base / "auth.sqlite"
            ),
            mock.patch.object(
                account_auth,
                "CREDENTIALS_FILE",
                self._base / "账号清单.txt",
            ),
            mock.patch.object(
                account_auth,
                "LEGACY_CREDENTIALS_FILE",
                self._base / "账号密码记录.txt",
            ),
            mock.patch.object(
                account_auth,
                "DEFAULT_ADMIN_PASSWORD",
                "TestAdmin2026",
            ),
            mock.patch.object(
                account_auth,
                "DEFAULT_USER_PASSWORD",
                "TestUser2026",
            ),
            mock.patch.object(records, "USER_DIR", self._base / "user_records"),
            mock.patch.object(records, "USERS_FILE", self._base / "user_records" / "users.json"),
            mock.patch.object(
                records, "USER_RECORDS_DB", self._base / "user_records.sqlite"
            ),
        ]
        for p in self._patchers:
            p.start()
        account_auth._CREDENTIALS.clear()
        account_auth._CREDENTIALS_LOADED = False
        self.client = app.test_client()
        account_auth._ensure_db()
        admin = account_auth.authenticate(
            account_auth.DEFAULT_ADMIN_USERNAME,
            account_auth.DEFAULT_ADMIN_PASSWORD,
        )
        assert admin is not None
        self.admin_headers = {
            "Authorization": f"Bearer {account_auth.create_session(admin['id'])}"
        }

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        self._tmp.cleanup()

    def test_health_and_versioned_route(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json["status"], "ok")
        versioned = self.client.get("/api/v1/status")
        self.assertEqual(versioned.status_code, 200)

    def test_backend_version_endpoint(self):
        resp = self.client.get("/api/version")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json["backend_version"])
        self.assertTrue(resp.json["data_signature"])
        self.assertTrue(resp.json["backend_started_at"])

    @staticmethod
    def _solve(question: str) -> int:
        parts = question.split()
        left_value = int(parts[0])
        operator = parts[1]
        right_value = int(parts[2])
        return left_value + right_value if operator == "+" else left_value - right_value

    def test_auth_login_requires_correct_challenge(self):
        bad = self.client.post(
            "/api/auth/login",
            json={
                "username": account_auth.DEFAULT_ADMIN_USERNAME,
                "password": account_auth.DEFAULT_ADMIN_PASSWORD,
                "challenge_id": "missing",
                "answer": "0",
            },
        )
        self.assertEqual(bad.status_code, 400)

        challenge = self.client.get("/api/auth/challenge").json
        login = self.client.post(
            "/api/auth/login",
            json={
                "username": account_auth.DEFAULT_ADMIN_USERNAME,
                "password": account_auth.DEFAULT_ADMIN_PASSWORD,
                "challenge_id": challenge["challenge_id"],
                "answer": self._solve(challenge["question"]),
            },
        )
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.json["account"]["role"], "admin")
        self.assertTrue(login.json["token"])

    def test_registration_and_user_scope(self):
        challenge = self.client.get("/api/auth/challenge").json
        registered = self.client.post(
            "/api/auth/register",
            json={
                "username": "new_user01",
                "password": "NewUser2026",
                "display_name": "新用户",
                "challenge_id": challenge["challenge_id"],
                "answer": self._solve(challenge["question"]),
                "security_questions": account_auth.SECURITY_QUESTION_OPTIONS[:3],
                "security_answers": ["蓝色", "上海", "饺子"],
            },
        )
        self.assertEqual(registered.status_code, 201)
        token = registered.json["token"]
        headers = {"Authorization": f"Bearer {token}"}

        created = self.client.post(
            "/api/users",
            json={"name": "本人档案", "age": 70},
            headers=headers,
        )
        self.assertEqual(created.status_code, 201)
        users = self.client.get("/api/users", headers=headers).json
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["owner_account_id"], registered.json["account"]["id"])

        forbidden = self.client.get("/api/data_health", headers=headers)
        self.assertEqual(forbidden.status_code, 403)

    def test_password_recovery_and_profile_update(self):
        challenge = self.client.get("/api/auth/challenge").json
        registered = self.client.post(
            "/api/auth/register",
            json={
                "username": "recover_user",
                "password": "Recover2026",
                "display_name": "找回测试",
                "challenge_id": challenge["challenge_id"],
                "answer": self._solve(challenge["question"]),
                "security_questions": account_auth.SECURITY_QUESTION_OPTIONS[:3],
                "security_answers": ["蓝色", "上海", "饺子"],
            },
        )
        self.assertEqual(registered.status_code, 201)
        headers = {"Authorization": f"Bearer {registered.json['token']}"}

        questions = self.client.post(
            "/api/auth/recovery/questions",
            json={"username": "recover_user"},
        )
        self.assertEqual(questions.status_code, 200)
        self.assertEqual(len(questions.json["questions"]), 3)

        reset = self.client.post(
            "/api/auth/recovery/reset",
            json={
                "username": "recover_user",
                "answers": ["蓝色", "上海", "饺子"],
                "new_password": "RecoverNew2026",
            },
        )
        self.assertEqual(reset.status_code, 200)

        old_login_challenge = self.client.get("/api/auth/challenge").json
        old_login = self.client.post(
            "/api/auth/login",
            json={
                "username": "recover_user",
                "password": "Recover2026",
                "challenge_id": old_login_challenge["challenge_id"],
                "answer": self._solve(old_login_challenge["question"]),
            },
        )
        self.assertEqual(old_login.status_code, 401)

        challenge = self.client.get("/api/auth/challenge").json
        new_login = self.client.post(
            "/api/auth/login",
            json={
                "username": "recover_user",
                "password": "RecoverNew2026",
                "challenge_id": challenge["challenge_id"],
                "answer": self._solve(challenge["question"]),
            },
        )
        self.assertEqual(new_login.status_code, 200)
        headers = {"Authorization": f"Bearer {new_login.json['token']}"}

        profile = self.client.patch(
            "/api/account/profile",
            json={
                "current_password": "RecoverNew2026",
                "username": "recover_user_new",
                "display_name": "已改名用户",
            },
            headers=headers,
        )
        self.assertEqual(profile.status_code, 200)
        self.assertEqual(profile.json["account"]["username"], "recover_user_new")

        credentials = (self._base / "账号清单.txt").read_text(encoding="utf-8")
        self.assertIn("recover_user_new", credentials)
        self.assertNotIn("RecoverNew2026", credentials)
        self.assertNotIn("密码：", credentials)

    def test_username_and_password_rules(self):
        self.assertIsNotNone(account_auth.username_error("bad name"))
        self.assertIsNotNone(account_auth.password_error("short"))
        self.assertIsNotNone(account_auth.password_error("alllowercase"))
        self.assertIsNone(account_auth.username_error("mintz_01"))
        self.assertIsNone(account_auth.password_error("GoodPass1"))

    def test_legacy_plaintext_credentials_are_purged(self):
        legacy = self._base / "账号密码记录.txt"
        legacy.write_text("用户名：demo\n密码：PlainSecret1\n", encoding="utf-8")
        account_auth._ensure_db()
        self.assertFalse(legacy.exists())

    def test_admin_can_create_another_admin(self):
        created = self.client.post(
            "/api/admin/accounts",
            json={
                "username": "admin_two",
                "password": "SecondAdmin2026",
                "display_name": "第二管理员",
                "role": "admin",
            },
            headers=self.admin_headers,
        )
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json["role"], "admin")

        duplicate = self.client.post(
            "/api/admin/accounts",
            json={
                "username": "ADMIN_TWO",
                "password": "SecondAdmin2026",
                "role": "admin",
            },
            headers=self.admin_headers,
        )
        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("已存在", duplicate.json["error"])

    def test_smoothing_endpoints_are_admin_only(self):
        self.assertEqual(self.client.post("/api/smoothing/subgroup").status_code, 401)
        with mock.patch.object(
            services, "run_subgroup_shrinkage", return_value={"method": "shrinkage"}
        ):
            response = self.client.post(
                "/api/smoothing/subgroup", headers=self.admin_headers
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["method"], "shrinkage")

        with mock.patch.object(
            services,
            "run_robust_aggregation_ablation",
            return_value={"method": "aggregation"},
        ):
            response = self.client.post(
                "/api/smoothing/aggregation",
                json={"repeats": 1, "splits": 3, "pca_dims": 8},
                headers=self.admin_headers,
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["method"], "aggregation")

    def test_data_health_endpoint(self):
        resp = self.client.get("/api/data_health", headers=self.admin_headers)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("sources", resp.json)
        self.assertIn("issues", resp.json)
        self.assertIn("warnings", resp.json)
        self.assertIn("healthy", resp.json)

    def test_cmdc_status_and_report_are_admin_scoped(self):
        with (
            mock.patch.object(services, "CMDC_CSV", self._base / "cmdc.csv"),
            mock.patch.object(services, "CMDC_REPORT", self._base / "report.csv"),
            mock.patch.object(
                services,
                "CMDC_MODEL_SUMMARY",
                self._base / "summary.json",
            ),
        ):
            unauthorised = self.client.get("/api/cmdc/summary")
            summary = self.client.get(
                "/api/cmdc/summary",
                headers=self.admin_headers,
            )
            report = self.client.get(
                "/api/cmdc/report",
                headers=self.admin_headers,
            )

        self.assertEqual(unauthorised.status_code, 401)
        self.assertEqual(summary.status_code, 200)
        self.assertFalse(summary.json["ready"])
        self.assertEqual(report.status_code, 200)
        self.assertFalse(report.json["ready"])

    def test_user_crud(self):
        created = self.client.post(
            "/api/users",
            json={"name": "测试用户", "age": 70},
            headers=self.admin_headers,
        )
        self.assertEqual(created.status_code, 201)
        uid = created.json["id"]
        self.assertEqual(
            self.client.get("/api/users", headers=self.admin_headers).json[0]["name"],
            "测试用户",
        )

        updated = self.client.put(
            f"/api/users/{uid}", json={"age": 72}, headers=self.admin_headers
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json["age"], 72)

        deleted = self.client.delete(f"/api/users/{uid}", headers=self.admin_headers)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(
            self.client.get("/api/users", headers=self.admin_headers).json, []
        )

    def test_assessment_and_agent(self):
        assessment = self.client.post(
            "/api/assessment",
            json={
                "transcript": [{"role": "user", "text": "最近睡不好"}],
                "scale_answers": {"cesd10": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]},
                "demographics": {"age": 78, "gender": "female", "married": False, "comorbidity_count": 2},
                "expression_summary": {},
            },
            headers=self.admin_headers,
        )
        self.assertEqual(assessment.status_code, 200)
        self.assertIn("depression_probability", assessment.json)

        agent = self.client.post(
            "/api/agent/chat",
            json={"messages": [{"role": "user", "text": "最近睡不好"}], "context": {"mode": "interview"}},
            headers=self.admin_headers,
        )
        self.assertEqual(agent.status_code, 200)
        self.assertTrue(agent.json["reply"])
        self.assertIn("engine", agent.json)
        self.assertIn("llm_error", agent.json)
        self.assertIn("focus", agent.json)
        self.assertIn("positive_context", agent.json)
        self.assertIn("protective_factors", agent.json)
        self.assertIn("addressed_issues", agent.json)
        self.assertIn("alliance_repair_count", agent.json)
        self.assertIn("last_alliance_repair", agent.json)
        self.assertIn("accepted_actions", agent.json)
        self.assertIn("last_guidance_ids", agent.json)
        self.assertIn("last_guidance_titles", agent.json)
        self.assertIn("medical_sources", agent.json)
        self.assertIn("safety_plan_status", agent.json)
        self.assertIn("clinical_findings", agent.json)
        self.assertIn("clinical_summary", agent.json)
        self.assertIn("support_preference", agent.json)

    def test_agent_memory_and_session_reset(self):
        chat = self.client.post(
            "/api/agent/chat",
            json={
                "session_id": "api-memory",
                "messages": [
                    {
                        "role": "user",
                        "text": "我叫李阿姨，今年72岁，住在苏州，我喜欢养花。",
                    }
                ],
                "context": {"mode": "interview"},
            },
            headers=self.admin_headers,
        )
        self.assertEqual(chat.status_code, 200)
        self.assertEqual(chat.json["memory"].get("姓名"), "李阿姨")
        self.assertIn("memory_summary", chat.json)
        self.assertIn("latency_ms", chat.json)
        self.assertIn("fastpath_reason", chat.json)
        self.assertIn("scale_evidence", chat.json)
        self.assertIn("clinical_alert", chat.json)
        reset = self.client.post(
            "/api/agent/reset",
            json={"session_id": "api-memory", "mode": "interview"},
            headers=self.admin_headers,
        )
        self.assertEqual(reset.status_code, 200)
        self.assertGreaterEqual(reset.json["removed"], 1)
        after = self.client.post(
            "/api/agent/chat",
            json={
                "session_id": "api-memory",
                "messages": [{"role": "user", "text": "最近睡不好"}],
                "context": {"mode": "interview"},
            },
            headers=self.admin_headers,
        )
        self.assertEqual(after.status_code, 200)
        self.assertNotIn("姓名", after.json["memory"])

    def test_user_backup_export_import(self):
        created = self.client.post(
            "/api/users",
            json={"name": "备份用户", "age": 72, "comorbidity_count": 1},
            headers=self.admin_headers,
        )
        uid = created.json["id"]
        self.client.post(
            f"/api/users/{uid}/assessments",
            json={
                "record": {
                    "depression_probability": 0.55,
                    "risk_level": "medium",
                    "scale_score": 11,
                    "scale_name": "phq9",
                    "storage_consent": True,
                    "consent_recorded_at": "2026-10-26T00:00:00+00:00",
                }
            },
            headers=self.admin_headers,
        )
        backup = self.client.get("/api/users/export", headers=self.admin_headers)
        self.assertEqual(backup.status_code, 200)
        self.assertEqual(len(backup.json["users"]), 1)

        imported = self.client.post(
            "/api/users/import",
            json={"backup": backup.json, "mode": "replace"},
            headers=self.admin_headers,
        )
        self.assertEqual(imported.status_code, 200)
        self.assertEqual(imported.json["users"], 1)
        self.assertEqual(imported.json["assessments"], 1)
        self.assertEqual(
            self.client.get("/api/users", headers=self.admin_headers).json[0]["name"],
            "备份用户",
        )

    def test_assessment_limit_param(self):
        created = self.client.post(
            "/api/users",
            json={"name": "限额用户", "age": 70},
            headers=self.admin_headers,
        )
        uid = created.json["id"]
        no_consent = self.client.post(
            f"/api/users/{uid}/assessments",
            json={
                "record": {
                    "depression_probability": 0.5,
                    "risk_level": "low",
                }
            },
            headers=self.admin_headers,
        )
        self.assertEqual(no_consent.status_code, 400)
        for _ in range(5):
            self.client.post(
                f"/api/users/{uid}/assessments",
                json={
                    "record": {
                        "depression_probability": 0.5,
                        "risk_level": "low",
                        "storage_consent": True,
                        "consent_recorded_at": "2026-10-26T00:00:00+00:00",
                    }
                },
                headers=self.admin_headers,
            )
        limited = self.client.get(
            f"/api/users/{uid}/assessments?limit=2", headers=self.admin_headers
        )
        self.assertEqual(limited.status_code, 200)
        self.assertEqual(len(limited.json), 2)
        full = self.client.get(
            f"/api/users/{uid}/assessments?limit=100", headers=self.admin_headers
        )
        self.assertEqual(len(full.json), 5)

    def test_models_registry(self):
        resp = self.client.get("/api/models/registry", headers=self.admin_headers)
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json, list)

    def test_background_job_echo(self):
        resp = self.client.post(
            "/api/jobs",
            json={"task": "echo", "payload": {"x": 1}},
            headers=self.admin_headers,
        )
        self.assertEqual(resp.status_code, 202)
        job_id = resp.json["job_id"]
        job = None
        for _ in range(50):
            job = self.client.get(
                f"/api/jobs/{job_id}", headers=self.admin_headers
            ).json
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"], {"echo": {"x": 1}})
        self.assertIn("progress", job)
        self.assertIn("stage", job)

        cancel = self.client.post(
            f"/api/jobs/{job_id}/cancel", headers=self.admin_headers
        )
        self.assertEqual(cancel.status_code, 400)


if __name__ == "__main__":
    unittest.main()
