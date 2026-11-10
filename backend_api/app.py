"""Flask API for the elderly depression risk screening workbench."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, g, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from elderly_depression import auth as account_auth
from elderly_depression import services
from elderly_depression import records as user_records
from elderly_depression.scales import score_cesd10, score_gds15, score_hamd24, score_phq9, severity_level

from backend_api.agent_adapter import AGENT_CONTRACT, AgentAdapter
from backend_api import jobs as job_runner

app = Flask(__name__)
agent = AgentAdapter()
agent.warmup()
_BACKEND_STARTED_AT = datetime.now(timezone.utc).isoformat()
ALLOWED_ORIGINS = {"http://localhost:8501", "http://127.0.0.1:8501"}
RATE_LIMIT = 300
RATE_WINDOW = 60
_RATE_HITS: dict[str, list[float]] = defaultdict(list)
_RATE_LOCK = threading.Lock()

PUBLIC_API_PATHS = {
    "/health",
    "/ready",
    "/api/status",
    "/api/version",
    "/api/auth/challenge",
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/recovery/questions",
    "/api/auth/recovery/reset",
    "/api/auth/security-options",
}
ADMIN_API_PREFIXES = (
    "/api/admin/",
    "/api/charls/",
    "/api/fusion/",
    "/api/research/",
    "/api/smoothing/",
    "/api/imbalance/",
    "/api/mpdd/",
    "/api/cmdc/",
    "/api/synthetic/",
    "/api/real_profile/",
    "/api/jobs",
    "/api/models",
)
ADMIN_API_PATHS = {
    "/api/data_health",
    "/api/users/import",
}


@app.before_request
def attach_request_id():
    g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    g.start_time = time.time()
    ip = request.remote_addr or "local"
    now = time.time()
    with _RATE_LOCK:
        hits = [t for t in _RATE_HITS[ip] if now - t < RATE_WINDOW]
        hits.append(now)
        _RATE_HITS[ip] = hits
        allowed = len(hits) <= RATE_LIMIT
    if not allowed:
        return jsonify({"error": "请求过于频繁，请稍后再试", "code": 429}), 429


def _bearer_token() -> str:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def _requires_admin(path: str) -> bool:
    return path in ADMIN_API_PATHS or any(
        path.startswith(prefix) for prefix in ADMIN_API_PREFIXES
    )


@app.before_request
def enforce_account_auth():
    if request.method == "OPTIONS" or not request.path.startswith("/api/"):
        return None
    path = request.path
    if path.startswith("/api/v1/"):
        path = "/api/" + path[len("/api/v1/") :]
    if path in PUBLIC_API_PATHS:
        return None
    account = account_auth.resolve_session(_bearer_token())
    if account is None:
        return jsonify({"error": "请先登录", "code": 401}), 401
    if _requires_admin(path) and account.get("role") != "admin":
        return jsonify({"error": "仅管理员可执行此操作", "code": 403}), 403
    g.account = account
    return None


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def _coerce_user_fields(payload: dict) -> dict:
    fields = {}
    for key in ["name", "gender", "marital_status", "note"]:
        if key in payload:
            fields[key] = payload[key]
    for key in ["age", "comorbidity_count", "self_rated_health", "followup_label"]:
        if key in payload and payload[key] is not None:
            try:
                fields[key] = int(payload[key])
            except (TypeError, ValueError):
                raise ValueError(f"{key} 必须为整数")
    return fields


SCALE_ITEM_LIMITS = {
    "cesd10": (10, 3),
    "phq9": (9, 3),
    "gds15": (15, 1),
    "hamd24": (24, 4),
}


def _finite_number(value, field: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是数字") from exc
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"{field} 必须在 {minimum:g} 到 {maximum:g} 之间")
    return number


def _validated_scale_items(scale: str, items) -> tuple[str, list[float | None]]:
    scale_name = str(scale or "cesd10").lower()
    if scale_name not in SCALE_ITEM_LIMITS:
        raise ValueError("不支持的量表名称")
    if not isinstance(items, list):
        raise ValueError("items 必须是数组")
    item_count, item_max = SCALE_ITEM_LIMITS[scale_name]
    if len(items) > item_count:
        raise ValueError(f"{scale_name} 最多包含 {item_count} 道题")
    normalized = []
    for index, value in enumerate(items):
        if value is None or value == "":
            normalized.append(None)
            continue
        normalized.append(
            _finite_number(value, f"items[{index}]", 0, item_max)
        )
    return scale_name, normalized


def _validated_assessment_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是对象")
    transcript = payload.get("transcript", [])
    if not isinstance(transcript, list) or len(transcript) > 200:
        raise ValueError("transcript 必须是最多 200 条消息的数组")
    normalized_transcript = []
    for index, message in enumerate(transcript):
        if not isinstance(message, dict):
            raise ValueError(f"transcript[{index}] 必须是对象")
        role = message.get("role")
        text = message.get("text")
        if role not in {"user", "assistant"}:
            raise ValueError(f"transcript[{index}].role 必须是 user 或 assistant")
        if not isinstance(text, str) or len(text) > 4000:
            raise ValueError(f"transcript[{index}].text 必须是最多 4000 字的文本")
        normalized_transcript.append({"role": role, "text": text})

    scale_answers = payload.get("scale_answers", {})
    if not isinstance(scale_answers, dict):
        raise ValueError("scale_answers 必须是对象")
    scale_name = str(payload.get("scale_name", "cesd10")).lower()
    if scale_name not in SCALE_ITEM_LIMITS:
        raise ValueError("不支持的量表名称")
    normalized_answers = {}
    for key, items in scale_answers.items():
        normalized_key, normalized_items = _validated_scale_items(key, items)
        normalized_answers[normalized_key] = normalized_items
    if scale_name not in normalized_answers:
        normalized_answers[scale_name] = []

    demographics = payload.get("demographics", {})
    if not isinstance(demographics, dict):
        raise ValueError("demographics 必须是对象")
    normalized_demographics = {}
    if demographics.get("age") not in (None, ""):
        normalized_demographics["age"] = _finite_number(
            demographics["age"], "age", 18, 120
        )
    if demographics.get("gender") not in (None, ""):
        if demographics["gender"] not in {"female", "male", "女", "男"}:
            raise ValueError("gender 必须是 female、male、女或男")
        normalized_demographics["gender"] = demographics["gender"]
    if "married" in demographics:
        normalized_demographics["married"] = bool(demographics["married"])
    for key, minimum, maximum in [
        ("comorbidity_count", 0, 30),
        ("sleep_quality", 1, 4),
        ("self_rated_health", 1, 5),
    ]:
        if demographics.get(key) not in (None, ""):
            normalized_demographics[key] = _finite_number(
                demographics[key], key, minimum, maximum
            )

    expression = payload.get("expression_summary", {})
    if not isinstance(expression, dict):
        raise ValueError("expression_summary 必须是对象")
    if expression.get("risk") is not None:
        expression = {
            **expression,
            "risk": _finite_number(expression["risk"], "expression_summary.risk", 0, 1),
        }
    mpdd_risk = payload.get("mpdd_risk")
    if mpdd_risk is not None:
        mpdd_risk = _finite_number(mpdd_risk, "mpdd_risk", 0, 1)
    language_cues = payload.get("language_cues", [])
    if language_cues is not None and not isinstance(language_cues, list):
        raise ValueError("language_cues 必须是数组")
    normalized_cues = []
    for index, cue in enumerate(language_cues or []):
        if isinstance(cue, str):
            text = cue.strip()
            if not text or len(text) > 100:
                raise ValueError(f"language_cues[{index}] 长度无效")
            normalized_cues.append(text)
            continue
        if not isinstance(cue, dict):
            raise ValueError(f"language_cues[{index}] 必须是文本或对象")
        text = str(cue.get("cue") or cue.get("label") or "").strip()
        if not text or len(text) > 100:
            raise ValueError(f"language_cues[{index}] 名称无效")
        confidence = cue.get("confidence", 1.0)
        normalized_cues.append(
            {
                "cue": text,
                "confidence": _finite_number(
                    confidence,
                    f"language_cues[{index}].confidence",
                    0,
                    1,
                ),
            }
        )

    return {
        "transcript": normalized_transcript,
        "scale_answers": normalized_answers,
        "demographics": normalized_demographics,
        "expression_summary": expression,
        "scale_name": scale_name,
        "mpdd_risk": mpdd_risk,
        "language_cues": normalized_cues,
    }


def _account_is_admin() -> bool:
    return bool(getattr(g, "account", None) and g.account.get("role") == "admin")


def _can_access_user(user: dict | None) -> bool:
    if user is None:
        return False
    if _account_is_admin():
        return True
    return user.get("owner_account_id") == g.account.get("id")


def _user_for_access(user_id: str):
    user = user_records.get_user(user_id)
    if user is None:
        return None, (jsonify({"error": "用户不存在"}), 404)
    if not _can_access_user(user):
        return None, (jsonify({"error": "无权访问该用户档案", "code": 403}), 403)
    return user, None


@app.after_request
def add_cors(resp):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, PUT, OPTIONS"
    resp.headers["X-Request-ID"] = getattr(g, "request_id", "")
    duration_ms = (time.time() - getattr(g, "start_time", time.time())) * 1000
    app.logger.info(
        "%s %s %s %.1fms req=%s",
        request.method,
        request.path,
        resp.status_code,
        duration_ms,
        g.request_id,
    )
    return resp


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "elderly-depression-backend"})


@app.get("/ready")
def readiness():
    checks = {}
    for name, path in {
        "auth_store_dir": account_auth.AUTH_DB.parent,
        "record_store_dir": user_records.USER_RECORDS_DB.parent,
        "output_dir": services.OUTPUT_DIR,
    }.items():
        checks[name] = bool(path.exists() and os.access(path, os.R_OK | os.W_OK))
    ready = all(checks.values())
    return (
        jsonify(
            {
                "status": "ready" if ready else "not_ready",
                "service": "elderly-depression-backend",
                "checks": checks,
            }
        ),
        200 if ready else 503,
    )


@app.get("/api/status")
def api_status():
    return jsonify(services.status())


@app.get("/api/version")
def api_version():
    return jsonify(
        {
            "backend_version": services.code_signature(),
            "data_signature": services.data_signature(),
            "backend_started_at": _BACKEND_STARTED_AT,
        }
    )


@app.get("/api/auth/challenge")
def auth_challenge():
    return jsonify(account_auth.new_challenge())


@app.post("/api/auth/login")
def auth_login():
    payload = request.get_json(silent=True) or {}
    if not account_auth.verify_challenge(
        payload.get("challenge_id"), payload.get("answer")
    ):
        return (
            jsonify(
                {
                    "error": "人机验证答案不正确或已过期，请重新计算",
                    "code": "challenge_failed",
                }
            ),
            400,
        )
    account = account_auth.authenticate(
        payload.get("username"), payload.get("password")
    )
    if account is None:
        return jsonify({"error": "用户名或密码错误", "code": 401}), 401
    token = account_auth.create_session(
        account["id"], user_agent=request.headers.get("User-Agent")
    )
    return jsonify({"token": token, "account": account})


@app.post("/api/auth/register")
def auth_register():
    payload = request.get_json(silent=True) or {}
    if not account_auth.verify_challenge(
        payload.get("challenge_id"), payload.get("answer")
    ):
        return (
            jsonify(
                {
                    "error": "人机验证答案不正确或已过期，请重新计算",
                    "code": "challenge_failed",
                }
            ),
            400,
        )
    if not payload.get("security_questions") or not payload.get("security_answers"):
        return jsonify({"error": "注册必须选择并回答密保问题"}), 400
    try:
        account = account_auth.create_account(
            payload.get("username"),
            payload.get("password"),
            role="user",
            display_name=payload.get("display_name"),
            security_questions=payload.get("security_questions"),
            security_answers=payload.get("security_answers"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    token = account_auth.create_session(
        account["id"], user_agent=request.headers.get("User-Agent")
    )
    return jsonify({"token": token, "account": account}), 201


@app.get("/api/auth/me")
def auth_me():
    return jsonify({"account": g.account})


@app.post("/api/auth/logout")
def auth_logout():
    account_auth.revoke_session(_bearer_token())
    return jsonify({"logged_out": True})


@app.get("/api/auth/security-options")
def auth_security_options():
    return jsonify({"questions": account_auth.SECURITY_QUESTION_OPTIONS})


@app.post("/api/auth/recovery/questions")
def auth_recovery_questions():
    payload = request.get_json(silent=True) or {}
    questions = account_auth.get_security_questions(payload.get("username"))
    if questions is None:
        return (
            jsonify(
                {
                    "error": "该账号未设置密保问题，请联系管理员重置密码",
                    "code": "security_questions_missing",
                }
            ),
            404,
        )
    return jsonify({"questions": questions})


@app.post("/api/auth/recovery/reset")
def auth_recovery_reset():
    payload = request.get_json(silent=True) or {}
    try:
        account_auth.reset_password_with_security(
            payload.get("username"),
            payload.get("answers") or [],
            payload.get("new_password"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"reset": True})


@app.patch("/api/account/profile")
def account_profile_update():
    payload = request.get_json(silent=True) or {}
    try:
        account = account_auth.update_own_profile(
            g.account["id"],
            current_password=payload.get("current_password", ""),
            username=payload.get("username"),
            display_name=payload.get("display_name"),
            new_password=payload.get("new_password"),
            security_questions=payload.get("security_questions"),
            security_answers=payload.get("security_answers"),
            keep_token=_bearer_token(),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"account": account})


@app.get("/api/admin/accounts")
def admin_accounts_list():
    return jsonify({"accounts": account_auth.list_accounts()})


@app.post("/api/admin/accounts")
def admin_accounts_create():
    payload = request.get_json(silent=True) or {}
    try:
        account = account_auth.create_account(
            payload.get("username"),
            payload.get("password"),
            role=payload.get("role", "admin"),
            display_name=payload.get("display_name"),
            created_by=g.account["id"],
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(account), 201


@app.patch("/api/admin/accounts/<account_id>")
def admin_accounts_update(account_id: str):
    payload = request.get_json(silent=True) or {}
    if account_id == g.account["id"]:
        if payload.get("role") not in {None, "admin"}:
            return jsonify({"error": "不能取消自己的管理员角色"}), 400
        if payload.get("status") not in {None, "active"}:
            return jsonify({"error": "不能停用自己的账号"}), 400
    try:
        account = account_auth.update_account(
            account_id,
            role=payload.get("role"),
            status=payload.get("status"),
            display_name=payload.get("display_name"),
            password=payload.get("password"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if account is None:
        return jsonify({"error": "账号不存在"}), 404
    return jsonify(account)


@app.get("/api/data_health")
def data_health():
    return jsonify(services.data_health_report())


@app.get("/api/models/list")
def models_list():
    tabular = (
        [p.name for p in services.TABULAR_MODEL_DIR.glob("*.pkl")]
        if services.TABULAR_MODEL_DIR.exists()
        else []
    )
    multimodal = (
        [p.name for p in services.MULTIMODAL_MODEL_DIR.glob("*.pkl")]
        if services.MULTIMODAL_MODEL_DIR.exists()
        else []
    )
    return jsonify({"tabular_models": tabular, "multimodal_models": multimodal})


@app.get("/api/charls/summary")
def charls_summary():
    return jsonify(services.charls_summary())


@app.post("/api/charls/build")
def charls_build():
    services.build_charls_dataset()
    return jsonify(services.charls_summary())


@app.post("/api/charls/train")
def charls_train():
    payload = request.get_json(silent=True) or {}
    method = payload.get("resample_method", "smote")
    report = services.train_tabular_baseline(method)
    return jsonify(_records(report))


@app.post("/api/charls/train_temporal")
def charls_train_temporal():
    payload = request.get_json(silent=True) or {}
    method = payload.get("resample_method", "smote_tomek")
    report = services.train_charls_temporal(method)
    return jsonify(_records(report))


@app.get("/api/charls/report")
def charls_report():
    report = services.load_csv(services.MODEL_REPORT)
    if report is None:
        return jsonify({"ready": False})
    return jsonify({"ready": True, "report": _records(report)})


@app.get("/api/charls/temporal_report")
def charls_temporal_report():
    report = services.load_csv(services.CHARLS_TEMPORAL_REPORT)
    if report is None:
        return jsonify({"ready": False})
    return jsonify({"ready": True, "report": _records(report)})


@app.post("/api/charls/train_longitudinal")
def charls_train_longitudinal():
    payload = request.get_json(silent=True) or {}
    method = payload.get("resample_method", "smote_tomek")
    report = services.train_charls_longitudinal(method)
    return jsonify(_records(report))


@app.get("/api/charls/longitudinal_report")
def charls_longitudinal_report():
    report = services.load_csv(services.CHARLS_LONGITUDINAL_REPORT)
    if report is None:
        return jsonify({"ready": False})
    return jsonify({"ready": True, "report": _records(report)})


@app.post("/api/fusion/train")
def fusion_train():
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows", [])
    if not rows:
        return jsonify({"error": "缺少 rows 字段，需要上传含信号概率和标签的 CSV"}), 400
    label_col = payload.get("label_col", "depression")
    try:
        result = services.train_fusion_model(pd.DataFrame(rows), label_col=label_col)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(
        {
            "ready": True,
            "report": _records(result["report"]),
            "status": services.fusion_status(),
        }
    )


@app.get("/api/fusion/report")
def fusion_report():
    report = services.load_csv(services.FUSION_REPORT)
    if report is None:
        return jsonify({"ready": False})
    return jsonify({"ready": True, "report": _records(report)})


@app.get("/api/fusion/status")
def fusion_status():
    return jsonify(services.fusion_status())


@app.get("/api/fusion/components")
def fusion_components():
    return jsonify(services.fusion_components())


@app.get("/api/research/reports")
def research_reports():
    return jsonify(services.research_reports())


@app.post("/api/smoothing/subgroup")
def smoothing_subgroup():
    return jsonify(services.run_subgroup_shrinkage())


@app.post("/api/smoothing/aggregation")
def smoothing_aggregation():
    payload = request.get_json(silent=True) or {}
    return jsonify(
        services.run_robust_aggregation_ablation(
            repeats=int(payload.get("repeats", 2)),
            splits=int(payload.get("splits", 5)),
            pca_dims=int(payload.get("pca_dims", 32)),
        )
    )


@app.get("/api/users")
def users_list():
    owner_account_id = None if _account_is_admin() else g.account["id"]
    return jsonify(
        user_records.list_users(
            query=request.args.get("q"),
            owner_account_id=owner_account_id,
        )
    )


@app.post("/api/users")
def user_create():
    payload = request.get_json(silent=True) or {}
    try:
        fields = _coerce_user_fields(payload)
        if "name" not in fields:
            return jsonify({"error": "用户姓名不能为空"}), 400
        owner_account_id = g.account["id"]
        if _account_is_admin() and payload.get("owner_account_id"):
            owner_account_id = str(payload["owner_account_id"])
            if account_auth.get_account(owner_account_id) is None:
                return jsonify({"error": "归属账号不存在"}), 400
        user = user_records.create_user(
            owner_account_id=owner_account_id,
            **fields,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(user), 201


@app.get("/api/users/<user_id>")
def user_get(user_id: str):
    user, error = _user_for_access(user_id)
    if error:
        return error
    return jsonify(user)


@app.put("/api/users/<user_id>")
def user_update(user_id: str):
    user, error = _user_for_access(user_id)
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    try:
        fields = _coerce_user_fields(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    updated = user_records.update_user(user_id, **fields)
    if updated is None:
        return jsonify({"error": "用户不存在"}), 404
    return jsonify(updated)


@app.delete("/api/users/<user_id>")
def user_delete(user_id: str):
    user, error = _user_for_access(user_id)
    if error:
        return error
    deleted = user_records.delete_user(user_id)
    if not deleted:
        return jsonify({"error": "用户不存在"}), 404
    return jsonify({"deleted": True})


@app.get("/api/users/export")
def users_export():
    owner_account_id = None if _account_is_admin() else g.account["id"]
    return jsonify(user_records.export_all(owner_account_id=owner_account_id))


@app.post("/api/users/import")
def users_import():
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode", "merge")
    if mode not in {"merge", "replace"}:
        return jsonify({"error": "导入模式必须为 merge 或 replace"}), 400
    backup = payload.get("backup") if isinstance(payload.get("backup"), dict) else payload
    try:
        result = user_records.import_backup(backup, mode=mode)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.get("/api/users/<user_id>/assessments")
def user_assessments(user_id: str):
    user, error = _user_for_access(user_id)
    if error:
        return error
    limit_arg = request.args.get("limit")
    records = user_records.list_assessments(
        user_id,
        limit=int(limit_arg) if limit_arg and limit_arg.isdigit() else 200,
        risk=request.args.get("risk") or None,
        scale=request.args.get("scale") or None,
    )
    return jsonify(records)


@app.post("/api/users/<user_id>/assessments")
def user_assessment_create(user_id: str):
    user, error = _user_for_access(user_id)
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    record = payload.get("record", {})
    if not isinstance(record, dict) or not record:
        return jsonify({"error": "缺少 record 字段"}), 400
    if record.get("storage_consent") is not True or not record.get(
        "consent_recorded_at"
    ):
        return (
            jsonify(
                {
                    "error": "写入用户评估记录前必须获得明确的本机存储同意"
                }
            ),
            400,
        )
    try:
        entry = user_records.save_assessment(user_id, record)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(entry), 201


@app.post("/api/imbalance/compare")
def imbalance_compare():
    report = services.run_imbalance_comparison()
    return jsonify(_records(report))


@app.get("/api/imbalance/report")
def imbalance_report():
    report = services.load_csv(services.IMBALANCE_REPORT)
    if report is None:
        return jsonify({"ready": False})
    return jsonify({"ready": True, "report": _records(report)})


@app.get("/api/mpdd/summary")
def mpdd_summary():
    return jsonify(services.mpdd_summary())


@app.get("/api/cmdc/summary")
def cmdc_summary():
    return jsonify(services.cmdc_summary())


@app.get("/api/cmdc/report")
def cmdc_report():
    report = services.load_csv(services.CMDC_REPORT)
    benchmark = services.read_json_file(services.CMDC_MODEL_SUMMARY)
    return jsonify(
        {
            "ready": report is not None,
            "report": _records(report) if report is not None else [],
            "summary": benchmark or {},
        }
    )


@app.post("/api/cmdc/build")
def cmdc_build():
    payload = request.get_json(silent=True) or {}
    try:
        summary = services.build_cmdc_dataset(
            password=payload.get("archive_password")
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(summary)


@app.post("/api/cmdc/train")
def cmdc_train():
    payload = request.get_json(silent=True) or {}
    try:
        summary = services.train_cmdc_external_models(
            repeats=max(1, min(30, int(payload.get("repeats", 10))))
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(summary)


@app.post("/api/cmdc/advanced")
def cmdc_advanced():
    payload = request.get_json(silent=True) or {}
    try:
        summary = services.run_cmdc_advanced_analysis(
            repeats=max(1, min(20, int(payload.get("repeats", 10)))),
            model_repeats=max(1, min(10, int(payload.get("model_repeats", 5)))),
            bootstrap_iterations=max(
                50,
                min(5000, int(payload.get("bootstrap_iterations", 1000))),
            ),
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(summary)


@app.post("/api/cmdc/holdout")
def cmdc_holdout():
    payload = request.get_json(silent=True) or {}
    try:
        summary = services.run_cmdc_locked_holdout(
            repeats=max(1, min(20, int(payload.get("repeats", 10)))),
            bootstrap_iterations=max(
                50,
                min(5000, int(payload.get("bootstrap_iterations", 1000))),
            ),
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(summary)


@app.get("/api/mpdd/cache_status")
def mpdd_cache_status():
    return jsonify(services.mpdd_cache_status())


@app.get("/api/synthetic/status")
def synthetic_status():
    return jsonify(services.synthetic_status())


@app.post("/api/synthetic/generate")
def synthetic_generate():
    payload = request.get_json(silent=True) or {}
    report = services.generate_synthetic(
        dialogue_n=int(payload.get("dialogue_n", 80)),
        tabular_n=int(payload.get("tabular_n", 400)),
        multimodal_n=int(payload.get("multimodal_n", 300)),
        seed=int(payload.get("seed", 42)),
    )
    return jsonify(report)


@app.post("/api/synthetic/delete")
def synthetic_delete():
    removed = services.delete_synthetic()
    return jsonify({"deleted": removed})


@app.post("/api/synthetic/dialogue/validate")
def synthetic_dialogue_validate():
    return jsonify(services.validate_dialogue_samples())


@app.post("/api/real_profile/generate")
def real_profile_generate():
    return jsonify(services.generate_real_profile())


@app.get("/api/real_profile/status")
def real_profile_status():
    return jsonify(services.real_profile_status())


@app.post("/api/synthetic/v3/run")
def synthetic_v3_run():
    payload = request.get_json(silent=True) or {}
    result = services.run_synthetic_v3(
        n_subjects=int(payload.get("n_subjects", 120)),
        segments_per_subject=int(payload.get("segments_per_subject", 5)),
    )
    return jsonify(result)


@app.post("/api/synthetic/v3/grid")
def synthetic_v3_grid():
    result = services.run_synthetic_v3_grid()
    return jsonify(
        {
            "grid": _records(result["grid"]),
            "best": result["best"],
            "report": _records(result["report"]),
        }
    )


@app.post("/api/synthetic/v3/augment")
def synthetic_v3_augment():
    payload = request.get_json(silent=True) or {}
    result = services.run_segment_augmentation(
        noise_scale=float(payload.get("noise_scale", 0.03)),
        k=int(payload.get("k", 3)),
    )
    return jsonify(result)


@app.post("/api/synthetic/v3/augment/optimize")
def synthetic_v3_augment_optimize():
    result = services.run_augmentation_optimization()
    return jsonify(
        {
            "tuning_grid": _records(result["tuning_grid"]),
            "best": result["best"],
            "final_report": _records(result["final_report"]),
            "params": result["params"],
        }
    )


@app.post("/api/synthetic/v3/augment/repeated")
def synthetic_v3_augment_repeated():
    payload = request.get_json(silent=True) or {}
    result = services.run_repeated_holdout(n_repeats=int(payload.get("n_repeats", 5)))
    return jsonify(
        {
            "report": _records(result["report"]),
            "summary": _records(result["summary"]),
        }
    )


@app.post("/api/synthetic/train/tabular")
def synthetic_train_tabular():
    report = services.train_tabular_synthetic()
    return jsonify(_records(report))


@app.get("/api/synthetic/v3/status")
def synthetic_v3_status():
    return jsonify(services.synthetic_v3_status())


@app.post("/api/mpdd/build_cache")
def mpdd_build_cache():
    payload = request.get_json(silent=True) or {}
    report = services.build_mpdd_feature_cache(force=bool(payload.get("force", False)))
    return jsonify(report)


@app.post("/api/mpdd/build")
def mpdd_build():
    services.build_mpdd_dataset()
    return jsonify(services.mpdd_summary())


@app.post("/api/mpdd/build_utterances")
def mpdd_build_utterances():
    services.build_mpdd_utterance_dataset()
    return jsonify({"ready": True, "path": str(services.MPDD_UTTERANCE_CSV)})


@app.post("/api/mpdd/train_subject")
def mpdd_train_subject():
    report = services.train_mpdd_subject_baseline()
    return jsonify(_records(report))


@app.post("/api/mpdd/train_advanced")
def mpdd_train_advanced():
    result = services.train_mpdd_subject_advanced()
    return jsonify(
        {
            "label3": _records(result["label3"]),
            "phq9": _records(result["phq9"]),
        }
    )


@app.get("/api/mpdd/subject_report")
def mpdd_subject_report():
    report = services.load_csv(services.MULTIMODAL_SUBJECT_REPORT)
    if report is None:
        return jsonify({"ready": False})
    return jsonify({"ready": True, "report": _records(report)})


@app.get("/api/mpdd/advanced_report")
def mpdd_advanced_report():
    label3 = services.load_csv(services.MULTIMODAL_SUBJECT_LABEL3_REPORT)
    phq9 = services.load_csv(services.MULTIMODAL_SUBJECT_PHQ9_REPORT)
    return jsonify(
        {
            "label3_ready": label3 is not None,
            "phq9_ready": phq9 is not None,
            "label3": _records(label3) if label3 is not None else [],
            "phq9": _records(phq9) if phq9 is not None else [],
        }
    )


@app.post("/api/mpdd/export_test_predictions")
def mpdd_export_test_predictions():
    pred = services.export_mpdd_test_predictions()
    return jsonify(_records(pred))


@app.post("/api/mpdd/upload")
def mpdd_upload():
    if "file" not in request.files:
        return jsonify({"error": "缺少 file 字段"}), 400
    file = request.files["file"]
    services.MPDD_UPLOAD_CSV.parent.mkdir(parents=True, exist_ok=True)
    file.save(services.MPDD_UPLOAD_CSV)
    return jsonify(services.mpdd_summary())


@app.post("/api/mpdd/train")
def mpdd_train():
    payload = request.get_json(silent=True) or {}
    threshold = int(payload.get("threshold", 8))
    path = services.MPDD_UPLOAD_CSV if services.MPDD_UPLOAD_CSV.exists() else services.MPDD_CSV
    if not path.exists():
        return jsonify({"error": "没有可用的 MPDD 特征文件"}), 400
    df = pd.read_csv(path)
    report = services.train_multimodal(df, threshold=threshold)
    return jsonify(_records(report))


@app.get("/api/mpdd/report")
def mpdd_report():
    report = services.load_csv(services.MULTIMODAL_REPORT)
    if report is None:
        return jsonify({"ready": False})
    return jsonify({"ready": True, "report": _records(report)})


@app.post("/api/assessment")
def assessment():
    payload = request.get_json(silent=True) or {}
    try:
        normalized = _validated_assessment_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": 400}), 400
    result = services.run_assessment(**normalized)
    return jsonify(result)


@app.post("/api/scales/score")
def scale_score():
    payload = request.get_json(silent=True) or {}
    try:
        scale, items = _validated_scale_items(
            payload.get("scale", "cesd10"),
            payload.get("items", []),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": 400}), 400
    funcs = {
        "cesd10": score_cesd10,
        "phq9": score_phq9,
        "gds15": score_gds15,
        "hamd24": score_hamd24,
    }
    score = funcs[scale](items)
    return jsonify({"score": score, "level": severity_level(score, scale)})


@app.post("/api/agent/chat")
def agent_chat():
    payload = request.get_json(silent=True) or {}
    messages = payload.get("messages", [])
    context = payload.get("context", {})
    session_id = payload.get("session_id") or context.get("session_id")
    reply = agent.chat(messages, context, session_id=session_id)
    snapshot = agent.snapshot(context, session_id=session_id)
    return jsonify(
        {
            "reply": reply,
            "next_question": agent.next_question(context, session_id=session_id),
            "agent": "adaptive-interviewer" if snapshot["mode"] == "interview" else "companion",
            "mode": snapshot["mode"],
            "topic": snapshot["topic"],
            "risk_cues": snapshot["risk_cues"],
            "cue_confidence": snapshot["cue_confidence"],
            "covered_topics": snapshot["covered_topics"],
            "safety_alert": snapshot["safety_alert"],
            "engine": snapshot["engine"],
            "llm_error": snapshot["llm_error"],
            "memory": snapshot.get("memory", {}),
            "memory_summary": snapshot.get("memory_summary", ""),
            "latency_ms": snapshot.get("latency_ms", 0),
            "fastpath_reason": snapshot.get("fastpath_reason", ""),
            "clinical_alert": snapshot.get("clinical_alert"),
            "scale_evidence": snapshot.get("scale_evidence", []),
            "focus": snapshot.get("focus", ""),
            "focus_reason": snapshot.get("focus_reason", ""),
            "positive_context": snapshot.get("positive_context", ""),
            "addressed_issues": snapshot.get("addressed_issues", []),
            "suggested_topics": snapshot.get("suggested_topics", []),
            "protective_factors": snapshot.get("protective_factors", []),
            "alliance_repair_count": snapshot.get("alliance_repair_count", 0),
            "last_alliance_repair": snapshot.get("last_alliance_repair", ""),
            "accepted_actions": snapshot.get("accepted_actions", []),
            "last_guidance_ids": snapshot.get("last_guidance_ids", []),
            "last_guidance_titles": snapshot.get("last_guidance_titles", []),
            "safety_plan_status": snapshot.get("safety_plan_status", ""),
            "clinical_findings": snapshot.get("clinical_findings", {}),
            "clinical_summary": snapshot.get("clinical_summary", ""),
            "support_preference": snapshot.get("support_preference", ""),
            "general_medical_id": snapshot.get("general_medical_id", ""),
            "general_medical_title": snapshot.get("general_medical_title", ""),
            "general_medical_question": snapshot.get("general_medical_question", ""),
            "medical_sources": snapshot.get("medical_sources", []),
        }
    )


@app.post("/api/agent/reset")
def agent_reset():
    payload = request.get_json(silent=True) or {}
    removed = agent.reset_session(
        session_id=payload.get("session_id"),
        mode=payload.get("mode"),
    )
    return jsonify({"removed": removed})


@app.get("/api/agent/schema")
def agent_schema():
    return jsonify(AGENT_CONTRACT)


@app.get("/api/models/registry")
def models_registry():
    return jsonify(services.model_registry())


def _job_tasks() -> dict:
    return {
        "echo": lambda p: {"echo": p},
        "tabular_train": lambda p: _records(
            services.train_tabular_baseline(p.get("resample_method", "smote_tomek"))
        ),
        "tabular_temporal": lambda p: _records(
            services.train_charls_temporal(p.get("resample_method", "smote_tomek"))
        ),
        "tabular_longitudinal": lambda p: _records(
            services.train_charls_longitudinal(p.get("resample_method", "smote_tomek"))
        ),
        "tabular_synthetic": lambda p: _records(services.train_tabular_synthetic()),
        "imbalance": lambda p: _records(services.run_imbalance_comparison()),
        "mpdd_build": lambda p: services.mpdd_summary(),
        "mpdd_augmentation": lambda p: {
            "validation": services.run_segment_augmentation(
                noise_scale=float(p.get("noise_scale", 0.03)),
                k=int(p.get("k", 3)),
            )["validation"],
        },
        "synthetic_generate": lambda p: services.generate_synthetic(
            dialogue_n=int(p.get("dialogue_n", 500)),
            tabular_n=int(p.get("tabular_n", 1000)),
            multimodal_n=int(p.get("multimodal_n", 300)),
            seed=int(p.get("seed", 42)),
        ),
        "real_profile": lambda p: services.generate_real_profile(),
        "fusion_optimize": lambda p: _fusion_optimize(p),
    }


def _fusion_optimize(payload: dict) -> dict:
    result = services.run_augmentation_optimization()
    return {
        "best": result["best"],
        "params": result["params"],
        "final_report": _records(result["final_report"]),
        "tuning_grid": _records(result["tuning_grid"]),
    }


def _job_dedupe_key(task: str, payload: dict) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(f"{task}:{canonical}".encode("utf-8")).hexdigest()


@app.post("/api/jobs")
def jobs_create():
    payload = request.get_json(silent=True) or {}
    tasks = _job_tasks()
    task = payload.get("task")
    if task not in tasks:
        return jsonify({"error": "未知任务", "code": 400}), 400
    task_payload = payload.get("payload", {})
    if not isinstance(task_payload, dict):
        return jsonify({"error": "payload 必须是对象", "code": 400}), 400
    try:
        timeout = float(payload.get("timeout", 3600))
    except (TypeError, ValueError):
        return jsonify({"error": "timeout 必须是数字", "code": 400}), 400
    if not 30 <= timeout <= 86400:
        return jsonify(
            {"error": "timeout 必须在 30 到 86400 秒之间", "code": 400}
        ), 400
    job_id, created = job_runner.submit(
        tasks[task],
        task_payload,
        name=task,
        timeout=timeout,
        dedupe_key=_job_dedupe_key(task, task_payload),
    )
    return (
        jsonify(
            {
                "job_id": job_id,
                "deduplicated": not created,
            }
        ),
        202 if created else 200,
    )


@app.get("/api/jobs")
def jobs_list():
    return jsonify(job_runner.list_jobs())


@app.get("/api/jobs/<job_id>")
def jobs_get(job_id: str):
    job = job_runner.get(job_id)
    if job is None:
        return jsonify({"error": "任务不存在", "code": 404}), 404
    return jsonify(job)


@app.post("/api/jobs/<job_id>/cancel")
def jobs_cancel(job_id: str):
    cancelled = job_runner.cancel(job_id)
    if not cancelled:
        return jsonify({"error": "任务不存在或已结束", "code": 400}), 400
    return jsonify({"cancelled": True})


@app.errorhandler(400)
def bad_request(exc):
    return jsonify({"error": "请求参数错误", "code": 400}), 400


@app.errorhandler(404)
def not_found(exc):
    return jsonify({"error": "接口不存在", "code": 404}), 404


@app.errorhandler(405)
def method_not_allowed(exc):
    return jsonify({"error": "请求方法不允许", "code": 405}), 405


@app.errorhandler(Exception)
def handle_error(exc):
    app.logger.exception(exc)
    return jsonify({"error": str(exc), "code": 500}), 500


for _rule in list(app.url_map.iter_rules()):
    _rule_str = str(_rule.rule)
    if _rule_str.startswith("/api/") and not _rule_str.startswith("/api/v1/"):
        _versioned = "/api/v1" + _rule_str[len("/api"):]
        app.add_url_rule(
            _versioned,
            endpoint=f"{_rule.endpoint}_v1",
            view_func=app.view_functions[_rule.endpoint],
            methods=list(_rule.methods - {"HEAD", "OPTIONS"}),
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001, debug=False, threaded=True)
