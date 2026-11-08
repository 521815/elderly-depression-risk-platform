"""Streamlit client for the Flask API, with local fallback when the API is offline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression import services as _local
from elderly_depression import records as _local_records

API_BASE = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")

CHARLS_CSV = _local.CHARLS_CSV
MPDD_CSV = _local.MPDD_CSV
MPDD_UPLOAD_CSV = _local.MPDD_UPLOAD_CSV
MPDD_UTTERANCE_CSV = _local.MPDD_UTTERANCE_CSV
CMDC_CSV = _local.CMDC_CSV
CMDC_AUDIT_JSON = _local.CMDC_AUDIT_JSON
CMDC_REPORT = _local.CMDC_REPORT
CMDC_MODEL_SUMMARY = _local.CMDC_MODEL_SUMMARY
CMDC_MODEL_DIR = _local.CMDC_MODEL_DIR
MODEL_REPORT = _local.MODEL_REPORT
IMBALANCE_REPORT = _local.IMBALANCE_REPORT
MULTIMODAL_REPORT = _local.MULTIMODAL_REPORT
MULTIMODAL_SUBJECT_REPORT = _local.MULTIMODAL_SUBJECT_REPORT
MULTIMODAL_SUBJECT_LABEL3_REPORT = _local.MULTIMODAL_SUBJECT_LABEL3_REPORT
MULTIMODAL_SUBJECT_PHQ9_REPORT = _local.MULTIMODAL_SUBJECT_PHQ9_REPORT
MPDD_TEST_PREDICTIONS = _local.MPDD_TEST_PREDICTIONS
MULTIMODAL_MODEL_DIR = _local.MULTIMODAL_MODEL_DIR
FUSION_DIR = _local.FUSION_DIR
FUSION_REPORT = _local.FUSION_REPORT
FUSION_CALIBRATION_CURVE = _local.FUSION_CALIBRATION_CURVE
CHARLS_TEMPORAL_REPORT = _local.CHARLS_TEMPORAL_REPORT
CHARLS_LONGITUDINAL_REPORT = _local.CHARLS_LONGITUDINAL_REPORT
DATA_DIR = _local.DATA_DIR


_LOCAL_AGENT = None
_AUTH_TOKEN: str | None = None


def set_auth_token(token: str | None) -> None:
    global _AUTH_TOKEN
    _AUTH_TOKEN = token or None


def get_auth_token() -> str | None:
    return _AUTH_TOKEN


def _can_use_local_fallback() -> bool:
    """Never bypass backend authorization once a login session exists."""
    return not _AUTH_TOKEN


def _api(method: str, path: str, **kwargs):
    try:
        headers = dict(kwargs.pop("headers", {}) or {})
        if _AUTH_TOKEN:
            headers.setdefault("Authorization", f"Bearer {_AUTH_TOKEN}")
        if headers:
            kwargs["headers"] = headers
        resp = requests.request(method, f"{API_BASE}{path}", timeout=180, **kwargs)
        if resp.status_code < 400:
            return resp.json()
    except requests.RequestException:
        return None
    return None


def _api_auth(method: str, path: str, **kwargs) -> dict:
    """Return a structured result for auth calls so the UI can show errors."""
    try:
        headers = dict(kwargs.pop("headers", {}) or {})
        if _AUTH_TOKEN:
            headers.setdefault("Authorization", f"Bearer {_AUTH_TOKEN}")
        if headers:
            kwargs["headers"] = headers
        resp = requests.request(method, f"{API_BASE}{path}", timeout=30, **kwargs)
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        if resp.status_code >= 400:
            return {
                "ok": False,
                "status": resp.status_code,
                "error": payload.get("error") or "请求失败",
            }
        return {"ok": True, "status": resp.status_code, **payload}
    except requests.RequestException:
        return {
            "ok": False,
            "status": 0,
            "error": "无法连接后端服务，请确认桌面版后端已启动",
        }


def auth_challenge() -> dict | None:
    data = _api_auth("GET", "/api/auth/challenge")
    return data if data.get("ok") else None


def auth_login(
    username: str,
    password: str,
    challenge_id: str,
    answer: str,
) -> dict:
    return _api_auth(
        "POST",
        "/api/auth/login",
        json={
            "username": username,
            "password": password,
            "challenge_id": challenge_id,
            "answer": answer,
        },
    )


def auth_register(
    username: str,
    password: str,
    display_name: str,
    challenge_id: str,
    answer: str,
    security_questions: list[str],
    security_answers: list[str],
) -> dict:
    return _api_auth(
        "POST",
        "/api/auth/register",
        json={
            "username": username,
            "password": password,
            "display_name": display_name,
            "challenge_id": challenge_id,
            "answer": answer,
            "security_questions": security_questions,
            "security_answers": security_answers,
        },
    )


def auth_me() -> dict | None:
    data = _api_auth("GET", "/api/auth/me")
    return data.get("account") if data.get("ok") else None


def auth_logout() -> None:
    _api_auth("POST", "/api/auth/logout")


def auth_security_options() -> list[str]:
    data = _api_auth("GET", "/api/auth/security-options")
    return data.get("questions", []) if data.get("ok") else []


def auth_recovery_questions(username: str) -> dict:
    return _api_auth(
        "POST",
        "/api/auth/recovery/questions",
        json={"username": username},
    )


def auth_recovery_reset(
    username: str,
    answers: list[str],
    new_password: str,
) -> dict:
    return _api_auth(
        "POST",
        "/api/auth/recovery/reset",
        json={
            "username": username,
            "answers": answers,
            "new_password": new_password,
        },
    )


def update_own_profile(
    current_password: str,
    *,
    username: str | None = None,
    display_name: str | None = None,
    new_password: str | None = None,
    security_questions: list[str] | None = None,
    security_answers: list[str] | None = None,
) -> dict:
    payload = {
        "current_password": current_password,
        "username": username,
        "display_name": display_name,
        "new_password": new_password,
        "security_questions": security_questions,
        "security_answers": security_answers,
    }
    return _api_auth(
        "PATCH",
        "/api/account/profile",
        json={key: value for key, value in payload.items() if value is not None},
    )


def admin_list_accounts() -> list[dict]:
    data = _api_auth("GET", "/api/admin/accounts")
    return data.get("accounts", []) if data.get("ok") else []


def admin_create_account(
    username: str,
    password: str,
    role: str,
    display_name: str = "",
) -> dict:
    return _api_auth(
        "POST",
        "/api/admin/accounts",
        json={
            "username": username,
            "password": password,
            "role": role,
            "display_name": display_name,
        },
    )


def admin_update_account(
    account_id: str,
    *,
    role: str | None = None,
    status: str | None = None,
    display_name: str | None = None,
    password: str | None = None,
) -> dict:
    payload = {
        key: value
        for key, value in {
            "role": role,
            "status": status,
            "display_name": display_name,
            "password": password,
        }.items()
        if value is not None
    }
    return _api_auth("PATCH", f"/api/admin/accounts/{account_id}", json=payload)


def charls_exists() -> bool:
    status = _api("GET", "/api/status")
    if status is not None:
        return bool(status.get("charls_ready", False))
    return _local.charls_exists()


def mpdd_exists() -> bool:
    status = _api("GET", "/api/status")
    if status is not None:
        return bool(status.get("mpdd_ready", False))
    return _local.mpdd_exists()


def cmdc_exists() -> bool:
    status = _api("GET", "/api/status")
    if status is not None:
        return bool(status.get("cmdc_ready", False))
    return _local.cmdc_exists()


def status() -> dict:
    remote = _api("GET", "/api/status")
    if remote is not None:
        return {"connected": True, **remote}
    local = _local.status()
    local["backend_version"] = _local.code_signature()
    return {"connected": False, **local}


def backend_version() -> dict:
    remote = _api("GET", "/api/version")
    if remote is not None:
        return remote
    return {
        "backend_version": _local.code_signature(),
        "data_signature": _local.data_signature(),
        "backend_started_at": "",
    }


def data_health() -> dict:
    data = _api("GET", "/api/data_health")
    if data is not None:
        return data
    return _local.data_health_report()


def load_charls() -> pd.DataFrame:
    return _local.load_charls()


def load_csv(path: Path) -> pd.DataFrame | None:
    return _local.load_csv(path)


def build_charls_dataset():
    data = _api("POST", "/api/charls/build")
    if data is not None:
        return data
    return _local.build_charls_dataset()


def build_mpdd_dataset():
    data = _api("POST", "/api/mpdd/build")
    if data is not None:
        return data
    return _local.build_mpdd_dataset()


def build_mpdd_utterance_dataset():
    data = _api("POST", "/api/mpdd/build_utterances")
    if data is not None:
        return data
    return _local.build_mpdd_utterance_dataset()


def cmdc_summary() -> dict:
    data = _api("GET", "/api/cmdc/summary")
    if data is not None:
        return data
    return _local.cmdc_summary()


def build_cmdc_dataset(archive_password: str) -> dict:
    data = _api(
        "POST",
        "/api/cmdc/build",
        json={"archive_password": archive_password},
        timeout=300,
    )
    if data is not None:
        return data
    return _local.build_cmdc_dataset(password=archive_password)


def cmdc_report() -> tuple[pd.DataFrame | None, dict]:
    data = _api("GET", "/api/cmdc/report")
    if data is not None:
        report = pd.DataFrame(data.get("report", []))
        if report.empty:
            report = None
        return report, data.get("summary") or {}
    report = _local.load_csv(_local.CMDC_REPORT)
    return report, _local.read_json_file(_local.CMDC_MODEL_SUMMARY) or {}


def train_cmdc_models(repeats: int = 10) -> dict:
    data = _api(
        "POST",
        "/api/cmdc/train",
        json={"repeats": repeats},
        timeout=300,
    )
    if data is not None:
        return data
    return _local.train_cmdc_external_models(repeats=repeats)


def mpdd_cache_status() -> dict:
    data = _api("GET", "/api/mpdd/cache_status")
    if data is not None:
        return data
    return _local.mpdd_cache_status()


def build_mpdd_cache(force: bool = False) -> dict:
    data = _api("POST", "/api/mpdd/build_cache", json={"force": force})
    if data is not None:
        return data
    return _local.build_mpdd_feature_cache(force=force)


def synthetic_status() -> dict:
    data = _api("GET", "/api/synthetic/status")
    if data is not None:
        return data
    return _local.synthetic_status()


def generate_synthetic(
    dialogue_n: int = 80,
    tabular_n: int = 400,
    multimodal_n: int = 300,
    seed: int = 42,
) -> dict:
    data = _api(
        "POST",
        "/api/synthetic/generate",
        json={
            "dialogue_n": dialogue_n,
            "tabular_n": tabular_n,
            "multimodal_n": multimodal_n,
            "seed": seed,
        },
    )
    if data is not None:
        return data
    return _local.generate_synthetic(
        dialogue_n=dialogue_n,
        tabular_n=tabular_n,
        multimodal_n=multimodal_n,
        seed=seed,
    )


def delete_synthetic() -> bool:
    data = _api("POST", "/api/synthetic/delete")
    if data is not None:
        return bool(data.get("deleted", False))
    return _local.delete_synthetic()


def validate_dialogue_samples() -> dict:
    data = _api("POST", "/api/synthetic/dialogue/validate")
    if data is not None:
        return data
    return _local.validate_dialogue_samples()


def real_profile_status() -> dict:
    data = _api("GET", "/api/real_profile/status")
    if data is not None:
        return data
    return _local.real_profile_status()


def generate_real_profile() -> dict:
    data = _api("POST", "/api/real_profile/generate")
    if data is not None:
        return data
    return _local.generate_real_profile()


def synthetic_v3_status() -> dict:
    data = _api("GET", "/api/synthetic/v3/status")
    if data is not None:
        return data
    return _local.synthetic_v3_status()


def run_synthetic_v3(n_subjects: int = 120, segments_per_subject: int = 5) -> dict:
    data = _api(
        "POST",
        "/api/synthetic/v3/run",
        json={"n_subjects": n_subjects, "segments_per_subject": segments_per_subject},
    )
    if data is not None:
        return data
    return _local.run_synthetic_v3(
        n_subjects=n_subjects,
        segments_per_subject=segments_per_subject,
    )


def run_synthetic_v3_grid() -> dict:
    data = _api("POST", "/api/synthetic/v3/grid")
    if data is not None:
        return data
    result = _local.run_synthetic_v3_grid()
    return {
        "grid": result["grid"],
        "best": result["best"],
        "report": result["report"],
    }


def run_segment_augmentation(noise_scale: float = 0.03, k: int = 3) -> dict:
    data = _api(
        "POST",
        "/api/synthetic/v3/augment",
        json={"noise_scale": noise_scale, "k": k},
    )
    if data is not None:
        return data
    return _local.run_segment_augmentation(noise_scale=noise_scale, k=k)


def run_augmentation_optimization() -> dict:
    data = _api("POST", "/api/synthetic/v3/augment/optimize")
    if data is not None:
        return data
    result = _local.run_augmentation_optimization()
    return {
        "tuning_grid": result["tuning_grid"],
        "best": result["best"],
        "final_report": result["final_report"],
        "params": result["params"],
    }


def run_repeated_holdout(n_repeats: int = 5) -> dict:
    data = _api("POST", "/api/synthetic/v3/augment/repeated", json={"n_repeats": n_repeats})
    if data is not None:
        return data
    result = _local.run_repeated_holdout(n_repeats=n_repeats)
    return {"report": result["report"], "summary": result["summary"]}


def submit_job(task: str, payload: dict | None = None) -> dict:
    data = _api("POST", "/api/jobs", json={"task": task, "payload": payload or {}})
    if data is not None:
        return data
    return {"error": "后台任务服务不可用"}


def list_jobs() -> list[dict]:
    data = _api("GET", "/api/jobs")
    if data is not None:
        return data
    return []


def cancel_job(job_id: str) -> bool:
    data = _api("POST", f"/api/jobs/{job_id}/cancel")
    if data is not None:
        return bool(data.get("cancelled", False))
    return False


def train_tabular_synthetic() -> pd.DataFrame:
    data = _api("POST", "/api/synthetic/train/tabular")
    if data is not None:
        return pd.DataFrame(data)
    return _local.train_tabular_synthetic()


def train_mpdd_subject() -> pd.DataFrame:
    data = _api("POST", "/api/mpdd/train_subject")
    if data is not None:
        return pd.DataFrame(data)
    return _local.train_mpdd_subject_baseline()


def train_mpdd_advanced() -> dict:
    data = _api("POST", "/api/mpdd/train_advanced")
    if data is not None:
        return {
            "label3": pd.DataFrame(data.get("label3", [])),
            "phq9": pd.DataFrame(data.get("phq9", [])),
        }
    return _local.train_mpdd_subject_advanced()


def export_mpdd_test_predictions() -> pd.DataFrame:
    data = _api("POST", "/api/mpdd/export_test_predictions")
    if data is not None:
        return pd.DataFrame(data)
    return _local.export_mpdd_test_predictions()


def predict_mpdd_subjects(df: pd.DataFrame) -> pd.DataFrame:
    import joblib

    from elderly_depression.models.multimodal import predict_subject_pipeline

    pipeline_path = MULTIMODAL_MODEL_DIR / "subject_fusion_pipeline.pkl"
    pipeline = joblib.load(pipeline_path)
    return predict_subject_pipeline(pipeline, df)


def train_tabular_baseline(method: str = "smote_tomek") -> pd.DataFrame:
    data = _api("POST", "/api/charls/train", json={"resample_method": method})
    if data is not None:
        return pd.DataFrame(data)
    return _local.train_tabular_baseline(method)


def train_charls_temporal(method: str = "smote_tomek") -> pd.DataFrame:
    data = _api("POST", "/api/charls/train_temporal", json={"resample_method": method})
    if data is not None:
        return pd.DataFrame(data)
    return _local.train_charls_temporal(method)


def charls_temporal_report() -> pd.DataFrame | None:
    data = _api("GET", "/api/charls/temporal_report")
    if data is not None:
        if not data.get("ready"):
            return None
        return pd.DataFrame(data["report"])
    return _local.load_csv(_local.CHARLS_TEMPORAL_REPORT)


def train_charls_longitudinal(method: str = "smote_tomek") -> pd.DataFrame:
    data = _api("POST", "/api/charls/train_longitudinal", json={"resample_method": method})
    if data is not None:
        return pd.DataFrame(data)
    return _local.train_charls_longitudinal(method)


def charls_longitudinal_report() -> pd.DataFrame | None:
    data = _api("GET", "/api/charls/longitudinal_report")
    if data is not None:
        if not data.get("ready"):
            return None
        return pd.DataFrame(data["report"])
    return _local.load_csv(_local.CHARLS_LONGITUDINAL_REPORT)


def fusion_status() -> dict:
    data = _api("GET", "/api/fusion/status")
    if data is not None:
        return data
    from elderly_depression.models.fusion import load_weights

    weights = load_weights(_local.FUSION_WEIGHTS_JSON)
    if weights is None:
        return {"ready": False, "mode": "literature_prior"}
    return {
        "ready": True,
        "mode": "learned",
        "trained_at": weights.get("trained_at"),
        "variants": weights.get("variants", {}),
    }


def fusion_components() -> dict:
    data = _api("GET", "/api/fusion/components")
    if data is not None:
        return data
    return _local.fusion_components()


def research_reports() -> dict:
    data = _api("GET", "/api/research/reports")
    if data is not None:
        return data
    return _local.research_reports()


def run_subgroup_shrinkage() -> dict | None:
    return _api("POST", "/api/smoothing/subgroup")


def run_robust_aggregation_ablation(
    repeats: int = 2,
    splits: int = 5,
    pca_dims: int = 32,
) -> dict | None:
    return _api(
        "POST",
        "/api/smoothing/aggregation",
        json={"repeats": repeats, "splits": splits, "pca_dims": pca_dims},
    )


def fusion_report() -> pd.DataFrame | None:
    data = _api("GET", "/api/fusion/report")
    if data is not None:
        if not data.get("ready"):
            return None
        return pd.DataFrame(data["report"])
    return _local.load_csv(_local.FUSION_REPORT)


def fusion_calibration_curve() -> pd.DataFrame | None:
    return _local.load_csv(_local.FUSION_CALIBRATION_CURVE)


def train_fusion(rows: list[dict], label_col: str = "depression") -> dict:
    data = _api("POST", "/api/fusion/train", json={"rows": rows, "label_col": label_col})
    if data is not None:
        return data
    from elderly_depression.models.fusion import train_fusion as local_train

    result = local_train(pd.DataFrame(rows), label_col=label_col, output_dir=_local.FUSION_DIR)
    return {"ready": True, "report": result["report"], "status": _local.fusion_status()}


def list_users(query: str | None = None) -> list[dict]:
    data = _api("GET", "/api/users", params={"q": query} if query else None)
    if data is not None:
        return data
    if not _can_use_local_fallback():
        return []
    return _local_records.list_users(query=query)


def create_user(name: str, **profile) -> dict:
    data = _api("POST", "/api/users", json={"name": name, **profile})
    if data is not None:
        return data
    if not _can_use_local_fallback():
        raise RuntimeError("登录状态已失效，请重新登录后再保存档案")
    return _local_records.create_user(name, **profile)


def update_user(user_id: str, **profile) -> dict | None:
    data = _api("PUT", f"/api/users/{user_id}", json=profile)
    if data is not None:
        return data
    if not _can_use_local_fallback():
        raise RuntimeError("登录状态已失效，请重新登录后再修改档案")
    return _local_records.update_user(user_id, **profile)


def delete_user(user_id: str) -> bool:
    data = _api("DELETE", f"/api/users/{user_id}")
    if data is not None:
        return bool(data.get("deleted", False))
    if not _can_use_local_fallback():
        return False
    return _local_records.delete_user(user_id)


def list_assessments(
    user_id: str,
    risk: str | None = None,
    scale: str | None = None,
    limit: int | None = 200,
) -> list[dict]:
    params = {}
    if limit is not None:
        params["limit"] = limit
    if risk:
        params["risk"] = risk
    if scale:
        params["scale"] = scale
    data = _api("GET", f"/api/users/{user_id}/assessments", params=params or None)
    if data is not None:
        return data
    if not _can_use_local_fallback():
        return []
    return _local_records.list_assessments(user_id, risk=risk, scale=scale, limit=limit)


def save_assessment(user_id: str, record: dict) -> dict:
    data = _api("POST", f"/api/users/{user_id}/assessments", json={"record": record})
    if data is not None:
        return data
    if not _can_use_local_fallback():
        raise RuntimeError("登录状态已失效，请重新登录后再保存评估记录")
    return _local_records.save_assessment(user_id, record)


def export_users() -> dict:
    data = _api("GET", "/api/users/export")
    if data is not None:
        return data
    if not _can_use_local_fallback():
        raise RuntimeError("登录状态已失效，请重新登录后再导出")
    return _local_records.export_all()


def import_users(payload: dict, mode: str = "merge") -> dict:
    data = _api("POST", "/api/users/import", json={"backup": payload, "mode": mode})
    if data is not None:
        return data
    if not _can_use_local_fallback():
        raise RuntimeError("登录状态已失效，请重新登录后再导入")
    return _local_records.import_backup(payload, mode=mode)


def run_imbalance_comparison() -> pd.DataFrame:
    data = _api("POST", "/api/imbalance/compare")
    if data is not None:
        return pd.DataFrame(data)
    return _local.run_imbalance_comparison()


def run_assessment(
    transcript: list[dict[str, str]],
    scale_answers: dict[str, list[float | int]],
    demographics: dict,
    expression_summary: dict,
    scale_name: str,
    mpdd_risk: float | None = None,
    language_cues: list[dict | str] | None = None,
    llm_complete=None,
) -> dict:
    payload = {
        "transcript": transcript,
        "scale_answers": scale_answers,
        "demographics": demographics,
        "expression_summary": expression_summary,
        "scale_name": scale_name,
        "mpdd_risk": mpdd_risk,
        "language_cues": language_cues,
    }
    data = _api("POST", "/api/assessment", json=payload)
    if data is not None:
        return data
    return _local.run_assessment(
        transcript=transcript,
        scale_answers=scale_answers,
        demographics=demographics,
        expression_summary=expression_summary,
        scale_name=scale_name,
        mpdd_risk=mpdd_risk,
        language_cues=language_cues,
        llm_complete=llm_complete,
    )


def expression_risk_from_openface(df: pd.DataFrame) -> dict:
    return _local.expression_risk_from_openface(df)


def agent_chat(messages: list[dict[str, str]], context: dict, session_id: str | None = None) -> dict:
    payload = {
        "session_id": session_id,
        "messages": messages,
        "context": context,
    }
    data = _api("POST", "/api/agent/chat", json=payload)
    if data is not None:
        return data

    global _LOCAL_AGENT
    if _LOCAL_AGENT is None:
        from backend_api.agent_adapter import AgentAdapter

        _LOCAL_AGENT = AgentAdapter()
        _LOCAL_AGENT.warmup()
    _LOCAL_AGENT.set_api_key(context.get("api_key"))
    reply = _LOCAL_AGENT.chat(messages, context, session_id=session_id)
    snapshot = _LOCAL_AGENT.snapshot(context, session_id=session_id)
    return {
        "reply": reply,
        "next_question": _LOCAL_AGENT.next_question(context, session_id=session_id),
        "agent": "adaptive-interviewer-local" if snapshot["mode"] == "interview" else "companion-local",
        "mode": snapshot["mode"],
        "topic": snapshot["topic"],
        "risk_cues": snapshot["risk_cues"],
        "cue_confidence": snapshot["cue_confidence"],
        "covered_topics": snapshot["covered_topics"],
        "safety_alert": snapshot["safety_alert"],
        "engine": getattr(_LOCAL_AGENT, "last_engine", "local"),
        "llm_error": getattr(_LOCAL_AGENT, "last_llm_error", None),
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


def agent_reset(session_id: str | None = None, mode: str | None = None) -> dict:
    data = _api(
        "POST",
        "/api/agent/reset",
        json={"session_id": session_id, "mode": mode},
    )
    if data is not None:
        return data
    if _LOCAL_AGENT is not None:
        return {
            "removed": _LOCAL_AGENT.reset_session(
                session_id=session_id,
                mode=mode,
            )
        }
    return {"removed": 0}


def train_multimodal(df: pd.DataFrame, threshold: int = 8) -> pd.DataFrame:
    return _local.train_multimodal(df, threshold=threshold)
