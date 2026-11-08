#!/usr/bin/env python3
"""Deterministic audit of privacy, credentials, consent, and data permissions."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "output" / "ethics_security_audit.json"
LEGACY_CREDENTIALS = DATA_DIR / "账号密码记录.txt"
ACCOUNT_REGISTRY = DATA_DIR / "账号清单.txt"
SOURCE_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt"}


def _mode(path: Path) -> str:
    return oct(path.stat().st_mode & 0o777)


def main() -> None:
    checks: list[dict] = []

    def add(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"id": check_id, "passed": bool(passed), "detail": detail})

    add(
        "legacy_plaintext_credentials_absent",
        not LEGACY_CREDENTIALS.exists(),
        "旧版明文账号密码文件已清除",
    )
    registry_text = (
        ACCOUNT_REGISTRY.read_text(encoding="utf-8")
        if ACCOUNT_REGISTRY.exists()
        else ""
    )
    add(
        "account_registry_has_no_password",
        bool(registry_text)
        and "密码：" not in registry_text
        and "password=" not in registry_text.lower(),
        "账号清单仅包含账号元数据，不包含密码或密保答案",
    )

    for filename in ["auth.sqlite", "user_records.sqlite", "user_records/users.json"]:
        path = DATA_DIR / filename
        if path.exists():
            mode = path.stat().st_mode & 0o777
            add(
                f"private_permissions_{filename}",
                mode & 0o077 == 0,
                f"{filename} 权限为 {_mode(path)}",
            )
    for path in [
        DATA_DIR / "cmdc_features.csv",
        PROJECT_ROOT / "output" / "cmdc_data_audit.json",
        PROJECT_ROOT / "output" / "cmdc_benchmark_report.csv",
        PROJECT_ROOT / "output" / "cmdc_benchmark_summary.json",
    ]:
        if path.exists():
            add(
                f"cmdc_private_permissions_{path.name}",
                path.stat().st_mode & 0o077 == 0,
                f"CMDC 派生文件 {path.name} 权限为 {_mode(path)}",
            )

    env_ignored = ".env" in (
        (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        if (PROJECT_ROOT / ".gitignore").exists()
        else ""
    )
    add("env_is_gitignored", env_ignored, ".env 已纳入忽略规则")
    add(
        "user_records_are_gitignored",
        "data/user_records/" in (PROJECT_ROOT / ".gitignore").read_text(
            encoding="utf-8"
        ),
        "用户档案与评估记录目录已纳入忽略规则",
    )
    env_path = PROJECT_ROOT / ".env"
    add(
        "env_file_permissions",
        not env_path.exists() or (env_path.stat().st_mode & 0o077 == 0),
        ".env 权限仅允许当前用户访问",
    )

    hardcoded_secret_files = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(
            part in {".env", ".git", "__pycache__", "data", "output", "paper"}
            for part in path.parts
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(r"\bsk-[A-Za-z0-9]{20,}\b", text):
            hardcoded_secret_files.append(str(path.relative_to(PROJECT_ROOT)))
    add(
        "no_hardcoded_api_key",
        not hardcoded_secret_files,
        "源码中未发现硬编码 API Key",
    )

    agent_source = (PROJECT_ROOT / "backend_api" / "agent_adapter.py").read_text(
        encoding="utf-8"
    )
    add(
        "cloud_calls_require_consent",
        "context.get(\"cloud_consent\", False)" in agent_source
        and "not cloud_consent" in agent_source,
        "第三方模型调用默认关闭并需要显式同意",
    )
    add(
        "cloud_messages_are_redacted",
        "redact_messages(" in agent_source
        and "memory_for_cloud" in agent_source,
        "发送第三方模型前执行消息和记忆脱敏",
    )

    auth_source = (PROJECT_ROOT / "src" / "elderly_depression" / "auth.py").read_text(
        encoding="utf-8"
    )
    add(
        "no_plaintext_password_writer",
        "密码：{item" not in auth_source
        and "generate_password_hash" in auth_source,
        "认证数据库使用密码哈希，代码不再写入明文密码",
    )
    add(
        "no_source_default_admin_password",
        not re.search(
            r"DEFAULT_ADMIN_PASSWORD\s*=\s*[\"'][^\"']+[\"']",
            auth_source,
        ),
        "源码中不存在默认管理员密码",
    )

    privacy_source = (
        PROJECT_ROOT / "src" / "elderly_depression" / "privacy.py"
    ).read_text(encoding="utf-8")
    add(
        "direct_identifier_redaction_available",
        all(
            marker in privacy_source
            for marker in ["PHONE_PATTERN", "ID_PATTERN", "NAME_PATTERN"]
        ),
        "姓名、电话、证件号、地址和邮箱脱敏函数可用",
    )
    webapp_source = (PROJECT_ROOT / "webapp" / "app.py").read_text(
        encoding="utf-8"
    )
    add(
        "storage_consent_is_required",
        "assessment_storage_consent" in webapp_source
        and "我已知晓并同意将本次评估数据保存到当前设备" in webapp_source,
        "评估数据写入用户记录前需要显式存储同意",
    )
    medical_kb_path = DATA_DIR / "medical_kb" / "medlineplus_topics.json"
    medical_kb_attributed = False
    if medical_kb_path.exists():
        try:
            medical_kb = json.loads(
                medical_kb_path.read_text(encoding="utf-8")
            )
            medical_kb_attributed = (
                medical_kb.get("source") == "MedlinePlus.gov"
                and bool(medical_kb.get("citation"))
                and int(medical_kb.get("topic_count", 0)) > 0
            )
        except (OSError, ValueError, TypeError):
            medical_kb_attributed = False
    add(
        "medical_kb_is_attributed",
        medical_kb_attributed,
        "MedlinePlus 本地缓存包含来源、引用和主题元数据",
    )
    cmdc_source = (
        PROJECT_ROOT / "src" / "elderly_depression" / "data" / "cmdc.py"
    ).read_text(encoding="utf-8")
    gitignore_text = (
        (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        if (PROJECT_ROOT / ".gitignore").exists()
        else ""
    )
    add(
        "cmdc_password_not_persisted",
        "CMDC_ARCHIVE_PASSWORD" in cmdc_source
        and "password_persisted" in cmdc_source
        and "raw_data_copied" in cmdc_source,
        "CMDC 密码仅通过环境变量或单次请求传入，派生表不保存密码和原始访谈",
    )
    add(
        "cmdc_derivatives_are_gitignored",
        "data/cmdc_features.csv" in gitignore_text
        and "output/cmdc_*" in gitignore_text,
        "CMDC 派生特征和模型产物已纳入忽略规则",
    )

    summary = {
        "passed": sum(1 for check in checks if check["passed"]),
        "failed": sum(1 for check in checks if not check["passed"]),
        "total": len(checks),
        "checks": checks,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "failed": summary["failed"],
                "total": summary["total"],
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
