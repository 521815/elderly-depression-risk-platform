"""Account authentication, roles, and arithmetic human verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
AUTH_DB = DATA_DIR / "auth.sqlite"
CREDENTIALS_FILE = DATA_DIR / "账号清单.txt"
LEGACY_CREDENTIALS_FILE = DATA_DIR / "账号密码记录.txt"
_LOCK = threading.RLock()

SESSION_TTL_HOURS = 12
CHALLENGE_TTL_SECONDS = 10 * 60
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9@_&\-¥￥$#]{2,32}$")
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128
SECURITY_QUESTION_OPTIONS = [
    "您最喜欢的颜色是？",
    "您小时候居住的城市是？",
    "您最喜欢的一道菜是？",
    "您第一只宠物的名字是？",
]
SECURITY_QUESTION_COUNT = 3

DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "mintz")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD")
DEFAULT_USER_USERNAME = os.getenv("DEFAULT_USER_USERNAME", "careuser01")
DEFAULT_USER_PASSWORD = os.getenv("DEFAULT_USER_PASSWORD")

_CHALLENGES: dict[str, dict] = {}
_RANDOM = secrets.SystemRandom()
_CREDENTIALS: dict[str, dict] = {}
_CREDENTIALS_LOADED = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_credentials() -> None:
    global _CREDENTIALS_LOADED
    if _CREDENTIALS_LOADED:
        return
    _CREDENTIALS.clear()
    if CREDENTIALS_FILE.exists():
        blocks = CREDENTIALS_FILE.read_text(encoding="utf-8").split("---")
        for block in blocks:
            item: dict[str, str] = {}
            for line in block.splitlines():
                if "：" not in line:
                    continue
                key, value = line.split("：", 1)
                item[key.strip()] = value.strip()
            username = item.get("用户名")
            if not username:
                continue
            _CREDENTIALS[username.lower()] = {
                "username": username,
                "role": item.get("角色") or "普通用户",
                "status": item.get("状态") or "正常",
                "display_name": item.get("显示名称") or username,
                "created_at": item.get("创建时间") or "",
                "updated_at": item.get("更新时间") or "",
            }
    _CREDENTIALS_LOADED = True


def _role_label(role: str) -> str:
    return "管理员" if role == "admin" else "普通用户"


def _status_label(status: str) -> str:
    return "正常" if status == "active" else "已停用"


def _save_credentials() -> None:
    CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 老年抑郁风险预测大模型账号清单",
        "# 本文件不保存密码、密保答案、登录令牌或其他认证秘密。",
        f"# 更新时间：{_now()}",
        "",
    ]
    for item in sorted(_CREDENTIALS.values(), key=lambda value: value["username"].lower()):
        lines.extend(
            [
                "---",
                f"用户名：{item['username']}",
                f"角色：{item.get('role', '普通用户')}",
                f"状态：{item.get('status', '正常')}",
                f"显示名称：{item.get('display_name', item['username'])}",
                f"创建时间：{item.get('created_at', '')}",
                f"更新时间：{item.get('updated_at', '')}",
                "",
            ]
        )
    CREDENTIALS_FILE.write_text("\n".join(lines), encoding="utf-8")
    try:
        os.chmod(CREDENTIALS_FILE, 0o600)
    except OSError:
        pass


def _sync_credential(
    *,
    username: str,
    password: str | None,
    role: str,
    status: str,
    display_name: str,
    created_at: str,
    updated_at: str | None = None,
    previous_username: str | None = None,
) -> None:
    with _LOCK:
        _load_credentials()
        previous_key = (previous_username or username).lower()
        key = username.lower()
        item = _CREDENTIALS.pop(previous_key, {})
        item.update(
            {
                "username": username,
                "role": _role_label(role),
                "status": _status_label(status),
                "display_name": display_name or username,
                "created_at": created_at,
                "updated_at": updated_at or _now(),
            }
        )
        item.pop("password", None)
        _CREDENTIALS[key] = item
        _save_credentials()


def _ensure_credential_entries(conn: sqlite3.Connection) -> None:
    with _LOCK:
        _load_credentials()
        changed = False
        rows = conn.execute("SELECT * FROM accounts").fetchall()
        for row in rows:
            key = row["username_lower"]
            if key in _CREDENTIALS:
                continue
            _CREDENTIALS[key] = {
                "username": row["username"],
                "role": _role_label(row["role"]),
                "status": _status_label(row["status"]),
                "display_name": row["display_name"] or row["username"],
                "created_at": row["created_at"] or "",
                "updated_at": _now(),
            }
            changed = True
        if changed:
            _save_credentials()


def _purge_legacy_credentials() -> None:
    if not LEGACY_CREDENTIALS_FILE.exists():
        return
    try:
        size = LEGACY_CREDENTIALS_FILE.stat().st_size
        with LEGACY_CREDENTIALS_FILE.open("r+b") as handle:
            handle.write(b"\0" * size)
            handle.flush()
            os.fsync(handle.fileno())
        LEGACY_CREDENTIALS_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _get_conn() -> sqlite3.Connection:
    AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB))
    conn.row_factory = sqlite3.Row
    try:
        os.chmod(AUTH_DB, 0o600)
    except OSError:
        pass
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS accounts (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            username_lower TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'disabled')),
            display_name TEXT,
            created_at TEXT,
            created_by TEXT,
            last_login_at TEXT,
            security_question_1 TEXT,
            security_answer_hash_1 TEXT,
            security_question_2 TEXT,
            security_answer_hash_2 TEXT,
            security_question_3 TEXT,
            security_answer_hash_3 TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS auth_sessions (
            token_hash TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            created_at TEXT,
            expires_at TEXT,
            revoked_at TEXT,
            user_agent TEXT
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_auth_sessions_account ON auth_sessions(account_id)"
    )
    conn.commit()


def _ensure_db() -> None:
    _purge_legacy_credentials()
    with closing(_get_conn()) as conn:
        _init_db(conn)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(accounts)")]
        for column in [
            "security_question_1",
            "security_answer_hash_1",
            "security_question_2",
            "security_answer_hash_2",
            "security_question_3",
            "security_answer_hash_3",
        ]:
            if column not in columns:
                conn.execute(f"ALTER TABLE accounts ADD COLUMN {column} TEXT")
        _seed_defaults(conn)
        _ensure_credential_entries(conn)


def _account_to_dict(row: sqlite3.Row) -> dict:
    questions = [
        row["security_question_1"],
        row["security_question_2"],
        row["security_question_3"],
    ]
    questions = [question for question in questions if question]
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "status": row["status"],
        "display_name": row["display_name"] or row["username"],
        "created_at": row["created_at"],
        "created_by": row["created_by"],
        "last_login_at": row["last_login_at"],
        "security_questions": questions,
        "has_security_questions": len(questions) == SECURITY_QUESTION_COUNT,
    }


def username_error(username: str) -> str | None:
    value = str(username or "")
    if not value:
        return "请输入用户名"
    if len(value) > 32:
        return "用户名最多 32 位"
    if not USERNAME_PATTERN.fullmatch(value):
        return "用户名仅允许 @ _ - & ¥ $ # 和数字、大小写字母，长度 2-32 位"
    return None


def password_error(password: str) -> str | None:
    value = str(password or "")
    if len(value) < PASSWORD_MIN_LENGTH:
        return "密码至少 8 位"
    if len(value) > PASSWORD_MAX_LENGTH:
        return "密码不能超过 128 位"
    if any(ch.isspace() for ch in value):
        return "密码不能包含空格"
    if not re.search(r"[a-z]", value):
        return "密码必须包含至少一个小写字母"
    if not re.search(r"[A-Z]", value):
        return "密码必须包含至少一个大写字母"
    return None


def _normalize_security_answers(answers: list[str]) -> list[str]:
    return [str(answer or "").strip().lower() for answer in answers]


def _validate_security_payload(
    questions: list[str] | None,
    answers: list[str] | None,
) -> tuple[list[str], list[str]]:
    question_list = [str(item).strip() for item in (questions or [])]
    answer_list = [str(item).strip() for item in (answers or [])]
    if len(question_list) != SECURITY_QUESTION_COUNT:
        raise ValueError(f"请选择 {SECURITY_QUESTION_COUNT} 个密保问题")
    if len(answer_list) != SECURITY_QUESTION_COUNT:
        raise ValueError("请完整填写密保答案")
    if len(set(question_list)) != SECURITY_QUESTION_COUNT:
        raise ValueError("密保问题不能重复")
    for question in question_list:
        if question not in SECURITY_QUESTION_OPTIONS:
            raise ValueError("密保问题不在可选列表中")
    if any(not answer for answer in answer_list):
        raise ValueError("密保答案不能为空")
    return question_list, answer_list


def security_payload_error(
    questions: list[str] | None,
    answers: list[str] | None,
) -> str | None:
    try:
        _validate_security_payload(questions, answers)
    except ValueError as exc:
        return str(exc)
    return None


def create_account(
    username: str,
    password: str,
    *,
    role: str = "user",
    display_name: str | None = None,
    created_by: str | None = None,
    security_questions: list[str] | None = None,
    security_answers: list[str] | None = None,
) -> dict:
    if role not in {"admin", "user"}:
        raise ValueError("账号角色必须为 admin 或 user")
    if error := username_error(username):
        raise ValueError(error)
    if error := password_error(password):
        raise ValueError(error)
    questions: list[str] = []
    answers: list[str] = []
    if security_questions or security_answers:
        questions, answers = _validate_security_payload(
            security_questions, security_answers
        )
    security_values: list[str | None] = [None] * (SECURITY_QUESTION_COUNT * 2)
    if questions:
        security_values = [
            value
            for index in range(SECURITY_QUESTION_COUNT)
            for value in (
                questions[index],
                generate_password_hash(
                    _normalize_security_answers([answers[index]])[0]
                ),
            )
        ]

    username = username.strip()
    username_lower = username.lower()
    account = {
        "id": uuid.uuid4().hex,
        "username": username,
        "username_lower": username_lower,
        "password_hash": generate_password_hash(password),
        "role": role,
        "status": "active",
        "display_name": (display_name or username).strip() or username,
        "created_at": _now(),
        "created_by": created_by,
        "last_login_at": None,
        "security_questions": questions,
        "security_answers": answers,
    }
    with _LOCK:
        _ensure_db()
        with closing(_get_conn()) as conn:
            exists = conn.execute(
                "SELECT 1 FROM accounts WHERE username_lower = ?", (username_lower,)
            ).fetchone()
            if exists:
                raise ValueError("用户名已存在，请换一个")
            conn.execute(
                """INSERT INTO accounts (
                    id, username, username_lower, password_hash, role, status,
                    display_name, created_at, created_by, last_login_at,
                    security_question_1, security_answer_hash_1,
                    security_question_2, security_answer_hash_2,
                    security_question_3, security_answer_hash_3
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    account["id"],
                    account["username"],
                    account["username_lower"],
                    account["password_hash"],
                    account["role"],
                    account["status"],
                    account["display_name"],
                    account["created_at"],
                    account["created_by"],
                    account["last_login_at"],
                    *security_values,
                ),
            )
            conn.commit()
    _sync_credential(
        username=account["username"],
        password=password,
        role=account["role"],
        status=account["status"],
        display_name=account["display_name"],
        created_at=account["created_at"],
    )
    account.pop("username_lower", None)
    account.pop("password_hash", None)
    account.pop("security_answers", None)
    return account


def _seed_defaults(conn: sqlite3.Connection) -> None:
    defaults = [
        (
            DEFAULT_ADMIN_USERNAME,
            DEFAULT_ADMIN_PASSWORD,
            "admin",
            "系统管理员",
        ),
        (
            DEFAULT_USER_USERNAME,
            DEFAULT_USER_PASSWORD,
            "user",
            "示例用户",
        ),
    ]
    for username, password, role, display_name in defaults:
        exists = conn.execute(
            "SELECT 1 FROM accounts WHERE username_lower = ?", (username.lower(),)
        ).fetchone()
        if exists:
            continue
        if not password:
            raise RuntimeError(
                "首次创建账号库必须设置 DEFAULT_ADMIN_PASSWORD 和 "
                "DEFAULT_USER_PASSWORD 环境变量"
            )
        conn.execute(
            """INSERT INTO accounts (
                id, username, username_lower, password_hash, role, status,
                display_name, created_at, created_by, last_login_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                uuid.uuid4().hex,
                username,
                username.lower(),
                generate_password_hash(password),
                role,
                "active",
                display_name,
                _now(),
                "system",
                None,
            ),
        )
    conn.commit()


def get_account(account_id: str) -> dict | None:
    _ensure_db()
    with closing(_get_conn()) as conn:
        row = conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
    return _account_to_dict(row) if row else None


def list_accounts() -> list[dict]:
    _ensure_db()
    with closing(_get_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM accounts ORDER BY created_at ASC"
        ).fetchall()
    return [_account_to_dict(row) for row in rows]


def authenticate(username: str, password: str) -> dict | None:
    _ensure_db()
    username_lower = str(username or "").strip().lower()
    with closing(_get_conn()) as conn:
        row = conn.execute(
            "SELECT * FROM accounts WHERE username_lower = ?", (username_lower,)
        ).fetchone()
        if row is None or row["status"] != "active":
            return None
        if not check_password_hash(row["password_hash"], str(password or "")):
            return None
        conn.execute(
            "UPDATE accounts SET last_login_at = ? WHERE id = ?",
            (_now(), row["id"]),
        )
        conn.commit()
    return _account_to_dict(row)


def verify_account_password(account_id: str, password: str) -> bool:
    _ensure_db()
    with closing(_get_conn()) as conn:
        row = conn.execute(
            "SELECT password_hash FROM accounts WHERE id = ?", (account_id,)
        ).fetchone()
    if row is None:
        return False
    return check_password_hash(row["password_hash"], str(password or ""))


def get_security_questions(username: str) -> list[str] | None:
    _ensure_db()
    username_lower = str(username or "").strip().lower()
    with closing(_get_conn()) as conn:
        row = conn.execute(
            "SELECT * FROM accounts WHERE username_lower = ?", (username_lower,)
        ).fetchone()
    if row is None or not row["security_question_1"]:
        return None
    return [
        row["security_question_1"],
        row["security_question_2"],
        row["security_question_3"],
    ]


def _security_answers_match(row: sqlite3.Row, answers: list[str]) -> bool:
    normalized = _normalize_security_answers(answers)
    if len(normalized) != SECURITY_QUESTION_COUNT:
        return False
    hashes = [
        row["security_answer_hash_1"],
        row["security_answer_hash_2"],
        row["security_answer_hash_3"],
    ]
    if any(not item for item in hashes):
        return False
    return all(
        check_password_hash(answer_hash, answer)
        for answer_hash, answer in zip(hashes, normalized)
    )


def reset_password_with_security(
    username: str,
    answers: list[str],
    new_password: str,
) -> dict:
    if error := password_error(new_password):
        raise ValueError(error)
    _ensure_db()
    submitted = str(username or "").strip()
    with _LOCK:
        with closing(_get_conn()) as conn:
            row = conn.execute(
                "SELECT * FROM accounts WHERE username = ?", (submitted,)
            ).fetchone()
            if row is None:
                raise ValueError("用户名填写不完整或不正确")
            if not _security_answers_match(row, answers):
                raise ValueError("密保答案不正确")
            conn.execute(
                "UPDATE accounts SET password_hash = ? WHERE id = ?",
                (generate_password_hash(new_password), row["id"]),
            )
            conn.commit()
    revoke_account_sessions(row["id"])
    _sync_credential(
        username=row["username"],
        password=new_password,
        role=row["role"],
        status=row["status"],
        display_name=row["display_name"] or row["username"],
        created_at=row["created_at"] or "",
    )
    return get_account(row["id"]) or {}


def update_own_profile(
    account_id: str,
    *,
    current_password: str,
    username: str | None = None,
    display_name: str | None = None,
    new_password: str | None = None,
    security_questions: list[str] | None = None,
    security_answers: list[str] | None = None,
    keep_token: str | None = None,
) -> dict:
    if not verify_account_password(account_id, current_password):
        raise ValueError("当前密码不正确")
    _ensure_db()
    with closing(_get_conn()) as conn:
        old = conn.execute(
            "SELECT * FROM accounts WHERE id = ?", (account_id,)
        ).fetchone()
    if old is None:
        raise ValueError("账号不存在")

    updates: dict[str, str] = {}
    new_username = old["username"]
    if username is not None and username.strip() != old["username"]:
        if error := username_error(username):
            raise ValueError(error)
        new_username = username.strip()
        with closing(_get_conn()) as conn:
            duplicate = conn.execute(
                "SELECT 1 FROM accounts WHERE username_lower = ? AND id != ?",
                (new_username.lower(), account_id),
            ).fetchone()
        if duplicate:
            raise ValueError("用户名已存在，请换一个")
        updates["username"] = new_username
        updates["username_lower"] = new_username.lower()
    if display_name is not None:
        updates["display_name"] = display_name.strip() or new_username
    if new_password is not None:
        if error := password_error(new_password):
            raise ValueError(error)
        updates["password_hash"] = generate_password_hash(new_password)
    if security_questions or security_answers:
        questions, answers = _validate_security_payload(
            security_questions, security_answers
        )
        for index in range(SECURITY_QUESTION_COUNT):
            updates[f"security_question_{index + 1}"] = questions[index]
            updates[f"security_answer_hash_{index + 1}"] = generate_password_hash(
                _normalize_security_answers([answers[index]])[0]
            )
    if not updates:
        return get_account(account_id) or {}

    assignments = ", ".join(f"{key} = ?" for key in updates)
    with _LOCK:
        with closing(_get_conn()) as conn:
            conn.execute(
                f"UPDATE accounts SET {assignments} WHERE id = ?",
                list(updates.values()) + [account_id],
            )
            conn.commit()
    if new_password is not None or "username" in updates:
        revoke_account_sessions(account_id, keep_token=keep_token)
    _sync_credential(
        username=new_username,
        password=new_password,
        role=old["role"],
        status=old["status"],
        display_name=updates.get("display_name") or old["display_name"] or new_username,
        created_at=old["created_at"] or "",
        previous_username=old["username"],
    )
    return get_account(account_id) or {}


def revoke_account_sessions(account_id: str, keep_token: str | None = None) -> None:
    keep_hash = _hash_token(keep_token) if keep_token else None
    with _LOCK:
        with closing(_get_conn()) as conn:
            if keep_hash:
                conn.execute(
                    """UPDATE auth_sessions SET revoked_at = ?
                       WHERE account_id = ? AND revoked_at IS NULL AND token_hash != ?""",
                    (_now(), account_id, keep_hash),
                )
            else:
                conn.execute(
                    """UPDATE auth_sessions SET revoked_at = ?
                       WHERE account_id = ? AND revoked_at IS NULL""",
                    (_now(), account_id),
                )
            conn.commit()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(account_id: str, user_agent: str | None = None) -> str:
    _ensure_db()
    token = secrets.token_urlsafe(32)
    created = datetime.now(timezone.utc)
    expires = created + timedelta(hours=SESSION_TTL_HOURS)
    with _LOCK:
        with closing(_get_conn()) as conn:
            conn.execute(
                """INSERT INTO auth_sessions (
                    token_hash, account_id, created_at, expires_at, revoked_at, user_agent
                ) VALUES (?,?,?,?,?,?)""",
                (
                    _hash_token(token),
                    account_id,
                    created.isoformat(),
                    expires.isoformat(),
                    None,
                    user_agent,
                ),
            )
            conn.commit()
    return token


def resolve_session(token: str) -> dict | None:
    if not token:
        return None
    _ensure_db()
    token_hash = _hash_token(token)
    with closing(_get_conn()) as conn:
        row = conn.execute(
            """SELECT s.expires_at, s.revoked_at, a.*
               FROM auth_sessions s
               JOIN accounts a ON a.id = s.account_id
               WHERE s.token_hash = ?""",
            (token_hash,),
        ).fetchone()
    if row is None or row["revoked_at"]:
        return None
    expires = datetime.fromisoformat(row["expires_at"])
    if expires <= datetime.now(timezone.utc):
        revoke_session(token)
        return None
    if row["status"] != "active":
        return None
    return _account_to_dict(row)


def revoke_session(token: str) -> bool:
    if not token:
        return False
    with _LOCK:
        with closing(_get_conn()) as conn:
            cur = conn.execute(
                "UPDATE auth_sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
                (_now(), _hash_token(token)),
            )
            conn.commit()
    return cur.rowcount > 0


def new_challenge() -> dict:
    now = time.time()
    with _LOCK:
        expired = [
            challenge_id
            for challenge_id, item in _CHALLENGES.items()
            if item["expires_at"] <= now
        ]
        for challenge_id in expired:
            _CHALLENGES.pop(challenge_id, None)

    operator = _RANDOM.choice(["+", "-"])
    if operator == "+":
        left = _RANDOM.randint(1, 98)
        right = _RANDOM.randint(1, 100 - left)
        answer = left + right
    else:
        left = _RANDOM.randint(2, 100)
        right = _RANDOM.randint(1, left - 1)
        answer = left - right

    challenge_id = secrets.token_urlsafe(16)
    with _LOCK:
        _CHALLENGES[challenge_id] = {
            "answer": answer,
            "expires_at": now + CHALLENGE_TTL_SECONDS,
        }
    return {
        "challenge_id": challenge_id,
        "question": f"{left} {operator} {right} = ?",
        "expires_in": CHALLENGE_TTL_SECONDS,
    }


def verify_challenge(challenge_id: str, answer) -> bool:
    with _LOCK:
        item = _CHALLENGES.pop(str(challenge_id or ""), None)
    if item is None or item["expires_at"] <= time.time():
        return False
    try:
        submitted = int(str(answer).strip())
    except (TypeError, ValueError):
        return False
    return secrets.compare_digest(str(submitted), str(item["answer"]))


def update_account(
    account_id: str,
    *,
    role: str | None = None,
    status: str | None = None,
    display_name: str | None = None,
    password: str | None = None,
) -> dict | None:
    _ensure_db()
    with closing(_get_conn()) as conn:
        old = conn.execute(
            "SELECT * FROM accounts WHERE id = ?", (account_id,)
        ).fetchone()
    if old is None:
        return None
    updates: dict[str, str] = {}
    if role is not None:
        if role not in {"admin", "user"}:
            raise ValueError("账号角色必须为 admin 或 user")
        updates["role"] = role
    if status is not None:
        if status not in {"active", "disabled"}:
            raise ValueError("账号状态必须为 active 或 disabled")
        updates["status"] = status
    if display_name is not None:
        updates["display_name"] = str(display_name).strip()
    if password is not None:
        if error := password_error(password):
            raise ValueError(error)
        updates["password_hash"] = generate_password_hash(password)
    if not updates:
        return get_account(account_id)
    assignments = ", ".join(f"{key} = ?" for key in updates)
    with _LOCK:
        with closing(_get_conn()) as conn:
            cur = conn.execute(
                f"UPDATE accounts SET {assignments} WHERE id = ?",
                list(updates.values()) + [account_id],
            )
            conn.commit()
    if cur.rowcount == 0:
        return None
    updated = get_account(account_id)
    if updated:
        _sync_credential(
            username=updated["username"],
            password=password,
            role=updated["role"],
            status=updated["status"],
            display_name=updated["display_name"],
            created_at=updated["created_at"] or old["created_at"] or "",
        )
    return updated
