"""SQLite-backed user profiles and assessment history.

Existing JSON/JSONL records under ``data/user_records/`` are migrated once on
first use and kept untouched as a backup.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
USER_DIR = DATA_DIR / "user_records"
USERS_FILE = USER_DIR / "users.json"
USER_RECORDS_DB = DATA_DIR / "user_records.sqlite"
_LOCK = threading.Lock()
BACKUP_FORMAT = "elderly-depression-user-backup"
BACKUP_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _get_conn() -> sqlite3.Connection:
    USER_RECORDS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(USER_RECORDS_DB))
    conn.row_factory = sqlite3.Row
    try:
        os.chmod(USER_RECORDS_DB, 0o600)
    except OSError:
        pass
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            gender TEXT,
            marital_status TEXT,
            comorbidity_count INTEGER,
            self_rated_health INTEGER,
            followup_label INTEGER,
            note TEXT,
            owner_account_id TEXT,
            created_at TEXT,
            updated_at TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS assessments (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            created_at TEXT,
            record TEXT
        )"""
    )
    conn.commit()


def _ensure_db() -> None:
    try:
        if USER_DIR.exists():
            for path in USER_DIR.iterdir():
                if path.is_file():
                    os.chmod(path, 0o600)
    except OSError:
        pass
    if USER_RECORDS_DB.exists():
        with closing(_get_conn()) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(assessments)")]
            if "record" not in cols:
                conn.execute("DROP TABLE IF EXISTS assessments")
                conn.execute("DROP TABLE IF EXISTS users")
                _init_db(conn)
                if USERS_FILE.exists():
                    _migrate_json(conn)
            user_cols = [r[1] for r in conn.execute("PRAGMA table_info(users)")]
            if "owner_account_id" not in user_cols:
                conn.execute("ALTER TABLE users ADD COLUMN owner_account_id TEXT")
            _init_db(conn)
        return
    with closing(_get_conn()) as conn:
        _init_db(conn)
        if USERS_FILE.exists():
            _migrate_json(conn)
        conn.commit()


def _migrate_json(conn: sqlite3.Connection) -> None:
    for user in _read_json(USERS_FILE, []):
        conn.execute(
            """INSERT OR IGNORE INTO users (
                id, name, age, gender, marital_status, comorbidity_count,
                self_rated_health, followup_label, note, owner_account_id,
                created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                user.get("id"),
                user.get("name"),
                user.get("age"),
                user.get("gender"),
                user.get("marital_status"),
                user.get("comorbidity_count"),
                user.get("self_rated_health"),
                user.get("followup_label"),
                user.get("note"),
                user.get("owner_account_id"),
                user.get("created_at"),
                user.get("updated_at"),
            ),
        )
        for record in _read_jsonl(USER_DIR / f"{user.get('id')}.jsonl"):
            conn.execute(
                "INSERT OR IGNORE INTO assessments VALUES (?,?,?,?)",
                (
                    record.get("id"),
                    record.get("user_id"),
                    record.get("created_at"),
                    json.dumps(record, ensure_ascii=False),
                ),
            )


def _user_to_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in row.keys()}


def list_users(
    query: str | None = None,
    owner_account_id: str | None = None,
) -> list[dict]:
    with _LOCK:
        _ensure_db()
        with closing(_get_conn()) as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM users")]
    if owner_account_id is not None:
        rows = [u for u in rows if u.get("owner_account_id") == owner_account_id]
    if query:
        q = str(query).strip().lower()
        rows = [
            u
            for u in rows
            if q in str(u.get("name", "")).lower()
            or q in str(u.get("id", "")).lower()
            or q in str(u.get("note", "")).lower()
        ]
    return sorted(rows, key=lambda u: u.get("created_at", ""), reverse=True)


def get_user(user_id: str) -> dict | None:
    with _LOCK:
        _ensure_db()
        with closing(_get_conn()) as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def create_user(
    name: str,
    owner_account_id: str | None = None,
    **profile,
) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("用户姓名不能为空")
    now = _now()
    user = {
        "id": uuid.uuid4().hex,
        "name": name,
        "owner_account_id": owner_account_id,
        "created_at": now,
        "updated_at": now,
    }
    user.update({k: v for k, v in profile.items() if v is not None})
    with _LOCK:
        _ensure_db()
        with closing(_get_conn()) as conn:
            conn.execute(
                """INSERT INTO users (
                    id, name, age, gender, marital_status, comorbidity_count,
                    self_rated_health, followup_label, note, owner_account_id,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    user.get("id"),
                    user.get("name"),
                    user.get("age"),
                    user.get("gender"),
                    user.get("marital_status"),
                    user.get("comorbidity_count"),
                    user.get("self_rated_health"),
                    user.get("followup_label"),
                    user.get("note"),
                    user.get("owner_account_id"),
                    user.get("created_at"),
                    user.get("updated_at"),
                ),
            )
            conn.commit()
    return user


def update_user(user_id: str, **fields) -> dict | None:
    user = get_user(user_id)
    if user is None:
        return None
    allowed = {
        "name",
        "age",
        "gender",
        "marital_status",
        "comorbidity_count",
        "self_rated_health",
        "followup_label",
        "note",
    }
    updates = {}
    for key, value in fields.items():
        if key not in allowed:
            continue
        if value is None and key not in {"note", "followup_label"}:
            continue
        updates[key] = value
    if updates:
        updates["updated_at"] = _now()
        assignments = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [user_id]
        with _LOCK:
            _ensure_db()
            with closing(_get_conn()) as conn:
                conn.execute(f"UPDATE users SET {assignments} WHERE id = ?", values)
                conn.commit()
    return get_user(user_id)


def delete_user(user_id: str) -> bool:
    with _LOCK:
        _ensure_db()
        with closing(_get_conn()) as conn:
            cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.execute("DELETE FROM assessments WHERE user_id = ?", (user_id,))
            conn.commit()
    return cur.rowcount > 0


def save_assessment(user_id: str, record: dict) -> dict:
    if get_user(user_id) is None:
        raise ValueError(f"用户不存在: {user_id}")
    entry = {
        **record,
        "id": uuid.uuid4().hex,
        "user_id": user_id,
        "created_at": _now(),
    }
    with _LOCK:
        _ensure_db()
        with closing(_get_conn()) as conn:
            conn.execute(
                "INSERT INTO assessments VALUES (?,?,?,?)",
                (
                    entry["id"],
                    user_id,
                    entry["created_at"],
                    json.dumps(entry, ensure_ascii=False),
                ),
            )
            conn.commit()
    return entry


def list_assessments(
    user_id: str,
    limit: int | None = 200,
    risk: str | None = None,
    scale: str | None = None,
) -> list[dict]:
    with _LOCK:
        _ensure_db()
        with closing(_get_conn()) as conn:
            rows = [
                json.loads(r["record"])
                for r in conn.execute(
                    "SELECT record FROM assessments WHERE user_id = ?", (user_id,)
                )
            ]
    if risk:
        rows = [r for r in rows if r.get("risk_level") == risk]
    if scale:
        rows = [r for r in rows if r.get("scale_name") == scale]
    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return rows if limit is None else rows[:limit]


def get_assessment(user_id: str, record_id: str) -> dict | None:
    for record in list_assessments(user_id, limit=None):
        if record.get("id") == record_id:
            return record
    return None


def export_all(owner_account_id: str | None = None) -> dict:
    """Return a portable JSON snapshot of all users and their assessments."""
    users = list_users(owner_account_id=owner_account_id)
    return {
        "format": BACKUP_FORMAT,
        "version": BACKUP_VERSION,
        "exported_at": _now(),
        "users": users,
        "assessments": {
            user["id"]: list_assessments(user["id"], limit=None) for user in users
        },
    }


def import_backup(payload: dict, mode: str = "merge") -> dict:
    """Import a backup snapshot, either merging or replacing existing data."""
    if not isinstance(payload, dict) or payload.get("format") != BACKUP_FORMAT:
        raise ValueError("不是有效的用户数据备份文件")
    if mode not in {"merge", "replace"}:
        raise ValueError("导入模式必须为 merge 或 replace")

    users = payload.get("users") or []
    assessments = payload.get("assessments") or {}
    imported_users = 0
    imported_assessments = 0
    merge_allowed = {
        "name",
        "age",
        "gender",
        "marital_status",
        "comorbidity_count",
        "self_rated_health",
        "followup_label",
        "note",
    }

    with _LOCK:
        _ensure_db()
        with closing(_get_conn()) as conn:
            if mode == "replace":
                conn.execute("DELETE FROM assessments")
                conn.execute("DELETE FROM users")

            for user in users or []:
                uid = user.get("id")
                if not uid:
                    continue
                existing = conn.execute(
                    "SELECT id FROM users WHERE id = ?", (uid,)
                ).fetchone()
                if existing and mode == "merge":
                    updates = {k: v for k, v in user.items() if k in merge_allowed and v is not None}
                    if updates:
                        updates["updated_at"] = _now()
                        assignments = ", ".join(f"{k} = ?" for k in updates)
                        conn.execute(
                            f"UPDATE users SET {assignments} WHERE id = ?",
                            list(updates.values()) + [uid],
                        )
                    imported_users += 1
                    continue
                if existing:
                    continue
                row = (
                    uid,
                    user.get("name"),
                    user.get("age"),
                    user.get("gender"),
                    user.get("marital_status"),
                    user.get("comorbidity_count"),
                    user.get("self_rated_health"),
                    user.get("followup_label"),
                    user.get("note"),
                    user.get("owner_account_id"),
                    user.get("created_at") or _now(),
                    user.get("updated_at") or _now(),
                )
                conn.execute(
                    """INSERT INTO users (
                        id, name, age, gender, marital_status, comorbidity_count,
                        self_rated_health, followup_label, note, owner_account_id,
                        created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    row,
                )
                imported_users += 1

            for uid, records in (assessments or {}).items():
                for record in records or []:
                    record_id = record.get("id")
                    if not record_id:
                        continue
                    if mode == "merge":
                        exists = conn.execute(
                            "SELECT id FROM assessments WHERE id = ?", (record_id,)
                        ).fetchone()
                        if exists:
                            continue
                    entry = {
                        "id": record_id,
                        "user_id": record.get("user_id") or uid,
                        "created_at": record.get("created_at") or _now(),
                        **record,
                    }
                    conn.execute(
                        "INSERT OR REPLACE INTO assessments VALUES (?,?,?,?)",
                        (
                            entry["id"],
                            entry["user_id"],
                            entry["created_at"],
                            json.dumps(entry, ensure_ascii=False),
                        ),
                    )
                    imported_assessments += 1
            conn.commit()

    return {
        "mode": mode,
        "users": imported_users,
        "assessments": imported_assessments,
    }


def stats() -> dict:
    """Return lightweight storage stats for the system status panel."""
    db_size = USER_RECORDS_DB.stat().st_size if USER_RECORDS_DB.exists() else 0
    with _LOCK:
        _ensure_db()
        with closing(_get_conn()) as conn:
            users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            assessments = conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
    return {"users": users, "assessments": assessments, "db_size_bytes": db_size}
