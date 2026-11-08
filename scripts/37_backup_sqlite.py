"""Back up user profiles and assessments into SQLite."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.records import list_assessments, list_users


def main() -> None:
    out = PROJECT_ROOT / "data" / "user_records.sqlite"
    conn = sqlite3.connect(out)
    conn.execute("DROP TABLE IF EXISTS users")
    conn.execute("DROP TABLE IF EXISTS assessments")
    conn.execute(
        """CREATE TABLE users (
            id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            gender TEXT,
            marital_status TEXT,
            comorbidity_count INTEGER,
            self_rated_health INTEGER,
            followup_label INTEGER,
            note TEXT,
            created_at TEXT,
            updated_at TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE assessments (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            created_at TEXT,
            depression_probability REAL,
            risk_level TEXT,
            scale_score REAL,
            text_risk REAL,
            scale_risk REAL,
            context_risk REAL,
            expression_risk REAL,
            mpdd_risk REAL
        )"""
    )
    for user in list_users():
        conn.execute(
            "INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?)",
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
                user.get("created_at"),
                user.get("updated_at"),
            ),
        )
        for record in list_assessments(user["id"]):
            conn.execute(
                "INSERT INTO assessments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.get("id"),
                    record.get("user_id"),
                    record.get("created_at"),
                    record.get("depression_probability"),
                    record.get("risk_level"),
                    record.get("scale_score"),
                    record.get("text_risk"),
                    record.get("scale_risk"),
                    record.get("context_risk"),
                    record.get("expression_risk"),
                    record.get("mpdd_risk"),
                ),
            )
    conn.commit()
    users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    assessments = conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
    conn.close()
    print(f"Saved: {out} ({users} users, {assessments} assessments)")


if __name__ == "__main__":
    main()
