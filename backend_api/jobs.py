"""In-memory background job runner with progress, cancel, timeout and history."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_HISTORY_FILE = Path(__file__).resolve().parents[1] / "output" / "jobs_history.jsonl"


def _persist(job: dict) -> None:
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(job, ensure_ascii=False, default=str) + "\n")


def submit(
    func: Callable[..., Any],
    *args,
    name: str | None = None,
    timeout: float = 3600.0,
    **kwargs,
) -> str:
    job_id = uuid.uuid4().hex
    with _LOCK:
        _JOBS[job_id] = {
            "id": job_id,
            "task": name,
            "status": "running",
            "progress": 0.0,
            "stage": "submitted",
            "created_at": time.time(),
            "timeout": float(timeout),
            "result": None,
            "error": None,
            "finished_at": None,
        }

    def run() -> None:
        try:
            with _LOCK:
                if _JOBS.get(job_id, {}).get("status") == "cancelled":
                    return
                _JOBS[job_id]["stage"] = "running"
            result = func(*args, **kwargs)
            status, error = "done", None
            progress = 1.0
            stage = "done"
        except Exception as exc:  # pragma: no cover - task failures
            result, status, error = None, "error", str(exc)
            progress = 1.0
            stage = "error"
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                return
            job.update(
                status=status,
                progress=progress,
                stage=stage,
                result=result,
                error=error,
                finished_at=time.time(),
            )
            _persist(dict(job))

    threading.Thread(target=run, daemon=True).start()
    return job_id


def update_progress(job_id: str, progress: float, stage: str | None = None) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["progress"] = float(progress)
        if stage:
            job["stage"] = stage


def cancel(job_id: str) -> bool:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None or job["status"] not in ("running",):
            return False
        job.update(
            status="cancelled",
            progress=1.0,
            stage="cancelled",
            finished_at=time.time(),
        )
        _persist(dict(job))
    return True


def _load_history(limit: int) -> list[dict]:
    if not _HISTORY_FILE.exists():
        return []
    rows = []
    for line in _HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return rows[:limit]


def get(job_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is not None and job["status"] == "running":
            timeout = float(job.get("timeout", 3600.0))
            if time.time() - job["created_at"] > timeout:
                job.update(
                    status="timeout",
                    progress=1.0,
                    stage="timeout",
                    finished_at=time.time(),
                )
                _persist(dict(job))
        return _JOBS.get(job_id)


def list_jobs(limit: int = 20) -> list[dict]:
    history = _load_history(limit * 2)
    with _LOCK:
        live = list(_JOBS.values())
    merged = {j["id"]: j for j in history}
    for job in live:
        merged[job["id"]] = job
    items = sorted(merged.values(), key=lambda j: j.get("created_at", 0), reverse=True)
    return items[:limit]
