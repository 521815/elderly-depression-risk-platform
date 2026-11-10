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


class JobCancelled(RuntimeError):
    """Raised inside a cooperative task after cancellation is requested."""


def _persist(job: dict) -> None:
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(job, ensure_ascii=False, default=str) + "\n")


def submit(
    func: Callable[..., Any],
    *args,
    name: str | None = None,
    timeout: float = 3600.0,
    dedupe_key: str | None = None,
    **kwargs,
) -> tuple[str, bool]:
    with _LOCK:
        if dedupe_key:
            for existing in _JOBS.values():
                if (
                    existing.get("status") == "running"
                    and existing.get("task") == name
                    and existing.get("dedupe_key") == dedupe_key
                ):
                    return str(existing["id"]), False
        job_id = uuid.uuid4().hex
        _JOBS[job_id] = {
            "id": job_id,
            "task": name,
            "status": "running",
            "progress": 0.0,
            "stage": "submitted",
            "created_at": time.time(),
            "timeout": float(timeout),
            "dedupe_key": dedupe_key,
            "cancel_requested": False,
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
        except JobCancelled:
            result, status, error = None, "cancelled", None
            progress = 1.0
            stage = "cancelled"
        except Exception as exc:  # pragma: no cover - task failures
            result, status, error = None, "error", str(exc)
            progress = 1.0
            stage = "error"
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                return
            if job.get("status") in {"cancelled", "timeout"}:
                if job.get("status") == "cancelled":
                    job["cancel_requested"] = True
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
    return job_id, True


def update_progress(job_id: str, progress: float, stage: str | None = None) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        if job.get("status") in {"cancelled", "timeout"}:
            raise JobCancelled(f"job {job_id} is {job['status']}")
        job["progress"] = float(progress)
        if stage:
            job["stage"] = stage


def check_cancelled(job_id: str) -> None:
    """Cooperative cancellation check for long-running task code."""
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        if job.get("status") in {"cancelled", "timeout"}:
            raise JobCancelled(f"job {job_id} is {job['status']}")


def cancel(job_id: str) -> bool:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None or job["status"] not in ("running",):
            return False
        job.update(
            status="cancelled",
            cancel_requested=True,
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


def _public_job(job: dict) -> dict:
    item = dict(job)
    item.pop("dedupe_key", None)
    end_time = item.get("finished_at") or time.time()
    item["elapsed_seconds"] = max(
        0.0,
        float(end_time) - float(item.get("created_at", end_time)),
    )
    item["is_active"] = item.get("status") == "running"
    return item


def get(job_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is not None and job["status"] == "running":
            timeout = float(job.get("timeout", 3600.0))
            if time.time() - job["created_at"] > timeout:
                job.update(
                    status="timeout",
                    cancel_requested=True,
                    progress=1.0,
                    stage="timeout",
                    finished_at=time.time(),
                )
                _persist(dict(job))
        job = _JOBS.get(job_id)
        return _public_job(job) if job is not None else None


def list_jobs(limit: int = 20) -> list[dict]:
    history = _load_history(limit * 2)
    with _LOCK:
        live = list(_JOBS.values())
    merged = {j["id"]: j for j in history}
    for job in live:
        merged[job["id"]] = job
    items = sorted(merged.values(), key=lambda j: j.get("created_at", 0), reverse=True)
    return [_public_job(job) for job in items[:limit]]
