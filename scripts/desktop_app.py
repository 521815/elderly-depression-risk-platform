"""Desktop app: embedded WebView with auto-started backend and frontend."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WEBAPP_PORT = 8501
BACKEND_PORT = 8001
LOCK_FILE = PROJECT / "output" / "desktop_app.lock"
APP_BUNDLE_ID = "com.elderly.depression.desktop"
FRONTEND_SUFFIXES = {
    ".css",
    ".html",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".png",
    ".py",
    ".svg",
    ".webp",
}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = None
        if pid and _pid_alive(pid):
            return False
        LOCK_FILE.unlink(missing_ok=True)
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _wait_health(port: int, path: str, timeout: float = 60.0) -> bool:
    url = f"http://127.0.0.1:{port}{path}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def _stop_port_listener(port: int) -> None:
    """Stop a stale project service so the desktop app always runs current code."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    pids = []
    for token in result.stdout.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + 4
    while pids and time.time() < deadline:
        pids = [pid for pid in pids if _pid_alive(pid)]
        if not pids:
            return
        time.sleep(0.2)


def _frontend_revision() -> int:
    """Return a monotonic revision for files that affect the visible app."""
    webapp_dir = PROJECT / "webapp"
    revisions = []
    for path in webapp_dir.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() not in FRONTEND_SUFFIXES:
            continue
        try:
            revisions.append(path.stat().st_mtime_ns)
        except OSError:
            continue
    return max(revisions, default=0)


def _watch_frontend(window: object, interval: float = 1.0) -> None:
    """Reload the embedded window when local frontend files change."""
    revision = _frontend_revision()
    while True:
        time.sleep(interval)
        next_revision = _frontend_revision()
        if next_revision <= revision:
            continue
        revision = next_revision
        # Give Streamlit a moment to finish its own source reload.
        time.sleep(0.6)
        try:
            window.evaluate_js("window.location.reload()")
        except Exception:
            pass


def main() -> None:
    if not _acquire_lock():
        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'tell application id "{APP_BUNDLE_ID}" to activate',
                ],
                timeout=5,
                check=False,
            )
        except Exception:
            pass
        print("桌面 App 已在运行，已尝试激活现有窗口。")
        return 0

    import webview

    try:
        import AppKit

        icon_path = PROJECT / "output" / "AppIcon.icns"
        if icon_path.exists():
            AppKit.NSApplication.sharedApplication().setApplicationIconImage_(
                AppKit.NSImage.alloc().initWithContentsOfFile_(str(icon_path))
            )
    except Exception:
        pass

    (PROJECT / "output").mkdir(parents=True, exist_ok=True)
    backend_log = (PROJECT / "output" / "desktop_backend.log").open("a", encoding="utf-8")
    webapp_log = (PROJECT / "output" / "desktop_webapp.log").open("a", encoding="utf-8")
    started = []

    # A healthy old process can still contain stale dialogue rules. Recreate the
    # two dedicated local services on every desktop launch.
    _stop_port_listener(BACKEND_PORT)
    _stop_port_listener(WEBAPP_PORT)
    started.append(
        subprocess.Popen(
            ["/bin/bash", str(PROJECT / "scripts" / "start_backend.sh")],
            cwd=str(PROJECT),
            stdout=backend_log,
            stderr=backend_log,
        )
    )

    started.append(
        subprocess.Popen(
            [
                "/opt/anaconda3/bin/streamlit",
                "run",
                str(PROJECT / "webapp" / "app.py"),
                "--server.headless",
                "true",
                "--server.port",
                str(WEBAPP_PORT),
                "--server.fileWatcherType",
                "auto",
            ],
            cwd=str(PROJECT),
            stdout=webapp_log,
            stderr=webapp_log,
        )
    )

    ok = _wait_health(BACKEND_PORT, "/health") and _wait_health(WEBAPP_PORT, "/healthz")
    if not ok:
        print("服务启动失败，请查看 output/desktop_backend.log 和 output/desktop_webapp.log")
        for proc in started:
            proc.terminate()
        return

    window = webview.create_window(
        "老年抑郁风险筛查",
        f"http://127.0.0.1:{WEBAPP_PORT}",
        width=1280,
        height=860,
        min_size=(960, 640),
    )
    try:
        webview.start(_watch_frontend, window)
    finally:
        LOCK_FILE.unlink(missing_ok=True)
        for proc in started:
            proc.terminate()
        backend_log.close()
        webapp_log.close()


if __name__ == "__main__":
    sys.exit(main())
