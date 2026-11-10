import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_api import jobs


class JobRunnerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._history = Path(self._tmp.name) / "jobs.jsonl"
        self._patcher = mock.patch.object(jobs, "_HISTORY_FILE", self._history)
        self._patcher.start()
        with jobs._LOCK:
            jobs._JOBS.clear()

    def tearDown(self):
        with jobs._LOCK:
            jobs._JOBS.clear()
        self._patcher.stop()
        self._tmp.cleanup()

    def _wait_for(self, job_id: str, statuses: set[str]) -> dict:
        for _ in range(100):
            job = jobs.get(job_id)
            if job and job["status"] in statuses:
                return job
            time.sleep(0.01)
        self.fail(f"job {job_id} did not reach {statuses}")

    def test_duplicate_submission_reuses_running_job(self):
        gate = threading.Event()
        job_id, created = jobs.submit(
            lambda payload: gate.wait(1) and payload,
            {"x": 1},
            name="test",
            dedupe_key="same",
        )
        duplicate_id, duplicate_created = jobs.submit(
            lambda payload: payload,
            {"x": 1},
            name="test",
            dedupe_key="same",
        )
        gate.set()
        self.assertTrue(created)
        self.assertFalse(duplicate_created)
        self.assertEqual(job_id, duplicate_id)

    def test_cancelled_job_result_cannot_overwrite_status(self):
        gate = threading.Event()
        job_id, _ = jobs.submit(
            lambda payload: gate.wait(1) and payload,
            {"x": 1},
            name="cancel",
        )
        self.assertTrue(jobs.cancel(job_id))
        gate.set()
        time.sleep(0.05)
        job = jobs.get(job_id)
        self.assertEqual(job["status"], "cancelled")
        self.assertTrue(job["cancel_requested"])
        self.assertIsNone(job["result"])

    def test_cooperative_check_raises_after_cancel(self):
        started = threading.Event()
        gate = threading.Event()
        job_ref = {}

        def cooperative(payload):
            started.set()
            gate.wait(1)
            jobs.check_cancelled(job_ref["id"])
            return payload

        job_id, _ = jobs.submit(cooperative, {"ok": True}, name="cooperative")
        job_ref["id"] = job_id
        self.assertTrue(started.wait(1))
        self.assertTrue(jobs.cancel(job_id))
        gate.set()
        time.sleep(0.05)
        self.assertEqual(jobs.get(job_id)["status"], "cancelled")

    def test_timed_out_job_result_cannot_overwrite_status(self):
        gate = threading.Event()
        job_id, _ = jobs.submit(
            lambda payload: gate.wait(1) and payload,
            {"x": 1},
            name="timeout",
            timeout=0.01,
        )
        time.sleep(0.03)
        self.assertEqual(jobs.get(job_id)["status"], "timeout")
        gate.set()
        time.sleep(0.05)
        job = jobs.get(job_id)
        self.assertEqual(job["status"], "timeout")
        self.assertIsNone(job["result"])


if __name__ == "__main__":
    unittest.main()
