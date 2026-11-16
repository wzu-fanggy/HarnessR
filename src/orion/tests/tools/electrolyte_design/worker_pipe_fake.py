"""A scripted stand-in for a resident worker's stdin/stdout pipe.

The fake exists to make the *concurrency* of one worker exchange observable: it
records how many callers were inside ``readline`` at the same time.  A real worker
pipe shows the same hazard, but it shows it as an occasional 900 s hang rather
than as a number a test can assert on.
"""

from __future__ import annotations

import json
import threading
import time


class FakeWorkerStdio:
    """Minimal text pipe whose reads overlap in time unless the caller serialises."""

    pid = 4242

    def __init__(self, *, read_delay_s: float = 0.1) -> None:
        self.stdin = _FakeStdin(self)
        self.stdout = _FakeStdout(self)
        self.stderr = _FakeStderr()
        self.requests: list[dict] = []
        self.read_delay_s = float(read_delay_s)
        self.max_concurrent_reads = 0
        self._responses: list[str] = []
        self._responses_lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._active_reads = 0

    # -- subprocess.Popen surface used by the adapters -----------------------

    def poll(self):
        return None

    def wait(self, timeout=None) -> int:
        return 0

    def kill(self) -> None:
        return None

    def terminate(self) -> None:
        return None

    # -- pipe behaviour ------------------------------------------------------

    def ingest(self, line: str) -> None:
        payload = json.loads(line)
        with self._responses_lock:
            self.requests.append(payload)
            self._responses.append(
                json.dumps({"status": "RESULT", "result": {"echo": payload}, "echo": payload}) + "\n"
            )

    def take(self) -> str:
        with self._probe_lock:
            self._active_reads += 1
            self.max_concurrent_reads = max(self.max_concurrent_reads, self._active_reads)
        try:
            time.sleep(self.read_delay_s)
            with self._responses_lock:
                if not self._responses:
                    return ""
                return self._responses.pop(0)
        finally:
            with self._probe_lock:
                self._active_reads -= 1


class _FakeStdin:
    def __init__(self, owner: FakeWorkerStdio) -> None:
        self._owner = owner

    def write(self, text: str) -> int:
        self._owner.ingest(text.strip())
        return len(text)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeStdout:
    def __init__(self, owner: FakeWorkerStdio) -> None:
        self._owner = owner

    def readline(self) -> str:
        return self._owner.take()

    def close(self) -> None:
        return None


class _FakeStderr:
    def read(self) -> str:
        return ""

    def close(self) -> None:
        return None
