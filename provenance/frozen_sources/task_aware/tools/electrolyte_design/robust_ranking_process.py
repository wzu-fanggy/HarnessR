"""CWSR adapter for the resident robust-ranking worker.

The worker holds a fitted group-bootstrap random-forest ensemble, so its
establishment cost is dominated by ensemble fitting (seconds, CPU only) and its
resident cost is the fitted members' memory.  Both are measured, never declared
for effect: see ``benchmark/cwsr_robust_ranking_resource_probe_v1.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import Any, Mapping

from capvm import ResidencyLevel, ResourceVector
from capvm.adapters.base import BaseCapabilityAdapter
from capvm.telemetry import process_rss_mb
from .process_exchange import exchange_lock, run_exchange
from .robust_ranking import RobustRankingError


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKER = Path(__file__).with_name("robust_ranking_worker.py")
# Frozen from the five-cycle lifecycle probe (2026-09-24): median RSS 773.8 MiB,
# max 774.5 MiB, median cold start 12.70 s, median shutdown 0.734 s.  The declared
# value rounds the measured maximum up instead of down so the resource is never
# under-declared; it is not raised to manufacture contention.
MEASURED_HOST_RAM_MB = 774.5
DEFAULT_DECLARED_HOST_RAM_MB = 800.0


class RobustRankingProcessAdapter(BaseCapabilityAdapter):
    """Own a reusable scikit-learn ranking worker and expose its measured lifecycle."""

    def __init__(
        self,
        *,
        worker_script: Path = DEFAULT_WORKER,
        bundle_dir: Path | None = None,
        declared_host_ram_mb: float = DEFAULT_DECLARED_HOST_RAM_MB,
        shutdown_timeout_s: float = 15.0,
    ) -> None:
        self.worker_script = worker_script.resolve()
        self.bundle_dir = Path(bundle_dir).resolve() if bundle_dir is not None else None
        self.declared_host_ram_mb = float(declared_host_ram_mb)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self._process: subprocess.Popen[str] | None = None
        self.boot_metadata: dict[str, Any] | None = None
        self.ready_metadata: dict[str, Any] | None = None
        self.last_startup_latency_s = 0.0
        self.last_shutdown_latency_s = 0.0
        self._exchange_lock = exchange_lock()

    @property
    def launcher_pid(self) -> int | None:
        process = self._process
        if process is None or process.poll() is not None:
            return None
        return int(process.pid)

    @property
    def pid(self) -> int | None:
        launcher_pid = self.launcher_pid
        if launcher_pid is None:
            return None
        try:
            return int((self.boot_metadata or {}).get("pid", launcher_pid))
        except (TypeError, ValueError):
            return launcher_pid

    @property
    def is_running(self) -> bool:
        return self.launcher_pid is not None

    @property
    def training_metadata(self) -> dict[str, Any]:
        return dict(self.ready_metadata or {})

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND:
            return ResourceVector()
        return ResourceVector(host_ram_mb=self.declared_host_ram_mb, process_slots=1.0)

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND or self.pid is None:
            return ResourceVector()
        return ResourceVector(host_ram_mb=process_rss_mb(self.pid), process_slots=1.0)

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current < ResidencyLevel.BOUND <= target and not self.is_running:
            self._start()

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current >= ResidencyLevel.BOUND > target:
            self._stop()

    def close(self) -> None:
        self._stop()

    def rank_formulations(self, request: Mapping[str, object]) -> dict[str, Any]:
        payload = {"action": "rank_formulations", "request": dict(request)}
        return self._exchange(payload, {"RESULT"})["result"]

    def _exchange(self, payload: Mapping[str, object], expected: set[str]) -> dict[str, Any]:
        """Send one request and read its response as a single atomic exchange."""

        return run_exchange(
            self._exchange_lock,
            lambda: self._send(payload),
            lambda: self._read_message(expected),
        )

    def _send(self, payload: Mapping[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeError("robust-ranking worker is not running")
        process.stdin.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        process.stdin.flush()

    def _start(self) -> None:
        if not self.worker_script.exists():
            raise FileNotFoundError(self.worker_script)
        if self.bundle_dir is not None and not self.bundle_dir.exists():
            raise FileNotFoundError(self.bundle_dir)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if self.bundle_dir is not None:
            env["ORION_ROBUST_RANKING_BUNDLE_DIR"] = str(self.bundle_dir)
        existing = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT.parent)))
        if existing:
            env["PYTHONPATH"] += os.pathsep + existing
        started = monotonic()
        process = subprocess.Popen(
            [sys.executable, "-u", str(self.worker_script)],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        self._process = process
        try:
            with self._exchange_lock:
                self.boot_metadata = self._read_message({"BOOTED"})
                # Some resource-backed science workers must be assigned to
                # an actual OS Job Object BEFORE costly model import. Other
                # robust-ranking adapters remain unchanged by this hook.
                on_booted = getattr(self, "_on_booted", None)
                if on_booted is not None:
                    on_booted()
                self.ready_metadata = self._read_message({"READY"})
        except Exception:
            process.kill()
            process.wait(timeout=self.shutdown_timeout_s)
            self._process = None
            raise
        self.last_startup_latency_s = monotonic() - started

    def _read_message(self, expected: set[str]) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("robust-ranking worker is not running")
        logs: list[str] = []
        while True:
            raw = process.stdout.readline()
            if raw == "":
                stderr = process.stderr.read().strip() if process.stderr is not None else ""
                raise RuntimeError(
                    f"robust-ranking worker terminated before {sorted(expected)}: {stderr}"
                )
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logs.append(line)
                continue
            status = str(payload.get("status", ""))
            if status == "ERROR":
                if payload.get("error") == "RobustRankingError":
                    raise RobustRankingError(str(payload.get("message") or "invalid ranking request"))
                raise RuntimeError(f"robust-ranking worker error: {payload}")
            if status in expected:
                payload["worker_log_tail"] = logs[-20:]
                return payload
            logs.append(line)

    def _stop(self) -> None:
        process = self._process
        if process is None:
            return
        started = monotonic()
        if process.poll() is None:
            try:
                self._exchange({"action": "stop"}, {"STOPPED"})
                process.wait(timeout=self.shutdown_timeout_s)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=self.shutdown_timeout_s)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=self.shutdown_timeout_s)
        self.last_shutdown_latency_s = monotonic() - started
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        self._process = None
