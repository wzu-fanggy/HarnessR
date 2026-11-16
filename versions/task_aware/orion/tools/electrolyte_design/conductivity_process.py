"""Reusable CWSR Process adapter for the sodium-electrolyte conductivity model."""



from __future__ import annotations



import json

import os

from collections.abc import Callable

from pathlib import Path

import subprocess

import sys

from time import monotonic

from typing import Any



from capvm import ResidencyLevel, ResourceVector

from capvm.adapters.base import BaseCapabilityAdapter

from capvm.telemetry import process_rss_mb

from .conductivity_ml import ConductivityFeatures, ConductivityPrediction, OutOfDomainError

from .process_exchange import exchange_lock, run_exchange





ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WORKER = ROOT / "benchmark" / "cwsr_na_conductivity_ml_worker_v1.py"





class SodiumConductivityMlProcessAdapter(BaseCapabilityAdapter):

    """CWSR backing object that owns the complete pandas/scikit-learn worker."""



    def __init__(

        self,

        *,

        training_records: list[dict[str, Any]] | None = None,

        model_path: Path | None = None,

        model_path_factory: Callable[[], Path] | None = None,

        worker_script: Path = DEFAULT_WORKER,

        declared_host_ram_mb: float = 384.0,

        shutdown_timeout_s: float = 5.0,

    ) -> None:

        source_count = sum(

            item is not None for item in (training_records, model_path, model_path_factory)

        )

        if source_count != 1:

            raise ValueError(

                "provide exactly one of training_records, model_path or model_path_factory"

            )

        self.training_records = training_records

        self.model_path = model_path.resolve() if model_path is not None else None

        self.model_path_factory = model_path_factory

        self.worker_script = worker_script.resolve()

        self.declared_host_ram_mb = float(declared_host_ram_mb)

        self.shutdown_timeout_s = float(shutdown_timeout_s)

        self._process: subprocess.Popen[str] | None = None

        self.boot_metadata: dict[str, Any] | None = None

        self.ready_metadata: dict[str, Any] | None = None

        self.prediction_model_reference: dict[str, str] = {}

        self.last_startup_latency_s = 0.0

        self.last_shutdown_latency_s = 0.0

        self._exchange_lock = exchange_lock()



    @property

    def launcher_pid(self) -> int | None:

        """返回由 ``subprocess.Popen`` 直接创建的启动器 PID。"""



        process = self._process

        if process is None or process.poll() is not None:

            return None

        return int(process.pid)



    @property

    def pid(self) -> int | None:

        """返回真正执行模型代码的 worker PID。



        Windows venv 的 ``Scripts/python.exe`` 可能是 redirector：Popen 看到的是

        一个很小的 launcher，而解释器进程会以子进程形式真正导入 pandas/sklearn。

        worker 在 BOOTED 消息中回报自己的 ``os.getpid()``，因此运行资源测量应使用

        该 PID；没有回报时才退回 launcher PID。

        """



        launcher_pid = self.launcher_pid

        if launcher_pid is None:

            return None

        metadata = self.boot_metadata or {}

        worker_pid = metadata.get("pid")

        try:

            return int(worker_pid) if worker_pid is not None else launcher_pid

        except (TypeError, ValueError):

            return launcher_pid



    @property

    def is_running(self) -> bool:

        return self.launcher_pid is not None



    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:

        if level < ResidencyLevel.BOUND:

            return ResourceVector()

        return ResourceVector(

            host_ram_mb=self.declared_host_ram_mb,

            process_slots=1.0,

        )



    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:

        if level < ResidencyLevel.BOUND or self.pid is None:

            return ResourceVector()

        return ResourceVector(

            host_ram_mb=process_rss_mb(self.pid),

            process_slots=1.0,

        )



    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:

        if current < ResidencyLevel.BOUND <= target and not self.is_running:

            self._start()



    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:

        if current >= ResidencyLevel.BOUND > target:

            self._stop()



    def close(self) -> None:

        self._stop()



    def predict_features(self, features: dict[str, float]) -> dict[str, Any]:

        return self._exchange(

            {"action": "predict_features", "features": features},

            {"RESULT", "OUT_OF_DOMAIN"},

        )



    def _exchange(self, payload: dict[str, Any], expected: set[str]) -> dict[str, Any]:

        """Send one request and read its response as a single atomic exchange."""



        return run_exchange(

            self._exchange_lock,

            lambda: self._send(payload),

            lambda: self._read_message(expected),

        )



    def _send(self, payload: dict[str, Any]) -> None:

        process = self._process

        if process is None or process.stdin is None or process.poll() is not None:

            raise RuntimeError("conductivity ML worker is not running")

        process.stdin.write(

            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"

        )

        process.stdin.flush()



    def _start(self) -> None:

        if not self.worker_script.exists():

            raise FileNotFoundError(self.worker_script)

        if self.model_path is None and self.model_path_factory is not None:

            self.model_path = Path(self.model_path_factory()).resolve()

        if self.model_path is not None and not self.model_path.exists():

            raise FileNotFoundError(self.model_path)

        from .prediction_records import artifact_model_reference

        self.prediction_model_reference = artifact_model_reference(self.model_path) if self.model_path else {}

        env = os.environ.copy()

        env["PYTHONIOENCODING"] = "utf-8"

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

        handshake = (

            {"action": "load", "model_path": str(self.model_path)}

            if self.model_path is not None

            else {

                "action": "fit",

                "records": self.training_records,

                "n_estimators": 192,

                "random_state": 20260906,

            }

        )

        try:

            with self._exchange_lock:

                self.boot_metadata = self._read_message({"BOOTED"})

                self._send(handshake)

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

            raise RuntimeError("conductivity ML worker is not running")

        logs: list[str] = []

        while True:

            raw = process.stdout.readline()

            if raw == "":

                stderr = process.stderr.read().strip() if process.stderr is not None else ""

                raise RuntimeError(

                    f"ML worker terminated before {sorted(expected)}: {stderr}"

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

                raise RuntimeError(f"conductivity ML worker error: {payload}")

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

        if process.stdin is not None:

            process.stdin.close()

        if process.stdout is not None:

            process.stdout.close()

        if process.stderr is not None:

            process.stderr.close()

        self._process = None





class ConductivityProcessPredictor:

    """Expose a managed conductivity worker through the predictor API."""



    def __init__(self, adapter: SodiumConductivityMlProcessAdapter) -> None:

        self.adapter = adapter



    def predict_features(self, features: ConductivityFeatures) -> ConductivityPrediction:

        payload = self.adapter.predict_features(

            {

                "inverse_temperature": float(features.inverse_temperature),

                "x_NaSalt": float(features.x_NaSalt),

                "x_EC": float(features.x_EC),

                "x_PC": float(features.x_PC),

            }

        )

        status = str(payload.get("status") or "")

        if status == "OUT_OF_DOMAIN":

            raise OutOfDomainError(str(payload.get("reason") or "out of domain"))

        if status != "RESULT":

            raise RuntimeError(f"unexpected conductivity worker status: {status!r}")

        raw_prediction = payload.get("prediction")

        if not isinstance(raw_prediction, dict):

            raise RuntimeError("conductivity worker RESULT is missing prediction payload")

        return ConductivityPrediction(

            conductivity_s_cm=float(raw_prediction["conductivity_s_cm"]),

            conductivity_ms_cm=float(raw_prediction["conductivity_ms_cm"]),

            log10_conductivity_s_cm=float(raw_prediction["log10_conductivity_s_cm"]),

            ensemble_std_log10=float(raw_prediction["ensemble_std_log10"]),

            estimated_factor_1sigma=float(raw_prediction["estimated_factor_1sigma"]),

            in_domain=bool(raw_prediction.get("in_domain", True)),

            features=features,

        )

