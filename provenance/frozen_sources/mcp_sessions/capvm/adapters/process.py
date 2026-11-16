"""真实子进程 backing resource 的 Capability adapter。"""

from __future__ import annotations

import subprocess
import sys
from time import monotonic

from ..model import ResidencyLevel, ResourceVector
from ..telemetry import process_rss_mb
from .base import BaseCapabilityAdapter


class ProcessCapabilityAdapter(BaseCapabilityAdapter):
    """在 R3+ 启动真实 worker process，降到 R2- 时释放。

    ``resident_mb`` 表示 worker payload；``runtime_overhead_mb`` 可显式计入解释器/运行时
    基线的声明成本。measured cost 仍读取真实 child RSS。worker 会实际分配并触碰对应
    bytearray，避免 benchmark 只统计虚拟地址空间。
    """

    def __init__(
        self,
        *,
        resident_mb: int = 8,
        runtime_overhead_mb: float = 0.0,
        startup_delay_s: float = 0.01,
        shutdown_timeout_s: float = 2.0,
        worker_executable: str | None = None,
    ) -> None:
        if resident_mb < 1:
            raise ValueError("resident_mb must be >= 1")
        if runtime_overhead_mb < 0:
            raise ValueError("runtime_overhead_mb must be non-negative")
        if startup_delay_s < 0:
            raise ValueError("startup_delay_s must be non-negative")
        self.resident_mb = int(resident_mb)
        self.runtime_overhead_mb = float(runtime_overhead_mb)
        self.startup_delay_s = float(startup_delay_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        selected_executable = str(worker_executable or "").strip()
        if not selected_executable:
            # Windows venv 的 Scripts/python.exe 可能是 redirector：Popen.pid 指向一个
            # 很小的 launcher，而实际 Python worker 是其子进程。若按 launcher PID 采 RSS，
            # payload 增长不会进入 telemetry，close 也可能只管理到壳进程。worker 仅依赖
            # stdlib，因此 Windows 下优先直接启动 base interpreter，使 adapter.pid 就是
            # allocation-bearing process。其他平台保留当前 sys.executable 语义。
            base_executable = str(getattr(sys, "_base_executable", "") or "").strip()
            if sys.platform == "win32" and base_executable:
                selected_executable = base_executable
            else:
                selected_executable = sys.executable
        self.worker_executable = selected_executable
        self._process: subprocess.Popen[str] | None = None
        self.last_startup_latency_s = 0.0
        self.last_shutdown_latency_s = 0.0

    @property
    def pid(self) -> int | None:
        process = self._process
        if process is None or process.poll() is not None:
            return None
        return int(process.pid)

    @property
    def is_running(self) -> bool:
        return self.pid is not None

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND:
            return ResourceVector()
        return ResourceVector(
            host_ram_mb=float(self.resident_mb) + self.runtime_overhead_mb,
            process_slots=1.0,
        )

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND:
            return ResourceVector()
        pid = self.pid
        if pid is None:
            raise RuntimeError("驻留的 Process worker 已退出，无法复核资源占用")
        rss = process_rss_mb(pid)
        if rss <= 0:
            raise RuntimeError(f"Process worker {pid} 的 WorkingSet 测量不可用")
        return ResourceVector(
            host_ram_mb=rss,
            process_slots=1.0,
        )

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current < ResidencyLevel.BOUND <= target and not self.is_running:
            self._start_worker()

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current >= ResidencyLevel.BOUND > target:
            self._stop_worker()

    def close(self) -> None:
        self._stop_worker()

    def _start_worker(self) -> None:
        allocation_bytes = self.resident_mb * 1024 * 1024
        worker_code = (
            "import sys,time;"
            "n=int(sys.argv[1]);delay=float(sys.argv[2]);"
            "buf=bytearray(n);"
            "page=4096;"
            "[(buf.__setitem__(i,1)) for i in range(0,n,page)];"
            "time.sleep(delay);"
            "print('READY',flush=True);"
            "time.sleep(3600)"
        )
        started = monotonic()
        process = subprocess.Popen(
            [
                self.worker_executable,
                "-c",
                worker_code,
                str(allocation_bytes),
                str(self.startup_delay_s),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdout is not None
            marker = process.stdout.readline().strip()
            if marker != "READY":
                stderr = ""
                if process.stderr is not None:
                    stderr = process.stderr.read().strip()
                raise RuntimeError(f"process capability worker failed to start: {stderr}")
        except Exception:
            process.kill()
            process.wait(timeout=self.shutdown_timeout_s)
            raise
        self._process = process
        self.last_startup_latency_s = monotonic() - started

    def _stop_worker(self) -> None:
        process = self._process
        if process is None:
            return
        started = monotonic()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.shutdown_timeout_s)
        self.last_shutdown_latency_s = monotonic() - started
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        self._process = None
