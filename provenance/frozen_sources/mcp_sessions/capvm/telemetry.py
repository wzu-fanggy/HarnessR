"""Capability Virtual Memory 的轻量 OS 资源测量。

V0 不引入 psutil 依赖。Windows 使用 GetProcessMemoryInfo，Linux 使用 /proc。
底层测量不可用时返回 0；驻留的 Process adapter 会把零读数转成错误，阻止准入。
"""

from __future__ import annotations

import os
from pathlib import Path


def process_rss_mb(pid: int) -> float:
    """返回指定进程当前 WorkingSet/RSS（MiB）；底层不可用时返回 0。"""

    if pid <= 0:
        return 0.0
    if os.name == "nt":
        return _windows_process_rss_mb(pid)
    return _procfs_process_rss_mb(pid)


def host_ram_pressure() -> float | None:
    """Return current system RAM utilization as a fraction when available."""

    try:
        import psutil

        pressure = float(psutil.virtual_memory().percent) / 100.0
    except (ImportError, OSError, AttributeError, TypeError, ValueError):
        return None
    return round(pressure, 6) if 0.0 <= pressure <= 1.0 else None


def _procfs_process_rss_mb(pid: int) -> float:
    statm = Path(f"/proc/{pid}/statm")
    try:
        fields = statm.read_text(encoding="utf-8").split()
        resident_pages = int(fields[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return resident_pages * page_size / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError, AttributeError):
        return 0.0


def _windows_process_rss_mb(pid: int) -> float:
    try:
        import ctypes
        from ctypes import wintypes

        process_query_information = 0x0400
        process_vm_read = 0x0010

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(
            process_query_information | process_vm_read,
            False,
            pid,
        )
        if not handle:
            return 0.0
        try:
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            ok = psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(counters),
                counters.cb,
            )
            if not ok:
                return 0.0
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError, ValueError):
        return 0.0
