"""ORION 主 Agent 的本地 hot-pluggable capability gateway。

主 ReAct/LangGraph 图只长期绑定两个稳定工具：列出动态能力和调用动态能力。第三方
capability 的增加、删除和代码更新只改变 gateway 后面的 registry，不要求重新
``bind_tools()`` 或重新编译主图。

用户侧 package 发现/批准由 ``LocalCapabilityFolderManager`` 负责；这里保持一个懒加载
的进程级 manager，CLI/Web 层可以复用同一个实例执行 attach/detach。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import tool

from orion.capvm import (
    LocalCapabilityEvent,
    LocalCapabilityFolderManager,
    LocalCapabilityStatus,
    StableLangGraphPluginHost,
)


_MANAGER_LOCK = threading.RLock()
_MANAGER: LocalCapabilityFolderManager | None = None
_RUNTIME_SYNC_HOOK: Callable[[], None] | None = None


def get_local_capability_manager() -> LocalCapabilityFolderManager:
    """返回进程级本地 capability manager，首次访问时才创建目录和稳定 Host。"""

    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is not None:
            return _MANAGER
        project_root = Path(__file__).resolve().parents[1]
        capability_root = Path(
            os.environ.get("ORION_CAPABILITY_ROOT", project_root / "capabilities")
        )
        state_root = Path(
            os.environ.get(
                "ORION_CAPABILITY_STATE_ROOT",
                project_root / ".orion_capability_state",
            )
        )
        host = StableLangGraphPluginHost(
            host_id="orion-main-capability-host",
            topology="audited",
        )
        _MANAGER = LocalCapabilityFolderManager(
            root=capability_root,
            state_root=state_root,
            host=host,
        )
        return _MANAGER


def set_local_capability_runtime_sync_hook(hook: Callable[[], None] | None) -> None:
    """Install the runtime-state synchronization hook used after safe lifecycle changes."""

    global _RUNTIME_SYNC_HOOK
    with _MANAGER_LOCK:
        _RUNTIME_SYNC_HOOK = hook


def _sync_runtime_state() -> None:
    hook = _RUNTIME_SYNC_HOOK
    if hook is not None:
        hook()


def reset_local_capability_manager_for_tests() -> None:
    """仅供测试隔离进程级 manager；生产路径不应调用。"""

    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is not None:
            mounted = _MANAGER.host.registry.mounted_ids()
            for capability_id in mounted:
                _MANAGER.detach(capability_id)
        _MANAGER = None
        _sync_runtime_state()


def scan_local_capabilities() -> tuple[LocalCapabilityEvent, ...]:
    """在安全 turn 边界扫描本地 capability 目录，不执行未批准的第三方代码。"""

    with _MANAGER_LOCK:
        events = get_local_capability_manager().scan()
        _sync_runtime_state()
        return events


def attach_local_capability(capability_id: str):
    """显式批准并接入一个已发现 capability；更新 package 时走 state-preserving hot swap。"""

    with _MANAGER_LOCK:
        result = get_local_capability_manager().approve(capability_id)
        _sync_runtime_state()
        return result


def detach_local_capability(capability_id: str) -> None:
    """从当前运行中的 Agent 热拔 capability，但不删除用户下载的 package。"""

    with _MANAGER_LOCK:
        get_local_capability_manager().detach(capability_id)
        _sync_runtime_state()


def local_capability_snapshot(*, scan: bool = True) -> list[dict[str, Any]]:
    """返回适合 CLI/UI/LLM 展示的动态 capability descriptor 快照。"""

    with _MANAGER_LOCK:
        manager = get_local_capability_manager()
        if scan:
            manager.scan()
        mounted = set(manager.host.registry.mounted_ids())
        rows: list[dict[str, Any]] = []
        for record in manager.records():
            generation: int | None = None
            active_version: str | None = None
            if record.manifest.capability_id in mounted:
                plugin = manager.host.registry.get(record.manifest.capability_id)
                active_version = plugin.version
                generation = int(plugin.learned_state.generation)
            rows.append(
                {
                    "id": record.manifest.capability_id,
                    "version": record.manifest.version,
                    "active_version": active_version,
                    "summary": record.manifest.summary,
                    "status": record.status.value,
                    "permissions": list(record.manifest.permissions),
                    "metadata": dict(record.manifest.metadata),
                    "learned_generation": generation,
                    "package_dir": str(record.package_dir),
                    "last_error": record.last_error,
                }
            )
        return rows


@tool
def list_local_capabilities() -> list[dict[str, Any]]:
    """列出 ORION 本地 capability 目录中已发现的动态能力及连接状态。

    本工具只读取 capability manifest/状态，不会自动执行尚未批准的第三方代码。
    ``pending_approval`` 或 ``update_pending`` 的能力必须先由用户在 CLI/UI 显式批准；
    ``attached`` 的能力可以通过 ``invoke_local_capability`` 调用。
    """

    # 动态目录扫描只在 CLI/宿主调度器定义的安全 turn 边界执行；ToolNode 可能并行
    # 执行多个工具，因此这里绝不在 invocation 途中触发 attach/detach reconciliation。
    return local_capability_snapshot(scan=False)


@tool
def invoke_local_capability(
    capability_id: str,
    signal: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """调用一个已经热挂载到当前 ORION runtime 的本地 capability。

    参数：
    - ``capability_id``：能力 manifest 中的稳定 ID；
    - ``signal``：适用于简单 signal→action capability；
    - ``payload``：适用于结构化 scientific/LangGraph capability 的 JSON 对象。

    ``signal`` 与 ``payload`` 必须且只能提供一个。未批准或未连接的 capability 会拒绝
    调用，不会在调用路径中偷偷加载第三方代码。
    """

    if (signal is None) == (payload is None):
        raise ValueError("provide exactly one of signal or payload")
    manager = get_local_capability_manager()
    record = manager.get(capability_id)
    if record.status != LocalCapabilityStatus.ATTACHED:
        raise RuntimeError(
            f"capability is not attached: {capability_id} ({record.status.value})"
        )
    if signal is not None:
        return dict(manager.host.invoke(capability_id=capability_id, signal=signal))
    assert payload is not None
    return manager.host.invoke_payload(capability_id=capability_id, payload=payload)
