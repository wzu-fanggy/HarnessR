"""稳定 local-capability host 的插件执行状态 residency adapter。"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from ..model import ResidencyLevel, ResourceVector
from .base import BaseCapabilityAdapter


class PluginRegistryLike(Protocol):
    def mounted_ids(self) -> tuple[str, ...]: ...
    def get(self, plugin_id: str) -> Any: ...


class LocalPluginRuntimeStateAdapter(BaseCapabilityAdapter):
    """把已批准 local capability 的已挂载执行状态作为一个 PLUGIN 资源管理。

    registry 中保留逻辑 plugin 对象；降驻留只调用 plugin.unmount() 释放执行侧状态，
    不撤销用户批准、不删除 package，也不改变稳定 gateway。再次 acquire 时重新 mount。
    """

    def __init__(self, registry_factory: Callable[[], PluginRegistryLike]) -> None:
        self._registry_factory = registry_factory

    def _plugins(self) -> tuple[Any, ...]:
        registry = self._registry_factory()
        return tuple(registry.get(item) for item in registry.mounted_ids())

    @staticmethod
    def _declared_plugin_mb(plugin: Any) -> float:
        try:
            return max(0.0, float(getattr(plugin, "resident_mb", 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _measured_plugin_mb(plugin: Any) -> float:
        try:
            resident_bytes = float(getattr(plugin, "resident_bytes", 0.0) or 0.0)
        except (TypeError, ValueError):
            resident_bytes = 0.0
        return max(0.0, resident_bytes / (1024.0 * 1024.0))

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND:
            return ResourceVector()
        return ResourceVector(
            host_ram_mb=sum(self._declared_plugin_mb(plugin) for plugin in self._plugins())
        )

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND:
            return ResourceVector()
        return ResourceVector(
            host_ram_mb=sum(
                self._measured_plugin_mb(plugin)
                for plugin in self._plugins()
                if bool(getattr(plugin, "is_mounted", False))
            )
        )

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current < ResidencyLevel.BOUND <= target:
            for plugin in self._plugins():
                if not bool(getattr(plugin, "is_mounted", False)):
                    plugin.mount()

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current >= ResidencyLevel.BOUND > target:
            for plugin in self._plugins():
                if bool(getattr(plugin, "is_mounted", False)):
                    plugin.unmount()

    def close(self) -> None:
        for plugin in self._plugins():
            if bool(getattr(plugin, "is_mounted", False)):
                plugin.unmount()

    def has_plugins(self) -> bool:
        return bool(self._plugins())

    def any_mounted(self) -> bool:
        return any(bool(getattr(plugin, "is_mounted", False)) for plugin in self._plugins())
