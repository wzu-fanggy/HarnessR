"""可演化 capability plugin 的最小可移植 bundle。

V2 bundle 不打包 Python 源码；它假设目标 LangGraph runtime 已安装兼容的 plugin
执行类。Bundle 负责携带 capability identity、版本、执行契约元数据和已验证 learned
state。这个边界故意把“代码分发”与“学习状态迁移”分开，便于后续替换为 wheel/OCI/
Agent-Plugin package 等真实发布格式。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .evolvable_plugin import EvolvableLangGraphPlugin, JsonCapabilityLearnedStateStore


_BUNDLE_SCHEMA_VERSION = "orion-evolvable-plugin-bundle-v1"


@dataclass(frozen=True)
class CapabilityPluginBundleManifest:
    plugin_id: str
    version: str
    fallback_action: str
    resident_mb: float
    learned_generation: int


class CapabilityPluginBundle:
    """导出/导入一个 capability 的 durable identity + learned state。"""

    MANIFEST_NAME = "manifest.json"
    STATE_NAME = "learned_state.json"

    @classmethod
    def export(
        cls,
        plugin: EvolvableLangGraphPlugin,
        destination: str | Path,
    ) -> Path:
        destination_path = Path(destination)
        destination_path.mkdir(parents=True, exist_ok=True)
        state = plugin.learned_state
        manifest = {
            "schema_version": _BUNDLE_SCHEMA_VERSION,
            "plugin_id": plugin.plugin_id,
            "version": plugin.version,
            "fallback_action": plugin.fallback_action,
            "resident_mb": plugin.resident_mb,
            "learned_generation": state.generation,
        }
        (destination_path / cls.MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        # 如果尚未写过 state 文件，先物化 generation=0 的状态，保证 bundle 自包含。
        if not plugin.state_store.path.exists():
            plugin.state_store.save(state)
        shutil.copy2(plugin.state_store.path, destination_path / cls.STATE_NAME)
        return destination_path

    @classmethod
    def inspect(cls, bundle_path: str | Path) -> CapabilityPluginBundleManifest:
        bundle = Path(bundle_path)
        payload = json.loads((bundle / cls.MANIFEST_NAME).read_text(encoding="utf-8"))
        if payload.get("schema_version") != _BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported plugin-bundle schema")
        return CapabilityPluginBundleManifest(
            plugin_id=str(payload["plugin_id"]),
            version=str(payload["version"]),
            fallback_action=str(payload["fallback_action"]),
            resident_mb=float(payload.get("resident_mb", 0.0)),
            learned_generation=int(payload.get("learned_generation", 0)),
        )

    @classmethod
    def import_plugin(
        cls,
        bundle_path: str | Path,
        *,
        state_root: str | Path,
        memory_namespace: str | None = None,
    ) -> EvolvableLangGraphPlugin:
        bundle = Path(bundle_path)
        manifest = cls.inspect(bundle)
        plugin = EvolvableLangGraphPlugin(
            plugin_id=manifest.plugin_id,
            version=manifest.version,
            state_root=state_root,
            fallback_action=manifest.fallback_action,
            resident_mb=manifest.resident_mb,
            memory_namespace=memory_namespace,
        )
        target_store = JsonCapabilityLearnedStateStore(
            state_root,
            memory_namespace or manifest.plugin_id,
        )
        target_store.namespace_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / cls.STATE_NAME, target_store.path)
        restored = target_store.load()
        if restored.generation != manifest.learned_generation:
            raise ValueError("bundle learned-state generation mismatch")
        return plugin
