"""Portable Capability Asset：把 capability lifecycle 元数据统一成可校验 bundle。

V11 不发明代码分发格式，也不把 Python 源码塞进资产包。目标 Host 仍需安装兼容的
executable implementation。该 bundle 只把此前分散的 identity/version、durable learned
state、evolution/validation record、deployment contracts 与 residency metadata 放在同一
可审计边界中。

Asset 的 state envelope 与具体 plugin class 解耦。默认 loader 可恢复 V0/V2 的
``EvolvableLangGraphPlugin``；真实 scientific ``StatefulLangGraphCapabilityPlugin`` 等实现
可通过 ``plugin_factory`` 注入兼容 executable，而无需改变资产格式。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .deployment import CapabilityDeploymentQualifier
from .evolvable_plugin import (
    CapabilityLearnedState,
    EvolvableLangGraphPlugin,
    JsonCapabilityLearnedStateStore,
)
from .plugin_evolution import PluginEvolutionResult


_ASSET_SCHEMA_VERSION = "orion-capability-asset-v1"
_STATE_ENVELOPE_SCHEMA_VERSION = "orion-capability-state-envelope-v1"
_DEFAULT_EXECUTABLE_REF = "orion.capvm:EvolvableLangGraphPlugin"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PortableCapabilityPlugin(Protocol):
    """Capability Asset 只要求的最小 stateful executable contract。"""

    plugin_id: str
    version: str
    resident_mb: float
    state_store: JsonCapabilityLearnedStateStore

    @property
    def learned_state(self) -> CapabilityLearnedState: ...

    def mount(self) -> None: ...

    def unmount(self) -> None: ...


@dataclass(frozen=True)
class CapabilityEvolutionRecord:
    """可随 Capability Asset 迁移的最小 validation/evolution 审计记录。"""

    validation_id: str
    generation_before: int
    generation_after: int
    promoted: bool
    rolled_back: bool
    baseline_correct: int
    candidate_correct: int
    case_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.validation_id.strip():
            raise ValueError("validation_id is required")
        if self.generation_before < 0 or self.generation_after < 0:
            raise ValueError("generation must be non-negative")
        if self.baseline_correct < 0 or self.candidate_correct < 0 or self.case_count < 0:
            raise ValueError("validation counts must be non-negative")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_result(
        cls,
        result: PluginEvolutionResult,
        *,
        validation_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "CapabilityEvolutionRecord":
        return cls(
            validation_id=validation_id,
            generation_before=result.generation_before,
            generation_after=result.generation_after,
            promoted=result.promoted,
            rolled_back=result.rolled_back,
            baseline_correct=result.baseline_correct,
            candidate_correct=result.candidate_correct,
            case_count=result.case_count,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityEvolutionRecord":
        return cls(
            validation_id=str(payload["validation_id"]),
            generation_before=int(payload["generation_before"]),
            generation_after=int(payload["generation_after"]),
            promoted=bool(payload["promoted"]),
            rolled_back=bool(payload["rolled_back"]),
            baseline_correct=int(payload["baseline_correct"]),
            candidate_correct=int(payload["candidate_correct"]),
            case_count=int(payload["case_count"]),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class CapabilityAssetManifest:
    """Capability Asset 的轻量 identity/lifecycle manifest。"""

    capability_id: str
    executable_version: str
    executable_ref: str
    learned_generation: int
    resident_mb: float
    deployment_contract_count: int
    evolution_record_count: int
    artifact_sha256: Mapping[str, str]
    executable_config: Mapping[str, Any] = field(default_factory=dict)
    state_namespace: str | None = None

    def __post_init__(self) -> None:
        if not self.capability_id.strip():
            raise ValueError("capability_id is required")
        if not self.executable_version.strip():
            raise ValueError("executable_version is required")
        if not self.executable_ref.strip():
            raise ValueError("executable_ref is required")
        if self.learned_generation < 0:
            raise ValueError("learned_generation must be non-negative")
        if self.resident_mb < 0:
            raise ValueError("resident_mb must be non-negative")
        if self.deployment_contract_count < 0 or self.evolution_record_count < 0:
            raise ValueError("asset record counts must be non-negative")
        state_namespace = (
            str(self.state_namespace).strip() if self.state_namespace is not None else None
        )
        if state_namespace == "":
            state_namespace = None
        object.__setattr__(self, "state_namespace", state_namespace)
        object.__setattr__(self, "artifact_sha256", dict(self.artifact_sha256))
        object.__setattr__(self, "executable_config", dict(self.executable_config))


@dataclass(frozen=True)
class CapabilityAssetImportResult:
    """导入后的 executable instance 与同行 lifecycle 元数据。"""

    plugin: PortableCapabilityPlugin
    manifest: CapabilityAssetManifest
    evolution_records: tuple[CapabilityEvolutionRecord, ...]
    deployment_contract_count: int


PluginFactory = Callable[
    [CapabilityAssetManifest, Path, str],
    PortableCapabilityPlugin,
]


class CapabilityAssetBundle:
    """统一导出/校验/导入一个 stateful evolvable capability asset。"""

    MANIFEST_NAME = "asset_manifest.json"
    PLUGIN_DIR = "plugin"
    PLUGIN_MANIFEST_NAME = "manifest.json"
    STATE_NAME = "learned_state.json"
    DEPLOYMENT_NAME = "deployment_contracts.json"
    EVOLUTION_NAME = "evolution_records.json"

    @classmethod
    def export(
        cls,
        plugin: PortableCapabilityPlugin,
        destination: str | Path,
        *,
        qualifier: CapabilityDeploymentQualifier | None = None,
        evolution_records: Sequence[CapabilityEvolutionRecord] = (),
        executable_ref: str = _DEFAULT_EXECUTABLE_REF,
        executable_config: Mapping[str, Any] | None = None,
    ) -> Path:
        destination_path = Path(destination)
        destination_path.mkdir(parents=True, exist_ok=True)
        plugin_dir = destination_path / cls.PLUGIN_DIR
        plugin_dir.mkdir(parents=True, exist_ok=True)

        state = plugin.learned_state
        if not plugin.state_store.path.exists():
            plugin.state_store.save(state)
        shutil.copy2(plugin.state_store.path, plugin_dir / cls.STATE_NAME)

        config = dict(executable_config or {})
        # 默认 simple-plugin loader 需要 fallback_action；通用 stateful plugin 则由调用方
        # 提供 factory，不要求把 graph_factory 序列化进资产。
        if "fallback_action" not in config and hasattr(plugin, "fallback_action"):
            config["fallback_action"] = str(getattr(plugin, "fallback_action"))

        state_namespace = str(plugin.state_store.plugin_id)
        plugin_manifest_payload = {
            "schema_version": _STATE_ENVELOPE_SCHEMA_VERSION,
            "plugin_id": plugin.plugin_id,
            "version": plugin.version,
            "resident_mb": float(plugin.resident_mb),
            "learned_generation": state.generation,
            "state_namespace": state_namespace,
        }
        (plugin_dir / cls.PLUGIN_MANIFEST_NAME).write_text(
            json.dumps(
                plugin_manifest_payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )

        contracts = (
            qualifier.export_contracts(plugin.plugin_id) if qualifier is not None else []
        )
        (destination_path / cls.DEPLOYMENT_NAME).write_text(
            json.dumps(contracts, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        evolution_payload = [item.to_dict() for item in evolution_records]
        (destination_path / cls.EVOLUTION_NAME).write_text(
            json.dumps(evolution_payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        artifact_paths = (
            f"{cls.PLUGIN_DIR}/{cls.PLUGIN_MANIFEST_NAME}",
            f"{cls.PLUGIN_DIR}/{cls.STATE_NAME}",
            cls.DEPLOYMENT_NAME,
            cls.EVOLUTION_NAME,
        )
        artifact_hashes = {
            relative: _sha256(destination_path / relative) for relative in artifact_paths
        }
        manifest_payload = {
            "schema_version": _ASSET_SCHEMA_VERSION,
            "capability_id": plugin.plugin_id,
            "executable_version": plugin.version,
            "executable_ref": executable_ref,
            "executable_config": config,
            "state_namespace": state_namespace,
            "learned_generation": state.generation,
            "resident_mb": float(plugin.resident_mb),
            "deployment_contract_count": len(contracts),
            "evolution_record_count": len(evolution_payload),
            "artifact_sha256": dict(sorted(artifact_hashes.items())),
        }
        (destination_path / cls.MANIFEST_NAME).write_text(
            json.dumps(manifest_payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return destination_path

    @classmethod
    def inspect(
        cls,
        bundle_path: str | Path,
        *,
        verify_integrity: bool = True,
    ) -> CapabilityAssetManifest:
        bundle = Path(bundle_path)
        payload = json.loads((bundle / cls.MANIFEST_NAME).read_text(encoding="utf-8"))
        if payload.get("schema_version") != _ASSET_SCHEMA_VERSION:
            raise ValueError("unsupported capability-asset schema")
        manifest = CapabilityAssetManifest(
            capability_id=str(payload["capability_id"]),
            executable_version=str(payload["executable_version"]),
            executable_ref=str(payload["executable_ref"]),
            learned_generation=int(payload["learned_generation"]),
            resident_mb=float(payload.get("resident_mb", 0.0)),
            deployment_contract_count=int(payload.get("deployment_contract_count", 0)),
            evolution_record_count=int(payload.get("evolution_record_count", 0)),
            artifact_sha256=dict(payload.get("artifact_sha256") or {}),
            executable_config=dict(payload.get("executable_config") or {}),
            state_namespace=(
                str(payload["state_namespace"])
                if payload.get("state_namespace") is not None
                else None
            ),
        )
        if verify_integrity:
            cls.verify(bundle, manifest=manifest)
        return manifest

    @classmethod
    def verify(
        cls,
        bundle_path: str | Path,
        *,
        manifest: CapabilityAssetManifest | None = None,
    ) -> None:
        bundle = Path(bundle_path)
        if manifest is None:
            manifest = cls.inspect(bundle, verify_integrity=False)
        for relative, expected in manifest.artifact_sha256.items():
            path = bundle / relative
            if not path.exists():
                raise ValueError(f"capability-asset artifact missing: {relative}")
            observed = _sha256(path)
            if observed != expected:
                raise ValueError(f"capability-asset integrity mismatch: {relative}")

    @classmethod
    def load_evolution_records(
        cls,
        bundle_path: str | Path,
    ) -> tuple[CapabilityEvolutionRecord, ...]:
        bundle = Path(bundle_path)
        payload = json.loads((bundle / cls.EVOLUTION_NAME).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("evolution_records must be a list")
        return tuple(CapabilityEvolutionRecord.from_dict(item) for item in payload)

    @classmethod
    def _read_state_envelope(cls, bundle: Path) -> dict[str, Any]:
        payload = json.loads(
            (bundle / cls.PLUGIN_DIR / cls.PLUGIN_MANIFEST_NAME).read_text(
                encoding="utf-8"
            )
        )
        if payload.get("schema_version") != _STATE_ENVELOPE_SCHEMA_VERSION:
            raise ValueError("unsupported capability state-envelope schema")
        return payload

    @classmethod
    def _default_plugin_factory(
        cls,
        manifest: CapabilityAssetManifest,
        state_root: Path,
        namespace: str,
    ) -> PortableCapabilityPlugin:
        fallback_action = str(manifest.executable_config.get("fallback_action") or "").strip()
        if not fallback_action:
            raise ValueError(
                "asset requires an explicit plugin_factory for this executable_ref"
            )
        return EvolvableLangGraphPlugin(
            plugin_id=manifest.capability_id,
            version=manifest.executable_version,
            state_root=state_root,
            fallback_action=fallback_action,
            resident_mb=manifest.resident_mb,
            memory_namespace=namespace,
        )

    @classmethod
    def import_asset(
        cls,
        bundle_path: str | Path,
        *,
        state_root: str | Path,
        qualifier: CapabilityDeploymentQualifier | None = None,
        memory_namespace: str | None = None,
        plugin_factory: PluginFactory | None = None,
    ) -> CapabilityAssetImportResult:
        bundle = Path(bundle_path)
        manifest = cls.inspect(bundle, verify_integrity=True)
        envelope = cls._read_state_envelope(bundle)
        if str(envelope["plugin_id"]) != manifest.capability_id:
            raise ValueError("capability-asset/plugin identity mismatch")
        if str(envelope["version"]) != manifest.executable_version:
            raise ValueError("capability-asset executable version mismatch")
        if int(envelope["learned_generation"]) != manifest.learned_generation:
            raise ValueError("capability-asset learned generation mismatch")

        target_root = Path(state_root)
        source_namespace = str(envelope.get("state_namespace") or manifest.capability_id)
        target_namespace = memory_namespace or source_namespace
        target_store = JsonCapabilityLearnedStateStore(target_root, target_namespace)
        target_store.namespace_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / cls.PLUGIN_DIR / cls.STATE_NAME, target_store.path)
        restored = target_store.load()
        if restored.generation != manifest.learned_generation:
            raise ValueError("capability-asset restored generation mismatch")

        factory = plugin_factory or cls._default_plugin_factory
        plugin = factory(manifest, target_root, target_namespace)
        if plugin.plugin_id != manifest.capability_id:
            raise ValueError("plugin_factory returned wrong capability identity")
        if plugin.version != manifest.executable_version:
            raise ValueError("plugin_factory returned wrong executable version")
        if plugin.learned_state.generation != manifest.learned_generation:
            raise ValueError("plugin_factory did not bind restored learned state")

        contracts_payload = json.loads(
            (bundle / cls.DEPLOYMENT_NAME).read_text(encoding="utf-8")
        )
        if not isinstance(contracts_payload, list):
            raise ValueError("deployment_contracts must be a list")
        if len(contracts_payload) != manifest.deployment_contract_count:
            raise ValueError("capability-asset deployment contract count mismatch")
        if qualifier is not None:
            qualifier.import_contracts(contracts_payload)

        evolution_records = cls.load_evolution_records(bundle)
        if len(evolution_records) != manifest.evolution_record_count:
            raise ValueError("capability-asset evolution record count mismatch")

        return CapabilityAssetImportResult(
            plugin=plugin,
            manifest=manifest,
            evolution_records=evolution_records,
            deployment_contract_count=len(contracts_payload),
        )
