"""本地 capability 文件夹的零依赖发现与热挂载前端。

本模块把“用户手动下载第三方能力并拖入 ``capabilities/``”映射为稳定 Host 上的
动态 capability lifecycle。扫描阶段只读取 manifest 和文件哈希，不 import/执行第三方
Python 代码；只有显式 ``approve()`` 后才解析 entrypoint 并 mount/hot-swap。

这里故意不实现云端 registry、插件市场或依赖安装器。它只负责本地 USB 式前端：

``drop folder -> discover -> approve -> attach``

已挂载目录被移除时，下一次安全扫描会 detach；已挂载目录内容变化时进入
``UPDATE_PENDING``，不会自动执行新代码，显式批准后复用 Host 的 state-preserving
hot swap。调用方应在 Agent turn 边界等无并发 invocation 的安全点执行 ``scan()``。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

from .plugin_host import PluginHotSwapResult, StableLangGraphPluginHost, StatefulCapabilityExecutable


_MANIFEST_SCHEMA_VERSION = "orion-local-capability-v1"
_MANIFEST_NAMES = ("capability.toml", "capability.json")
_IGNORED_PARTS = {"__pycache__", ".git", ".runtime"}


class LocalCapabilityStatus(str, Enum):
    """本地 capability package 在 Host 前端中的状态。"""

    PENDING_APPROVAL = "pending_approval"
    ATTACHED = "attached"
    UPDATE_PENDING = "update_pending"
    DETACHED = "detached"
    FAILED = "failed"


class LocalCapabilityEventType(str, Enum):
    """一次目录扫描产生的用户可见事件。"""

    DISCOVERED = "discovered"
    UPDATED = "updated"
    REMOVED = "removed"
    INVALID = "invalid"


@dataclass(frozen=True)
class LocalCapabilityManifest:
    """扫描阶段可安全读取的最小 capability descriptor。"""

    capability_id: str
    version: str
    summary: str
    entrypoint: str
    permissions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        capability_id = self.capability_id.strip()
        version = self.version.strip()
        summary = self.summary.strip()
        entrypoint = self.entrypoint.strip()
        if not capability_id:
            raise ValueError("capability id is required")
        if not version:
            raise ValueError("capability version is required")
        if not summary:
            raise ValueError("capability summary is required")
        if not entrypoint or ":" not in entrypoint:
            raise ValueError("entrypoint must use '<module-or-file>:<factory>'")
        object.__setattr__(self, "capability_id", capability_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "entrypoint", entrypoint)
        object.__setattr__(
            self,
            "permissions",
            tuple(sorted({str(item).strip() for item in self.permissions if str(item).strip()})),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class LocalCapabilityPackageContext:
    """批准后传给第三方 factory 的受控构建上下文。"""

    package_dir: Path
    state_root: Path
    manifest: LocalCapabilityManifest


@dataclass(frozen=True)
class LocalCapabilityEvent:
    """供 CLI/UI 转成“发现新能力/能力已变化”的提示。"""

    event_type: LocalCapabilityEventType
    capability_id: str | None
    package_dir: Path
    message: str


@dataclass
class LocalCapabilityRecord:
    """一个已发现 package 的本地状态。"""

    manifest: LocalCapabilityManifest
    package_dir: Path
    fingerprint: str
    status: LocalCapabilityStatus
    active_fingerprint: str | None = None
    last_error: str | None = None


CapabilityFactory = Callable[[LocalCapabilityPackageContext], StatefulCapabilityExecutable]
CandidateValidator = Callable[[StatefulCapabilityExecutable], None]


class LocalCapabilityFolderManager:
    """扫描本地目录并把经批准的 package 动态接入稳定 Host。

    ``scan()`` 不执行第三方代码，因此可以频繁在 Agent turn 边界调用。新目录进入
    ``PENDING_APPROVAL``；已挂载目录内容变化进入 ``UPDATE_PENDING``；``approve()``
    才真正 import factory，并 mount 或 state-preserving hot-swap。
    """

    def __init__(
        self,
        *,
        root: str | Path,
        state_root: str | Path,
        host: StableLangGraphPluginHost,
    ) -> None:
        self.root = Path(root).resolve()
        self.state_root = Path(state_root).resolve()
        self.host = host
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, LocalCapabilityRecord] = {}
        self._module_names: dict[str, str] = {}

    def records(self) -> tuple[LocalCapabilityRecord, ...]:
        """按 capability id 返回当前已发现 package 状态。"""

        return tuple(self._records[item] for item in sorted(self._records))

    def get(self, capability_id: str) -> LocalCapabilityRecord:
        try:
            return self._records[capability_id]
        except KeyError as exc:
            raise KeyError(f"unknown local capability: {capability_id}") from exc

    def scan(self) -> tuple[LocalCapabilityEvent, ...]:
        """扫描一次目录；只读 descriptor，不 import/执行第三方代码。

        调用方应在无并发 capability invocation 的安全点调用本方法。若一个已挂载
        package 被用户从目录移走，本次扫描会执行 detach；内容变化只产生待批准更新。
        """

        discovered: dict[str, tuple[LocalCapabilityManifest, Path, str]] = {}
        events: list[LocalCapabilityEvent] = []
        for package_dir in sorted(
            (item for item in self.root.iterdir() if item.is_dir()),
            key=lambda item: item.name.casefold(),
        ):
            try:
                manifest_path = self._find_manifest(package_dir)
                if manifest_path is None:
                    continue
                manifest = self._read_manifest(manifest_path)
                fingerprint = self._package_fingerprint(package_dir)
            except Exception as exc:
                events.append(
                    LocalCapabilityEvent(
                        event_type=LocalCapabilityEventType.INVALID,
                        capability_id=None,
                        package_dir=package_dir,
                        message=str(exc),
                    )
                )
                continue
            if manifest.capability_id in discovered:
                events.append(
                    LocalCapabilityEvent(
                        event_type=LocalCapabilityEventType.INVALID,
                        capability_id=manifest.capability_id,
                        package_dir=package_dir,
                        message=f"duplicate capability id: {manifest.capability_id}",
                    )
                )
                continue
            discovered[manifest.capability_id] = (manifest, package_dir.resolve(), fingerprint)

        previous_ids = set(self._records)
        current_ids = set(discovered)

        for capability_id in sorted(previous_ids - current_ids):
            record = self._records.pop(capability_id)
            if capability_id in self.host.registry.mounted_ids():
                self.host.unmount(capability_id)
            self._discard_module(capability_id)
            events.append(
                LocalCapabilityEvent(
                    event_type=LocalCapabilityEventType.REMOVED,
                    capability_id=capability_id,
                    package_dir=record.package_dir,
                    message=f"capability removed and detached: {capability_id}",
                )
            )

        for capability_id in sorted(current_ids):
            manifest, package_dir, fingerprint = discovered[capability_id]
            record = self._records.get(capability_id)
            if record is None:
                self._records[capability_id] = LocalCapabilityRecord(
                    manifest=manifest,
                    package_dir=package_dir,
                    fingerprint=fingerprint,
                    status=LocalCapabilityStatus.PENDING_APPROVAL,
                )
                events.append(
                    LocalCapabilityEvent(
                        event_type=LocalCapabilityEventType.DISCOVERED,
                        capability_id=capability_id,
                        package_dir=package_dir,
                        message=f"new capability discovered: {capability_id} {manifest.version}",
                    )
                )
                continue

            if record.package_dir != package_dir:
                record.last_error = "capability package path changed unexpectedly"
                record.status = LocalCapabilityStatus.FAILED
                events.append(
                    LocalCapabilityEvent(
                        event_type=LocalCapabilityEventType.INVALID,
                        capability_id=capability_id,
                        package_dir=package_dir,
                        message=record.last_error,
                    )
                )
                continue

            if record.fingerprint != fingerprint:
                record.manifest = manifest
                record.fingerprint = fingerprint
                record.last_error = None
                record.status = (
                    LocalCapabilityStatus.UPDATE_PENDING
                    if capability_id in self.host.registry.mounted_ids()
                    else LocalCapabilityStatus.PENDING_APPROVAL
                )
                events.append(
                    LocalCapabilityEvent(
                        event_type=LocalCapabilityEventType.UPDATED,
                        capability_id=capability_id,
                        package_dir=package_dir,
                        message=f"capability package changed: {capability_id} {manifest.version}",
                    )
                )

        return tuple(events)

    def approve(
        self,
        capability_id: str,
        *,
        validator: CandidateValidator | None = None,
    ) -> PluginHotSwapResult | None:
        """显式批准一个新 package 或待更新 package。

        首次批准执行 mount；已挂载 capability 的 package 更新执行现有的
        state-preserving hot swap。任何 factory/validator 异常都不会把失败候选提交为活动
        capability；记录会进入 ``FAILED`` 并保留错误供 UI 展示。
        """

        record = self.get(capability_id)
        if record.status not in {
            LocalCapabilityStatus.PENDING_APPROVAL,
            LocalCapabilityStatus.UPDATE_PENDING,
            LocalCapabilityStatus.DETACHED,
            LocalCapabilityStatus.FAILED,
        }:
            if record.status == LocalCapabilityStatus.ATTACHED:
                return None
            raise RuntimeError(f"capability cannot be approved from {record.status.value}")

        try:
            mounted = capability_id in self.host.registry.mounted_ids()
            if mounted:
                current = self.host.registry.get(capability_id)
                if (
                    record.active_fingerprint is not None
                    and record.fingerprint != record.active_fingerprint
                    and record.manifest.version == current.version
                ):
                    raise ValueError(
                        "changed capability package must bump executable version before hot update"
                    )
            plugin = self._load_plugin(record)
            self._validate_plugin_identity(record.manifest, plugin)
            if mounted:
                result = self.host.hot_swap(plugin, validator=validator)
            else:
                plugin.mount()
                try:
                    if validator is not None:
                        validator(plugin)
                except Exception:
                    plugin.unmount()
                    raise
                self.host.registry.mount(plugin)
                result = None
        except Exception as exc:
            record.status = LocalCapabilityStatus.FAILED
            record.last_error = str(exc)
            raise

        record.status = LocalCapabilityStatus.ATTACHED
        record.active_fingerprint = record.fingerprint
        record.last_error = None
        return result

    def detach(self, capability_id: str) -> None:
        """显式热拔一个已挂载 capability；package 文件仍可留在目录中。"""

        record = self.get(capability_id)
        if capability_id in self.host.registry.mounted_ids():
            self.host.unmount(capability_id)
        self._discard_module(capability_id)
        record.status = LocalCapabilityStatus.DETACHED
        record.last_error = None

    @staticmethod
    def _find_manifest(package_dir: Path) -> Path | None:
        matches = [package_dir / name for name in _MANIFEST_NAMES if (package_dir / name).is_file()]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(f"multiple capability manifests in {package_dir}")
        return matches[0]

    @staticmethod
    def _read_manifest(path: Path) -> LocalCapabilityManifest:
        if path.name == "capability.toml":
            with path.open("rb") as handle:
                payload = tomllib.load(handle)
        elif path.name == "capability.json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"unsupported capability manifest: {path.name}")
        if payload.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported local capability schema")
        return LocalCapabilityManifest(
            capability_id=str(payload["id"]),
            version=str(payload["version"]),
            summary=str(payload["summary"]),
            entrypoint=str(payload["entrypoint"]),
            permissions=tuple(payload.get("permissions") or ()),
            metadata=dict(payload.get("metadata") or {}),
        )

    @staticmethod
    def _package_fingerprint(package_dir: Path) -> str:
        digest = hashlib.sha256()
        files = sorted(
            (
                path
                for path in package_dir.rglob("*")
                if path.is_file() and not any(part in _IGNORED_PARTS for part in path.parts)
            ),
            key=lambda path: path.relative_to(package_dir).as_posix(),
        )
        for path in files:
            relative = path.relative_to(package_dir).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def _load_plugin(self, record: LocalCapabilityRecord) -> StatefulCapabilityExecutable:
        module_ref, factory_name = record.manifest.entrypoint.rsplit(":", 1)
        module_path = self._resolve_module_path(record.package_dir, module_ref)
        module_name = self._module_name(record.manifest, record.fingerprint)
        self._discard_module(record.manifest.capability_id)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load capability module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            self._exec_module_with_package_path(module, spec.loader.exec_module, record.package_dir)
            factory = getattr(module, factory_name)
            if not callable(factory):
                raise TypeError(f"capability factory is not callable: {factory_name}")
            context = LocalCapabilityPackageContext(
                package_dir=record.package_dir,
                state_root=self.state_root / record.manifest.capability_id,
                manifest=record.manifest,
            )
            plugin = factory(context)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        self._module_names[record.manifest.capability_id] = module_name
        return plugin

    @staticmethod
    def _exec_module_with_package_path(
        module: ModuleType,
        exec_module: Callable[[ModuleType], None],
        package_dir: Path,
    ) -> None:
        package_text = str(package_dir)
        sys.path.insert(0, package_text)
        try:
            exec_module(module)
        finally:
            try:
                sys.path.remove(package_text)
            except ValueError:
                pass

    @staticmethod
    def _resolve_module_path(package_dir: Path, module_ref: str) -> Path:
        raw = module_ref.strip()
        if not raw:
            raise ValueError("capability entrypoint module is required")
        if raw.endswith(".py") or "/" in raw or "\\" in raw:
            relative = Path(raw)
        else:
            relative = Path(*raw.split(".")).with_suffix(".py")
        candidate = (package_dir / relative).resolve()
        try:
            candidate.relative_to(package_dir.resolve())
        except ValueError as exc:
            raise ValueError("capability entrypoint escapes package directory") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"capability entrypoint not found: {relative.as_posix()}")
        return candidate

    @staticmethod
    def _module_name(manifest: LocalCapabilityManifest, fingerprint: str) -> str:
        safe_id = "".join(char if char.isalnum() else "_" for char in manifest.capability_id)
        return f"orion_local_capability_{safe_id}_{fingerprint[:12]}"

    @staticmethod
    def _validate_plugin_identity(
        manifest: LocalCapabilityManifest,
        plugin: StatefulCapabilityExecutable,
    ) -> None:
        if plugin.plugin_id != manifest.capability_id:
            raise ValueError(
                "capability factory id mismatch: "
                f"manifest={manifest.capability_id}, plugin={plugin.plugin_id}"
            )
        if plugin.version != manifest.version:
            raise ValueError(
                "capability factory version mismatch: "
                f"manifest={manifest.version}, plugin={plugin.version}"
            )

    def _discard_module(self, capability_id: str) -> None:
        module_name = self._module_names.pop(capability_id, None)
        if module_name is not None:
            sys.modules.pop(module_name, None)
