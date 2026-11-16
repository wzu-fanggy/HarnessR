"""Capability Virtual Memory 的逻辑地址空间。"""

from __future__ import annotations

from collections.abc import Iterable

from .model import (
    CapabilityFault,
    CapabilityRuntimeState,
    ResidencyLevel,
    VirtualCapability,
)


class CapabilityCatalog:
    """保存逻辑 capability 描述符，不把未驻留等同于不存在。"""

    def __init__(self, capabilities: Iterable[VirtualCapability] = ()) -> None:
        self._capabilities: dict[str, VirtualCapability] = {}
        for capability in capabilities:
            self.register(capability)

    def register(self, capability: VirtualCapability) -> None:
        if capability.capability_id in self._capabilities:
            raise ValueError(f"duplicate capability_id: {capability.capability_id}")
        self._capabilities[capability.capability_id] = capability

    def get(self, capability_id: str) -> VirtualCapability:
        try:
            return self._capabilities[capability_id]
        except KeyError as exc:
            raise KeyError(f"unknown capability: {capability_id}") from exc

    def contains(self, capability_id: str) -> bool:
        return capability_id in self._capabilities

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._capabilities))

    def all(self) -> tuple[VirtualCapability, ...]:
        return tuple(self._capabilities[item] for item in sorted(self._capabilities))

    def logical_metadata_bytes(self) -> int:
        """返回常驻 catalog descriptor 的确定性最小字节量。

        该指标不假装等于 Python heap footprint；它用于显式记录 logical address
        space 仍然有 O(N) metadata overhead，避免把“backing residency 有界”误写成
        “capability 数量增加完全不消耗内存”。
        """

        total = 0
        for capability in self._capabilities.values():
            fields = [
                capability.capability_id,
                capability.kind.value,
                capability.summary,
                capability.pinned_min_level.name,
                capability.maximum_level.name,
            ]
            fields.extend(
                f"{item.capability_id}:{item.required_level.name}"
                for item in capability.dependencies
            )
            total += len("|".join(fields).encode("utf-8"))
        return total

    def search(self, query: str, *, limit: int | None = None) -> tuple[VirtualCapability, ...]:
        """只在 catalog metadata 上做确定性检索，不触发 materialization。"""

        terms = tuple(
            item.casefold()
            for item in query.split()
            if item.strip()
        )
        scored: list[tuple[int, str, VirtualCapability]] = []
        for capability in self._capabilities.values():
            haystack = (
                capability.capability_id
                + " "
                + capability.kind.value
                + " "
                + capability.summary
            ).casefold()
            score = sum(term in haystack for term in terms)
            if not terms or score:
                scored.append((-score, capability.capability_id, capability))
        scored.sort(key=lambda item: (item[0], item[1]))
        values = tuple(item[2] for item in scored)
        return values if limit is None else values[:limit]

    def require_resident(
        self,
        capability_id: str,
        state: CapabilityRuntimeState,
        required_level: ResidencyLevel,
        *,
        request_id: str = "",
    ) -> VirtualCapability:
        """要求 capability 已达到驻留层级；不足时抛出可服务的 fault。"""

        capability = self.get(capability_id)
        if required_level == ResidencyLevel.ACTIVE:
            raise ValueError("ACTIVE residency must be entered through governor.acquire()")
        if state.current_level < required_level:
            raise CapabilityFault(
                capability_id=capability_id,
                current_level=state.current_level,
                required_level=required_level,
                request_id=request_id,
            )
        return capability
