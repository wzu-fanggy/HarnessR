from __future__ import annotations

import pytest

from orion.capvm import (
    CapabilityCatalog,
    CapabilityDependency,
    CapabilityKind,
    CapabilityResidencyGovernor,
    MeasuredResidencyBudgetExceeded,
    ResidencyBudgetExceeded,
    ResidencyLevel,
    ResourceVector,
    VirtualCapability,
)
from orion.capvm.adapters import (
    ProcessCapabilityAdapter,
    SessionCapabilityAdapter,
    SkillCapabilityAdapter,
    ToolCapabilityAdapter,
)


def _budget(
    *,
    prompt_tokens: float = 10_000,
    host_ram_mb: float = 10_000,
    process_slots: float = 100,
    connection_slots: float = 100,
) -> ResourceVector:
    return ResourceVector(
        prompt_tokens=prompt_tokens,
        host_ram_mb=host_ram_mb,
        process_slots=process_slots,
        connection_slots=connection_slots,
    )


def _tool(capability_id: str, *, pinned: ResidencyLevel = ResidencyLevel.CATALOG) -> VirtualCapability:
    return VirtualCapability(
        capability_id=capability_id,
        kind=CapabilityKind.TOOL,
        summary=f"tool {capability_id}",
        adapter=ToolCapabilityAdapter(
            interface_text="i" * 80,
            reasoning_text="r" * 120,
        ),
        pinned_min_level=pinned,
    )


class UnderdeclaredAdapter:
    """用于验证 page-in 后 telemetry reconciliation 的确定性 adapter。"""

    def __init__(self, *, declared_mb: float, measured_mb: float) -> None:
        self.declared_mb = declared_mb
        self.measured_mb = measured_mb
        self.resident = False

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level >= ResidencyLevel.BOUND:
            return ResourceVector(host_ram_mb=self.declared_mb)
        return ResourceVector()

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level >= ResidencyLevel.BOUND and self.resident:
            return ResourceVector(host_ram_mb=self.measured_mb)
        return ResourceVector()

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current < ResidencyLevel.BOUND <= target:
            self.resident = True

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current >= ResidencyLevel.BOUND > target:
            self.resident = False

    def close(self) -> None:
        self.resident = False


def _underdeclared(capability_id: str, *, declared_mb: float, measured_mb: float) -> VirtualCapability:
    return VirtualCapability(
        capability_id=capability_id,
        kind=CapabilityKind.PROCESS,
        summary=f"underdeclared {capability_id}",
        adapter=UnderdeclaredAdapter(declared_mb=declared_mb, measured_mb=measured_mb),
        maximum_level=ResidencyLevel.BOUND,
    )


def test_measured_reconciliation_evicts_inactive_resident_after_declared_fit() -> None:
    catalog = CapabilityCatalog(
        [
            _underdeclared("older", declared_mb=1.0, measured_mb=3.0),
            _underdeclared("newer", declared_mb=1.0, measured_mb=3.0),
        ]
    )
    governor = CapabilityResidencyGovernor(
        catalog=catalog,
        budget=_budget(host_ram_mb=4.0),
    )
    try:
        governor.access("older", ResidencyLevel.BOUND, request_id="older")
        governor.access("newer", ResidencyLevel.BOUND, request_id="newer")
        assert governor.total_declared_cost().host_ram_mb == 2.0
        assert governor.total_measured_cost().host_ram_mb == 6.0

        reconciled = governor.reconcile_measured_budget(protected_ids={"newer"})

        assert reconciled.host_ram_mb == 3.0
        assert governor.state("older").current_level == ResidencyLevel.CATALOG
        assert governor.state("newer").current_level == ResidencyLevel.BOUND
        assert any(
            event.event_type == "demote" and event.detail == "measured_budget_eviction"
            for event in governor.events
        )
    finally:
        governor.close()


def test_measured_reconciliation_rejects_active_working_set_that_really_exceeds_budget() -> None:
    capability = _underdeclared("active-heavy", declared_mb=1.0, measured_mb=5.0)
    catalog = CapabilityCatalog([capability])
    governor = CapabilityResidencyGovernor(
        catalog=catalog,
        budget=_budget(host_ram_mb=4.0),
    )
    try:
        with governor.acquire("active-heavy", ResidencyLevel.BOUND):
            assert governor.total_declared_cost().host_ram_mb == 1.0
            assert governor.total_measured_cost().host_ram_mb == 5.0
            with pytest.raises(MeasuredResidencyBudgetExceeded) as exc:
                governor.reconcile_measured_budget()
            assert exc.value.violated_dimensions == ("host_ram_mb",)
        assert any(
            event.event_type == "measured_budget_overrun"
            for event in governor.events
        )
    finally:
        governor.close()


def test_logical_addressability_survives_eviction_and_refault() -> None:
    catalog = CapabilityCatalog([_tool("tool-a"), _tool("tool-b")])
    governor = CapabilityResidencyGovernor(
        catalog=catalog,
        budget=_budget(prompt_tokens=60),
    )
    try:
        assert catalog.get("tool-a").capability_id == "tool-a"
        assert governor.state("tool-a").current_level == ResidencyLevel.CATALOG

        governor.access("tool-a", ResidencyLevel.REASONING, request_id="first-a")
        assert governor.state("tool-a").current_level == ResidencyLevel.REASONING
        assert governor.state("tool-a").compulsory_faults == 1

        governor.access("tool-b", ResidencyLevel.REASONING, request_id="first-b")
        assert governor.state("tool-a").current_level == ResidencyLevel.CATALOG
        assert governor.state("tool-b").current_level == ResidencyLevel.REASONING
        # 淘汰不删除逻辑 capability。
        assert catalog.get("tool-a").summary == "tool tool-a"

        governor.access("tool-a", ResidencyLevel.REASONING, request_id="refault-a")
        assert governor.state("tool-a").current_level == ResidencyLevel.REASONING
        assert governor.state("tool-a").refaults == 1
        assert governor.state("tool-b").current_level == ResidencyLevel.CATALOG
    finally:
        governor.close()


def test_pinned_capability_cannot_be_silently_evicted() -> None:
    catalog = CapabilityCatalog(
        [
            _tool("pinned", pinned=ResidencyLevel.REASONING),
            _tool("other"),
        ]
    )
    governor = CapabilityResidencyGovernor(
        catalog=catalog,
        budget=_budget(prompt_tokens=60),
    )
    try:
        assert governor.state("pinned").current_level == ResidencyLevel.REASONING
        with pytest.raises(ResidencyBudgetExceeded):
            governor.access("other", ResidencyLevel.REASONING)
        assert governor.state("pinned").current_level == ResidencyLevel.REASONING
        assert governor.state("other").current_level == ResidencyLevel.CATALOG
    finally:
        governor.close()


def test_cached_root_reloads_dependency_evicted_under_later_pressure() -> None:
    dependency = _underdeclared("dependency", declared_mb=4.0, measured_mb=4.0)
    parent = VirtualCapability(
        capability_id="parent",
        kind=CapabilityKind.TOOL,
        summary="parent tool",
        adapter=ToolCapabilityAdapter(interface_text="parent()"),
        dependencies=(
            CapabilityDependency("dependency", ResidencyLevel.BOUND),
        ),
        maximum_level=ResidencyLevel.INTERFACE,
    )
    competitor = _underdeclared("competitor", declared_mb=4.0, measured_mb=4.0)
    catalog = CapabilityCatalog([dependency, parent, competitor])
    governor = CapabilityResidencyGovernor(
        catalog=catalog,
        budget=_budget(host_ram_mb=4.0),
    )
    try:
        with governor.acquire("parent", ResidencyLevel.INTERFACE):
            assert governor.state("parent").current_level == ResidencyLevel.INTERFACE
            assert governor.state("dependency").current_level == ResidencyLevel.BOUND

        # 根 Tool 仍作为空闲接口缓存，但它的重型依赖可以被后续 RAM 压力回收。
        governor.access("competitor", ResidencyLevel.BOUND)
        assert governor.state("parent").current_level == ResidencyLevel.INTERFACE
        assert governor.state("dependency").current_level == ResidencyLevel.CATALOG
        assert governor.state("competitor").current_level == ResidencyLevel.BOUND

        # 再次调用已经缓存的根 Tool 时，runtime 必须先恢复被换出的依赖。
        with governor.acquire("parent", ResidencyLevel.INTERFACE):
            assert governor.state("parent").current_level == ResidencyLevel.INTERFACE
            assert governor.state("dependency").current_level == ResidencyLevel.BOUND
            assert governor.state("competitor").current_level == ResidencyLevel.CATALOG
    finally:
        governor.close()


def test_active_capability_and_dependencies_are_eviction_protected() -> None:
    dependency = _tool("dependency")
    parent = VirtualCapability(
        capability_id="parent",
        kind=CapabilityKind.SKILL,
        summary="parent skill",
        adapter=SkillCapabilityAdapter(
            index_text="p" * 40,
            instruction_text="q" * 80,
        ),
        dependencies=(
            CapabilityDependency("dependency", ResidencyLevel.INTERFACE),
        ),
    )
    competitor = _tool("competitor")
    catalog = CapabilityCatalog([dependency, parent, competitor])
    # dependency R1 = 20 tokens; parent R2 = 30 tokens，总计 50；competitor R2 = 50。
    governor = CapabilityResidencyGovernor(
        catalog=catalog,
        budget=_budget(prompt_tokens=55),
    )
    try:
        with governor.acquire("parent", ResidencyLevel.REASONING):
            assert governor.state("parent").active_count == 1
            assert governor.state("dependency").active_count == 1
            with pytest.raises(ResidencyBudgetExceeded):
                governor.access("competitor", ResidencyLevel.REASONING)
        assert governor.state("parent").active_count == 0
        assert governor.state("dependency").active_count == 0
    finally:
        governor.close()


def test_indirect_dependency_use_refreshes_shared_dependency_recency() -> None:
    shared = _tool("shared")
    parent_a = VirtualCapability(
        capability_id="parent-a",
        kind=CapabilityKind.SKILL,
        summary="parent a",
        adapter=SkillCapabilityAdapter(index_text="a" * 40, instruction_text="b" * 80),
        dependencies=(CapabilityDependency("shared", ResidencyLevel.INTERFACE),),
    )
    parent_b = VirtualCapability(
        capability_id="parent-b",
        kind=CapabilityKind.SKILL,
        summary="parent b",
        adapter=SkillCapabilityAdapter(index_text="c" * 40, instruction_text="d" * 80),
        dependencies=(CapabilityDependency("shared", ResidencyLevel.INTERFACE),),
    )
    governor = CapabilityResidencyGovernor(
        catalog=CapabilityCatalog([shared, parent_a, parent_b]),
        budget=_budget(),
    )
    try:
        governor.access("parent-a", ResidencyLevel.INTERFACE)
        first_tick = governor.state("shared").last_used_tick
        assert first_tick == governor.state("parent-a").last_used_tick
        assert first_tick > 0

        governor.access("parent-b", ResidencyLevel.INTERFACE)
        assert governor.state("shared").last_used_tick == governor.state("parent-b").last_used_tick
        assert governor.state("shared").last_used_tick > first_tick
    finally:
        governor.close()


def test_process_declared_cost_can_include_runtime_overhead_separately_from_payload() -> None:
    adapter = ProcessCapabilityAdapter(
        resident_mb=3,
        runtime_overhead_mb=12.5,
        startup_delay_s=0.0,
    )
    declared = adapter.declared_cost(ResidencyLevel.BOUND)

    assert adapter.resident_mb == 3
    assert adapter.runtime_overhead_mb == 12.5
    assert declared.host_ram_mb == pytest.approx(15.5)
    assert declared.process_slots == 1.0
    assert adapter.is_running is False


def test_process_measured_rss_tracks_payload_growth_and_close_releases_worker() -> None:
    small = ProcessCapabilityAdapter(resident_mb=3, startup_delay_s=0.0)
    large = ProcessCapabilityAdapter(resident_mb=32, startup_delay_s=0.0)
    try:
        small.promote(ResidencyLevel.CATALOG, ResidencyLevel.BOUND)
        small_pid = small.pid
        assert small_pid is not None
        small_rss = small.measured_cost(ResidencyLevel.BOUND).host_ram_mb
        small.close()
        assert small.pid is None

        large.promote(ResidencyLevel.CATALOG, ResidencyLevel.BOUND)
        large_pid = large.pid
        assert large_pid is not None
        large_rss = large.measured_cost(ResidencyLevel.BOUND).host_ram_mb

        # Worker 会逐页触碰 payload；真实 allocation-bearing PID 的 RSS 应随 3 -> 32 MB
        # 明显增长。这个回归专门防止 Windows venv redirector PID 被误当成 worker PID。
        assert large_rss > small_rss + 20.0
    finally:
        small.close()
        large.close()
        assert large.pid is None


def test_process_measured_rss_unavailable_fails_closed(monkeypatch) -> None:
    adapter = ProcessCapabilityAdapter(resident_mb=3, startup_delay_s=0.0)
    try:
        adapter.promote(ResidencyLevel.CATALOG, ResidencyLevel.BOUND)
        monkeypatch.setattr("orion.capvm.adapters.process.process_rss_mb", lambda _pid: 0.0)
        with pytest.raises(RuntimeError, match="WorkingSet 测量不可用"):
            adapter.measured_cost(ResidencyLevel.BOUND)
    finally:
        adapter.close()


def test_process_measurement_failure_blocks_admission_and_cleans_worker(monkeypatch) -> None:
    adapter = ProcessCapabilityAdapter(resident_mb=3, startup_delay_s=0.0)
    capability = VirtualCapability(
        capability_id="unmeasurable-worker",
        kind=CapabilityKind.PROCESS,
        summary="worker whose RSS is unavailable",
        adapter=adapter,
    )
    governor = CapabilityResidencyGovernor(
        catalog=CapabilityCatalog([capability]),
        budget=_budget(host_ram_mb=64, process_slots=1),
    )
    monkeypatch.setattr("orion.capvm.adapters.process.process_rss_mb", lambda _pid: 0.0)
    try:
        with pytest.raises(RuntimeError, match="WorkingSet 测量不可用"):
            with governor.acquire_many(
                [("unmeasurable-worker", ResidencyLevel.BOUND)],
                reconcile_before_active=True,
            ):
                pytest.fail("占用不可测的 worker 不应进入调用")
        assert adapter.is_running is False
    finally:
        governor.close()


def test_process_capability_reports_real_process_residency() -> None:
    adapter = ProcessCapabilityAdapter(resident_mb=6, startup_delay_s=0.01)
    capability = VirtualCapability(
        capability_id="worker",
        kind=CapabilityKind.PROCESS,
        summary="resident worker",
        adapter=adapter,
    )
    governor = CapabilityResidencyGovernor(
        catalog=CapabilityCatalog([capability]),
        budget=_budget(host_ram_mb=64, process_slots=1),
    )
    try:
        governor.access("worker", ResidencyLevel.BOUND)
        assert adapter.pid is not None
        measured = governor.total_measured_cost()
        assert measured.process_slots == 1
        assert measured.host_ram_mb > 0

        governor.demote("worker", ResidencyLevel.CATALOG)
        assert adapter.pid is None
        assert governor.total_measured_cost().process_slots == 0
    finally:
        governor.close()


def test_session_eviction_checkpoints_and_restores_state() -> None:
    first_adapter = SessionCapabilityAdapter(resident_mb=1.0)
    second_adapter = SessionCapabilityAdapter(resident_mb=1.0)
    catalog = CapabilityCatalog(
        [
            VirtualCapability(
                capability_id="session-a",
                kind=CapabilityKind.SESSION,
                summary="session a",
                adapter=first_adapter,
            ),
            VirtualCapability(
                capability_id="session-b",
                kind=CapabilityKind.SESSION,
                summary="session b",
                adapter=second_adapter,
            ),
        ]
    )
    governor = CapabilityResidencyGovernor(
        catalog=catalog,
        budget=_budget(host_ram_mb=1.5, connection_slots=1),
    )
    try:
        governor.access("session-a", ResidencyLevel.BOUND)
        first_adapter.write_state("iteration", 7)
        assert first_adapter.is_connected
        assert governor.total_measured_cost().host_ram_mb == pytest.approx(1.0, abs=0.01)

        governor.access("session-b", ResidencyLevel.BOUND)
        assert governor.state("session-a").current_level == ResidencyLevel.CATALOG
        assert first_adapter.checkpoint["iteration"] == 7
        assert not first_adapter.is_connected

        governor.access("session-a", ResidencyLevel.BOUND)
        assert first_adapter.is_connected
        assert first_adapter.read_state("iteration") == 7
        assert governor.state("session-a").refaults == 1
    finally:
        governor.close()


def test_acquire_many_rejects_a_working_set_that_only_fits_sequentially() -> None:
    catalog = CapabilityCatalog([_tool("tool-a"), _tool("tool-b")])
    governor = CapabilityResidencyGovernor(
        catalog=catalog,
        budget=_budget(prompt_tokens=60),
    )
    try:
        # 每个 tool 单独约 50 tokens，可顺序装入；但一个 phase 同时要求两者时必须失败，
        # 不能通过淘汰 phase 前半部分制造虚假成功。
        with pytest.raises(ResidencyBudgetExceeded):
            with governor.acquire_many(
                (
                    ("tool-a", ResidencyLevel.REASONING),
                    ("tool-b", ResidencyLevel.REASONING),
                ),
                request_id="joint-phase",
            ):
                pass
    finally:
        governor.close()


def test_one_governor_evicts_multiple_capability_kinds() -> None:
    process_a = ProcessCapabilityAdapter(resident_mb=5)
    process_b = ProcessCapabilityAdapter(resident_mb=5)
    session_a = SessionCapabilityAdapter()
    session_b = SessionCapabilityAdapter()
    catalog = CapabilityCatalog(
        [
            _tool("tool-a"),
            VirtualCapability(
                capability_id="skill-a",
                kind=CapabilityKind.SKILL,
                summary="skill a",
                adapter=SkillCapabilityAdapter(
                    index_text="s" * 80,
                    instruction_text="k" * 120,
                ),
            ),
            VirtualCapability(
                capability_id="process-a",
                kind=CapabilityKind.PROCESS,
                summary="process a",
                adapter=process_a,
            ),
            VirtualCapability(
                capability_id="process-b",
                kind=CapabilityKind.PROCESS,
                summary="process b",
                adapter=process_b,
            ),
            VirtualCapability(
                capability_id="session-a",
                kind=CapabilityKind.SESSION,
                summary="session a",
                adapter=session_a,
            ),
            VirtualCapability(
                capability_id="session-b",
                kind=CapabilityKind.SESSION,
                summary="session b",
                adapter=session_b,
            ),
        ]
    )
    governor = CapabilityResidencyGovernor(
        catalog=catalog,
        budget=_budget(
            prompt_tokens=60,
            host_ram_mb=8,
            process_slots=1,
            connection_slots=1,
        ),
    )
    try:
        governor.access("tool-a", ResidencyLevel.REASONING)
        governor.access("skill-a", ResidencyLevel.REASONING)
        assert governor.state("tool-a").current_level == ResidencyLevel.CATALOG

        governor.access("process-a", ResidencyLevel.BOUND)
        governor.access("process-b", ResidencyLevel.BOUND)
        assert governor.state("process-a").current_level == ResidencyLevel.CATALOG

        governor.access("session-a", ResidencyLevel.BOUND)
        governor.access("session-b", ResidencyLevel.BOUND)
        assert governor.state("session-a").current_level == ResidencyLevel.CATALOG

        demoted_ids = {
            event.capability_id
            for event in governor.events
            if event.event_type == "demote" and event.detail == "budget_eviction"
        }
        demoted_kinds = {catalog.get(item).kind for item in demoted_ids}
        assert CapabilityKind.TOOL in demoted_kinds
        assert CapabilityKind.PROCESS in demoted_kinds
        assert CapabilityKind.SESSION in demoted_kinds
        # Skill 也由同一 governor 管理，并保持逻辑可寻址。
        assert catalog.get("skill-a").kind == CapabilityKind.SKILL
    finally:
        governor.close()
