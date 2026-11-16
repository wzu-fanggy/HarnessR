"""V17 正式回收机制对独立简化全局 LRU 的同底层执行对照。

两个方案均使用真实 frozen builder 提供的同一类插件缓冲区/进程/会话 adapter、
同一 CapacityGovernor 准入与实测核查；仅替换候选排序 policy。
这是一项**选择策略消融**而非与独立缓存产品的完整系统效率比较。
对本脚本读取的每项输入记录 SHA-256，保持原始论文、模型和 benchmark 不改。
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT.parent), str(ROOT)]

from orion.benchmark import cwsr_provider_token_replacement_replication_v1 as calibration
from orion.benchmark import cwsr_retention_corrected_v2 as frozen
from orion.benchmark import cwsr_runtime_only_retention_v1 as runtime_only
from orion.benchmark import hcvm_shared_pool_vs_per_kind_v2 as inventory
from orion.benchmark.hcvm_global_admission_replacement_order_sensitivity_v1 import _orders
from orion.benchmark.evolvable_plugin_mixed_capability_runtime_v9r1_dev import build_mixed_inventory
from orion.benchmark.evolvable_plugin_mixed_capability_runtime_v9r2_dev import PROCESS_RUNTIME_OVERHEAD_MB
from orion.capvm import (
    CapabilityResidencyGovernor,
    MeasuredResidencyBudgetExceeded,
    ResidencyBudgetExceeded,
    ResourceVector,
)

OUT_DIR = ROOT / "reports" / "cwsr_strong_lru_20261010"
SOURCES = (
    "scripts/agent/cwsr_v17_minimal_lru_vs_runtime_v1.py",
    "benchmark/cwsr_retention_corrected_v2.py",
    "benchmark/cwsr_runtime_only_retention_v1.py",
    "benchmark/cwsr_retention_robust_window_v2.py",
    "benchmark/hcvm_shared_pool_vs_per_kind_v2.py",
    "benchmark/hcvm_global_admission_replacement_order_sensitivity_v1.py",
    "capvm/governor.py",
    "capvm/policies/lru_cost.py",
    "capvm/adapters/process.py",
)


class IndependentMinimalGlobalLRU:
    """一项极简容量感知 LRU；不导入/复用生产默认 LRU 的 choose 实现。

    只逐一读取当前闲置对象的释放占用，允许跨类别淘汰，
    按 last-used 以及资源 ID 取最久未用对象。
    """
    def choose(self, *, states, catalog, violated_dimensions, protected_ids, cost_basis="declared"):
        if cost_basis not in ("declared", "measured"):
            raise ValueError("unsupported cost basis")
        possible = []
        for name, state in states.items():
            resource = catalog.get(name)
            floor = resource.pinned_min_level
            if (name in protected_ids or state.active_count != 0
                    or state.current_level <= floor):
                continue
            cost = resource.adapter.declared_cost if cost_basis=="declared" else resource.adapter.measured_cost
            release = cost(state.current_level)-cost(floor)
            if not any(release.value(axis)>0 for axis in violated_dimensions):
                continue
            possible.append((state.last_used_tick, name, floor))
        if not possible:
            return None
        oldest = min(possible, key=lambda row: (row[0], row[1]))
        return oldest[1], oldest[2]


def measured_simple_case(phases, budget):
    """原始 frozen.run_case 的调用路径，差异仅为独立简化 LRU 的候选选择。"""
    with tempfile.TemporaryDirectory(prefix="cwsr-independent-lru-") as tmp:
        catalog, _, _ = build_mixed_inventory(
            state_root=Path(tmp),
            process_runtime_overhead_mb=PROCESS_RUNTIME_OVERHEAD_MB,
        )
        g = CapabilityResidencyGovernor(
            catalog=catalog, budget=budget, policy=IndependentMinimalGlobalLRU(),
        )
        flags, failures, peak = [], [], ResourceVector()
        try:
            for index, phase in enumerate(phases):
                try:
                    with g.acquire_many(
                        inventory._phase_requests(catalog, phase["capability_ids"]),
                        request_id=str(phase["phase_id"]),
                        reconcile_before_active=True,
                    ):
                        measured = g.total_measured_cost()
                        if not measured.fits(budget):
                            raise AssertionError("budget exceeded after preparation")
                        peak = ResourceVector(**{
                            key:max(getattr(peak,key),getattr(measured,key))
                            for key in asdict(peak)
                        })
                    flags.append("1")
                except (ResidencyBudgetExceeded, MeasuredResidencyBudgetExceeded) as ex:
                    flags.append("0")
                    failures.append([index,type(ex).__name__,list(ex.violated_dimensions)])

            prep = Counter()
            initial = Counter()
            for state in g.states():
                category = catalog.get(state.capability_id).kind.value
                if state.promotions:
                    initial[category]+=1
                    prep[category]+=max(0,state.promotions-1)
            return {
                "ready":flags.count("1"),
                "requested":len(flags),
                "ready_flags":"".join(flags),
                "repreparations":sum(prep.values()),
                "repreparations_by_kind":dict(sorted(prep.items())),
                "initial_preparations":sum(initial.values()),
                "failures":failures,
                "peak_measured":asdict(peak),
            }
        finally:
            g.close()


def summary(row):
    return {
        "ready":row["ready"],
        "requested":row["requested"],
        "flags":row["ready_flags"],
        "runtime_rebuilds":runtime_only._runtime_only(row["repreparations_by_kind"]),
        "all_rebuilds":row["repreparations"],
        "by_kind":row["repreparations_by_kind"],
        "initial":row["initial_preparations"],
        "failures":row["failures"],
        "measured_peak_ram_mib":row["peak_measured"]["host_ram_mb"],
    }


def run_suite(suite, hosts, orders_limit):
    source_hash={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    before_phases = [n for n,_ in _orders()]
    names=list(_orders())
    if suite=="quick":
        names=names[:1]
    elif suite=="capacity":
        names=names[:3]
    elif suite=="orders":
        names=names[:orders_limit]
    else:
        raise ValueError("unknown suite")
    records=[]
    old, budget_old = calibration._patch_token_estimator()
    try:
        for host in hosts:
            budget = runtime_only._budget(runtime_only.RELAXED_TOKENS,host,1,1)
            for ix,(order_name, ordered) in enumerate(names):
                phases=runtime_only._select_phases(ordered,True)
                if len(phases)!=20:
                    raise AssertionError("expected 20 managed-resource requests")
                strategies=("runtime","simple_lru") if ix%2 else ("simple_lru","runtime")
                arms={}
                for strategy in strategies:
                    data = (
                        frozen.run_case("cross_type",phases,budget)
                        if strategy=="runtime" else measured_simple_case(phases,budget)
                    )
                    arms[strategy]=summary(data)
                a,b=arms["runtime"],arms["simple_lru"]
                records.append({
                    "host_budget_mib":host,"order":order_name,"strategy_start_order":list(strategies),
                    "arms":arms,
                    "matching_readiness_flags":a["flags"]==b["flags"],
                    "matching_rebuilds":a["runtime_rebuilds"]==b["runtime_rebuilds"],
                    "matching_rebuild_by_kind":a["by_kind"]==b["by_kind"],
                    "matching_failure_classes":a["failures"]==b["failures"],
                })
    finally:
        calibration._restore(old,budget_old)
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h for name,h in source_hash.items()):
        raise RuntimeError("frozen input changed during trial")
    by_budget=[]
    for h in hosts:
        subset=[r for r in records if r["host_budget_mib"]==h]
        by_budget.append({
            "budget_mib":h,"orders":len(subset),
            "ready_all_both":sum(r["arms"]["runtime"]["ready"]==20 and r["arms"]["simple_lru"]["ready"]==20 for r in subset),
            "same_readiness":sum(r["matching_readiness_flags"] for r in subset),
            "same_rebuilds":sum(r["matching_rebuilds"] for r in subset),
            "same_by_kind":sum(r["matching_rebuild_by_kind"] for r in subset),
            "median_runtime_rebuilds":median(r["arms"]["runtime"]["runtime_rebuilds"] for r in subset),
            "median_simple_lru_rebuilds":median(r["arms"]["simple_lru"]["runtime_rebuilds"] for r in subset),
            "runtime_better_rebuilds":sum(r["arms"]["runtime"]["runtime_rebuilds"] < r["arms"]["simple_lru"]["runtime_rebuilds"] for r in subset),
            "simple_lru_better_rebuilds":sum(r["arms"]["runtime"]["runtime_rebuilds"] > r["arms"]["simple_lru"]["runtime_rebuilds"] for r in subset),
        })
    return {
        "schema":"cwsr-v17-minimal-global-budget-lru-v1",
        "suite":suite,
        "run_utc":datetime.now(timezone.utc).isoformat(),
        "purpose":"independent minimal LRU policy vs V17 policy in the same governor/adapters; not an independent caching system comparison",
        "real_process_adapters":True,"uses_agent_llm":False,
        "software_budget_not_os_hard_limit":True,
        "source_sha256":source_hash,"sources_unchanged":True,
        "all_27_order_names":before_phases,
        "by_budget":by_budget,
        "runs":records,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--suite",choices=["quick","capacity","orders"],required=True)
    parser.add_argument("--hosts",default="21.5")
    parser.add_argument("--orders-limit",type=int,default=27)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():
        parser.error("refuse overwriting")
    hosts=tuple(float(x) for x in a.hosts.split(","))
    if any(v<=0 for v in hosts) or not 1<=a.orders_limit<=27:
        parser.error("invalid hosts/orders-limit")
    result=run_suite(a.suite,hosts,a.orders_limit)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"suite":a.suite,"by_budget":result["by_budget"],"output":str(a.output)},ensure_ascii=False))


if __name__=="__main__":
    main()
