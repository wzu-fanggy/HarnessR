from __future__ import annotations



import argparse, hashlib, json

from collections import Counter

from dataclasses import asdict

from pathlib import Path

from time import monotonic



from orion.benchmark.cwsr_electrolyte_workflow_policy_replay_v1 import NoEvictionPolicy

from orion.benchmark.cwsr_real_rag_kg_trace_replay_v1 import (

    ExternalBgeModelAdapter, ROOT_PYTHON, WORKER_SCRIPT, EMBEDDING_MODEL,

)

from orion.capvm import (

    CapabilityCatalog, CapabilityKind, CapabilityResidencyGovernor,

    MeasuredResidencyBudgetExceeded, ResidencyBudgetExceeded,

    ResidencyLevel, ResourceVector, VirtualCapability,

)

from orion.tools.electrolyte_design.conductivity_ml import temperature_c_to_inverse_temperature

from orion.tools.electrolyte_design.conductivity_process import SodiumConductivityMlProcessAdapter



ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "model:bge-m3-real"

PROCESS_ID = "process:na-conductivity-real"

GPU_BUDGET_MIB = 3000.0

PROFILED_BGE_HOST_MIB = 558.13671875

PROFILED_PROCESS_HOST_MIB = 383.4609375

MODEL_PATH = ROOT.parents[1] / 'data/cwsr_conductivity_rf_formal_v1.joblib'



QUERIES = (

    "low temperature sodium-ion battery electrolyte hard carbon -20 C",

    "NaPF6 carbonate electrolyte ionic conductivity low temperature",

    "hard carbon electrolyte interphase sodium-ion battery low temperature",

    "NaPF6 EC EMC PC sodium-ion electrolyte conductivity",

)

FEATURES = {

    "x_NaSalt": 0.12438287472883666,

    "x_EC": 0.6545108436328688,

    "x_PC": 0.5161749545925916,

}

OPERATIONS = (

    ("bge", QUERIES[0]),

    ("bge", QUERIES[1]),

    ("conductivity", 20.0),

    ("conductivity", 0.0),

    ("bge", QUERIES[2]),

    ("bge", QUERIES[3]),

)





def _sha(value) -> str:

    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()





class ProfiledBgeModelAdapter(ExternalBgeModelAdapter):

    """BGE adapter whose host-RAM admission cost comes from an earlier independent profile."""



    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:

        base = super().declared_cost(level)

        if level < ResidencyLevel.BOUND:

            return base

        return ResourceVector(

            prompt_tokens=base.prompt_tokens,

            host_ram_mb=PROFILED_BGE_HOST_MIB,

            gpu_vram_mb=base.gpu_vram_mb,

            process_slots=base.process_slots,

            connection_slots=base.connection_slots,

        )





class SameKindEvictionPolicy:

    """Evict only inactive residents with the same CapabilityKind as the target set."""



    def choose(self, *, states, catalog, violated_dimensions, protected_ids, cost_basis="declared"):

        target_kinds = {catalog.get(item).kind for item in protected_ids}

        if not target_kinds:

            return None

        candidates = []

        for capability_id, state in states.items():

            if capability_id in protected_ids or state.active_count > 0:

                continue

            capability = catalog.get(capability_id)

            if capability.kind not in target_kinds:

                continue

            target = capability.pinned_min_level

            if state.current_level <= target:

                continue

            cost_fn = capability.adapter.measured_cost if cost_basis == "measured" else capability.adapter.declared_cost

            released = cost_fn(state.current_level) - cost_fn(target)

            if not any(released.value(dim) > 0 for dim in violated_dimensions):

                continue

            candidates.append((state.last_used_tick, capability_id, target))

        if not candidates:

            return None

        candidates.sort(key=lambda item: (item[0], item[1]))

        _, capability_id, target = candidates[0]

        return capability_id, target





def _release_all(governor: CapabilityResidencyGovernor) -> int:

    count = 0

    for state in list(governor.states()):

        if state.current_level > ResidencyLevel.CATALOG and state.active_count == 0:

            governor.demote(state.capability_id, ResidencyLevel.CATALOG)

            count += 1

    return count





def run(

    strategy: str,

    host_budget_mib: float,

    *,

    model_path: Path = MODEL_PATH,

) -> dict:

    if strategy not in {"retain_all", "type_local", "release_all", "cwsr"}:

        raise ValueError(strategy)

    model_path = model_path.resolve()

    if not model_path.exists():

        raise FileNotFoundError(model_path)



    bge = ProfiledBgeModelAdapter(

        python_exe=ROOT_PYTHON,

        worker_script=WORKER_SCRIPT,

        model_path=Path(EMBEDDING_MODEL),

    )

    process = SodiumConductivityMlProcessAdapter(

        model_path=model_path,

        declared_host_ram_mb=PROFILED_PROCESS_HOST_MIB,

    )

    catalog = CapabilityCatalog((

        VirtualCapability(

            capability_id=MODEL_ID, kind=CapabilityKind.MODEL,

            summary="real external CUDA BGE-M3 model", adapter=bge,

            maximum_level=ResidencyLevel.BOUND,

        ),

        VirtualCapability(

            capability_id=PROCESS_ID, kind=CapabilityKind.PROCESS,

            summary="real sodium-electrolyte conductivity prediction process", adapter=process,

            maximum_level=ResidencyLevel.BOUND,

        ),

    ))

    kwargs = {

        "catalog": catalog,

        "budget": ResourceVector(

            host_ram_mb=float(host_budget_mib),

            gpu_vram_mb=GPU_BUDGET_MIB,

            process_slots=1.0,

        ),

    }

    if strategy == "retain_all":

        kwargs["policy"] = NoEvictionPolicy()

    elif strategy == "type_local":

        kwargs["policy"] = SameKindEvictionPolicy()

    governor = CapabilityResidencyGovernor(**kwargs)



    rows = []

    completed = 0

    failure = None

    release_count = 0

    started_all = monotonic()

    try:

        for index, (kind, payload) in enumerate(OPERATIONS):

            cap = MODEL_ID if kind == "bge" else PROCESS_ID

            started = monotonic()

            try:

                with governor.acquire_many(

                    ((cap, ResidencyLevel.BOUND),),

                    request_id=f"cross-kind:{strategy}:{index}:{kind}",

                    reconcile_before_active=True,

                ):

                    if kind == "bge":

                        vector = bge.embed_query(str(payload), request_id=f"cross-kind:{index}")

                        result = {"embedding_dim": len(vector), "embedding_sha256": _sha([round(x, 10) for x in vector])}

                    else:

                        temperature_c = float(payload)

                        result_raw = process.predict_features({

                            "inverse_temperature": temperature_c_to_inverse_temperature(temperature_c),

                            **FEATURES,

                        })

                        result = {

                            "status": result_raw.get("status"),

                            "prediction": result_raw.get("prediction"),

                        }

                    measured = governor.total_measured_cost()

                    row = {

                        "index": index, "kind": kind, "result_sha256": _sha(result),

                        "elapsed_s": monotonic() - started,

                        "measured": asdict(measured),

                        "model_resident": bge.is_loaded,

                        "process_resident": process.is_running,

                        "model_load_count": bge.load_count,

                        "process_pid": process.pid,

                    }

                    rows.append(row)

                completed += 1

                if strategy == "release_all":

                    release_count += _release_all(governor)

            except (ResidencyBudgetExceeded, MeasuredResidencyBudgetExceeded) as exc:

                failure = {

                    "index": index, "kind": kind, "error_type": type(exc).__name__,

                    "message": str(exc),

                    "measured": asdict(governor.total_measured_cost()),

                    "model_resident": bge.is_loaded,

                    "process_resident": process.is_running,

                }

                break



        events = Counter(e.event_type for e in governor.events)

        evictions = [

            {"capability_id": e.capability_id, "detail": e.detail}

            for e in governor.events

            if e.event_type == "demote" and e.detail in {"budget_eviction", "measured_budget_eviction"}

        ]

        return {

            "schema_version": "cwsr-cross-kind-real-pressure-probe-v1.1",

            "strategy": strategy,

            "conductivity_model_path": str(model_path),

            "conductivity_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),

            "host_budget_mib": float(host_budget_mib),

            "gpu_budget_mib": GPU_BUDGET_MIB,

            "profiled_costs": {"bge_host_mib": PROFILED_BGE_HOST_MIB, "process_host_mib": PROFILED_PROCESS_HOST_MIB},

            "operation_sequence": [x[0] for x in OPERATIONS],

            "complete": completed == len(OPERATIONS),

            "completed": completed,

            "failure": failure,

            "rows": rows,

            "result_signature": [r["result_sha256"] for r in rows],

            "model_load_count": bge.load_count,

            "process_promotions": governor.state(PROCESS_ID).promotions,

            "model_promotions": governor.state(MODEL_ID).promotions,

            "budget_evictions": evictions,

            "explicit_release_count": release_count,

            "event_counts": dict(events),

            "elapsed_s": monotonic() - started_all,

        }

    finally:

        governor.close()





def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--strategy", choices=("retain_all", "type_local", "release_all", "cwsr"), required=True)

    ap.add_argument("--host-budget-mib", type=float, default=750.0)

    ap.add_argument("--model-path", type=Path, default=MODEL_PATH)

    ap.add_argument("--output", type=Path, default=None)

    args = ap.parse_args()

    result = run(

        args.strategy,

        args.host_budget_mib,

        model_path=args.model_path,

    )

    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)

    if args.output:

        args.output.write_text(text + "\n", encoding="utf-8")

    print(text)

    return 0



if __name__ == "__main__":

    raise SystemExit(main())

