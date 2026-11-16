"""Measure restart cost for a conductivity prediction process that loads a pretrained model.

The final RandomForest is fitted once before timing, serialized with joblib, and reused by
all process lifecycles. Timed intervals therefore cover process startup, Python/module
initialization, model loading, and readiness for prediction; model fitting is excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from capvm import (
    CapabilityCatalog,
    CapabilityKind,
    CapabilityResidencyGovernor,
    ResidencyLevel,
    ResourceVector,
    VirtualCapability,
)
from capvm.adapters import ProcessCapabilityAdapter
from tools.electrolyte_design.conductivity_ml import (
    FEATURE_COLUMNS,
    UPSTREAM_DATA_DOI,
    UPSTREAM_DATA_URL,
    SodiumConductivityRandomForest,
    download_upstream_na_frames,
    load_upstream_na_frames,
    prepare_training_table,
)
from tools.electrolyte_design.conductivity_process import SodiumConductivityMlProcessAdapter


CYCLE_COUNT = 10
N_ESTIMATORS = 192
RANDOM_STATE = 20260906
BUDGET = ResourceVector(host_ram_mb=1024.0, process_slots=1.0)


def _feature_requests(table) -> list[dict[str, float]]:
    ordered = table.sort_values(list(FEATURE_COLUMNS)).reset_index(drop=True)
    indices = (len(ordered) // 5, len(ordered) // 2, (4 * len(ordered)) // 5)
    requests: list[dict[str, float]] = []
    for index in indices:
        row = ordered.iloc[index]
        requests.append(
            {
                "inverse_temperature": float(row["inverse temperature"]),
                "x_NaSalt": float(row["x_NaSalt"]),
                "x_EC": float(row["x_EC"]),
                "x_PC": float(row["x_PC"]),
            }
        )
    return requests


def _prediction_projection(result: dict[str, Any]) -> tuple[object, ...]:
    if result.get("status") != "RESULT":
        return (str(result.get("status")),)
    prediction = result["prediction"]
    return (
        "RESULT",
        round(float(prediction["conductivity_s_cm"]), 12),
        round(float(prediction["log10_conductivity_s_cm"]), 12),
        round(float(prediction["ensemble_std_log10"]), 12),
    )


def _summary(values: list[float]) -> dict[str, float]:
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "median": statistics.median(values),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "min": min(values),
        "max": max(values),
    }


def _run_cycle(
    *,
    cycle_index: int,
    model_path: Path,
    feature_requests: list[dict[str, float]],
) -> dict[str, object]:
    adapter = SodiumConductivityMlProcessAdapter(
        model_path=model_path,
        worker_script=ROOT / "benchmark" / "cwsr_na_conductivity_ml_worker_v1.py",
    )
    pressure = ProcessCapabilityAdapter(
        resident_mb=8.0,
        runtime_overhead_mb=16.0,
        startup_delay_s=0.0,
    )
    governor = CapabilityResidencyGovernor(
        catalog=CapabilityCatalog(
            [
                VirtualCapability(
                    capability_id="na_electrolyte_conductivity_ml",
                    kind=CapabilityKind.PROCESS,
                    summary="pretrained sodium-electrolyte conductivity predictor",
                    adapter=adapter,
                    maximum_level=ResidencyLevel.BOUND,
                ),
                VirtualCapability(
                    capability_id="controlled_process_pressure",
                    kind=CapabilityKind.PROCESS,
                    summary="controlled process-slot pressure",
                    adapter=pressure,
                    maximum_level=ResidencyLevel.BOUND,
                ),
            ]
        ),
        budget=BUDGET,
    )

    try:
        governor.access(
            "na_electrolyte_conductivity_ml",
            ResidencyLevel.BOUND,
            request_id=f"serialized-timing-{cycle_index:02d}-first",
        )
        governor.reconcile_measured_budget()
        first_pid = adapter.pid
        first_time = float(adapter.last_startup_latency_s)
        first_model_load = float((adapter.ready_metadata or {}).get("model_load_seconds") or 0.0)
        first_rss = float(governor.total_measured_cost().host_ram_mb)
        first_predictions = [
            _prediction_projection(adapter.predict_features(features))
            for features in feature_requests
        ]

        governor.access(
            "controlled_process_pressure",
            ResidencyLevel.BOUND,
            request_id=f"serialized-timing-{cycle_index:02d}-pressure",
        )
        governor.reconcile_measured_budget()
        evicted_after_pressure = not adapter.is_running

        governor.access(
            "na_electrolyte_conductivity_ml",
            ResidencyLevel.BOUND,
            request_id=f"serialized-timing-{cycle_index:02d}-restart",
        )
        governor.reconcile_measured_budget()
        restart_pid = adapter.pid
        restart_time = float(adapter.last_startup_latency_s)
        restart_model_load = float((adapter.ready_metadata or {}).get("model_load_seconds") or 0.0)
        restart_rss = float(governor.total_measured_cost().host_ram_mb)
        restart_predictions = [
            _prediction_projection(adapter.predict_features(features))
            for features in feature_requests
        ]

        return {
            "cycle": cycle_index,
            "first_pid": first_pid,
            "restart_pid": restart_pid,
            "new_pid_after_eviction": first_pid is not None and restart_pid is not None and first_pid != restart_pid,
            "evicted_after_pressure": evicted_after_pressure,
            "first_startup_and_load_s": first_time,
            "first_model_load_s": first_model_load,
            "restart_startup_and_load_s": restart_time,
            "restart_model_load_s": restart_model_load,
            "first_rss_mib": first_rss,
            "restart_rss_mib": restart_rss,
            "predictions_identical_after_restart": first_predictions == restart_predictions,
        }
    finally:
        governor.close()


def run(*, data_zip: Path | None = None) -> dict[str, object]:
    frames = (
        load_upstream_na_frames(data_zip)
        if data_zip is not None
        else download_upstream_na_frames()
    )
    table = prepare_training_table(frames)
    feature_requests = _feature_requests(table)

    with tempfile.TemporaryDirectory(prefix="cwsr-conductivity-model-") as tmp:
        model_path = Path(tmp) / "na_conductivity_rf.joblib"
        model = SodiumConductivityRandomForest(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
        ).fit(table)
        joblib.dump(model, model_path, compress=0)
        model_bytes = model_path.stat().st_size

        cycles = [
            _run_cycle(
                cycle_index=index + 1,
                model_path=model_path,
                feature_requests=feature_requests,
            )
            for index in range(CYCLE_COUNT)
        ]

    first_times = [float(row["first_startup_and_load_s"]) for row in cycles]
    restart_times = [float(row["restart_startup_and_load_s"]) for row in cycles]
    first_loads = [float(row["first_model_load_s"]) for row in cycles]
    restart_loads = [float(row["restart_model_load_s"]) for row in cycles]
    first_rss = [float(row["first_rss_mib"]) for row in cycles]
    restart_rss = [float(row["restart_rss_mib"]) for row in cycles]

    passed = all(
        bool(row["new_pid_after_eviction"])
        and bool(row["evicted_after_pressure"])
        and bool(row["predictions_identical_after_restart"])
        for row in cycles
    )
    return {
        "schema_version": "cwsr-conductivity-process-timing-serialized-v2.0",
        "status": "PASS" if passed else "FAIL",
        "source": {
            "data_doi": UPSTREAM_DATA_DOI,
            "location": str(data_zip.resolve()) if data_zip is not None else UPSTREAM_DATA_URL,
            "training_rows": int(len(table)),
            "training_formulations": int(table["formulation_group"].nunique()),
        },
        "design": {
            "cycle_count": CYCLE_COUNT,
            "measured_process_startup_count": CYCLE_COUNT * 2,
            "model_pretrained_once_before_timing": True,
            "serialized_model_bytes": int(model_bytes),
            "random_forest_trees": N_ESTIMATORS,
            "random_seed": RANDOM_STATE,
            "budget": {
                "host_ram_mb": BUDGET.host_ram_mb,
                "process_slots": BUDGET.process_slots,
            },
        },
        "summary": {
            "first_startup_and_load_s": _summary(first_times),
            "restart_startup_and_load_s": _summary(restart_times),
            "first_model_load_s": _summary(first_loads),
            "restart_model_load_s": _summary(restart_loads),
            "first_rss_mib": _summary(first_rss),
            "restart_rss_mib": _summary(restart_rss),
            "all_new_pid_after_eviction": all(bool(row["new_pid_after_eviction"]) for row in cycles),
            "all_evicted_after_pressure": all(bool(row["evicted_after_pressure"]) for row in cycles),
            "all_predictions_identical_after_restart": all(
                bool(row["predictions_identical_after_restart"]) for row in cycles
            ),
        },
        "cycles": cycles,
        "interpretation": (
            "Timing excludes model fitting. Each measured startup creates a new Python process, "
            "loads the same pretrained RandomForest artifact, and ends when the worker is ready "
            "to serve predictions."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-zip", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(data_zip=args.data_zip), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
