"""Run all eight preselected real RAG/KG traces under one timing definition.

Each family is executed in a fresh child process using the current
cwsr_real_rag_kg_trace_replay_v1.py implementation. The BGE transport shell is
prepared before the timing window; only model load/unload, Neo4j session
materialization, and query execution contribute to elapsed_to_last_result_s.

Families are run strictly serially. Once a child has written a complete result
JSON, its process tree is terminated before the next family begins to prevent
cross-family CUDA/resource contamination.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "benchmark" / "cwsr_real_rag_kg_trace_replay_v1.py"
DETAIL = ROOT / "benchmark" / "cwsr_real_rag_uniform_all8_v1"
SUMMARY = ROOT / "benchmark" / "cwsr_real_rag_uniform_all8_v1_result.json"

FAMILIES = (
    "STUDY4C_C1_CAI_NADFP_SCOPE_MISSING",
    "STUDY4C_C2_LIN_HALFCELL_LIMIT_TO_FULLCELL",
    "STUDY4C_C3_CUI_COIN_TO_POUCH_MINUS40C",
    "STUDY4C_D1_FAN_NAFSI_CORROSION",
    "STUDY4C_P1_MOGENSEN_NABOB_50C",
    "STUDY4C_P2_LU_SA_MINUS20C_POUCH",
    "STUDY4C_U1_CHE_PCEMC_MINUS30C",
    "STUDY4C_U2_FAN_DUALSALT_MINUS20C",
)


def _kill_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("values required")
    if len(values) == 1:
        v = float(values[0])
        return {"median": v, "q1": v, "q3": v, "min": v, "max": v}
    q = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "median": float(statistics.median(values)),
        "q1": float(q[0]),
        "q3": float(q[2]),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _extract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"family result failed: {path}")
    row = payload["paired"][0]
    cwsr = row["cwsr"]
    release = row["release_all"]
    release_s = float(release["elapsed_to_last_result_s"])
    saved_s = float(row["seconds_saved_release_minus_cwsr"])
    return {
        "family_id": row["family_id"],
        "operation_count": int(cwsr["operation_count"]),
        "rag_operation_count": int(cwsr["rag_operation_count"]),
        "kg_operation_count": int(cwsr["kg_operation_count"]),
        "cwsr_elapsed_s": float(cwsr["elapsed_to_last_result_s"]),
        "release_all_elapsed_s": release_s,
        "seconds_saved_release_minus_cwsr": saved_s,
        "fraction_saved_release_minus_cwsr": saved_s / release_s,
        "cwsr_model_load_count": int(cwsr["model_load_count"]),
        "release_all_model_load_count": int(release["model_load_count"]),
        "cwsr_model_load_total_s": float(cwsr["model_load_total_s"]),
        "release_all_model_load_total_s": float(release["model_load_total_s"]),
        "cwsr_model_shutdown_total_s": float(cwsr["model_shutdown_total_s"]),
        "release_all_model_shutdown_total_s": float(release["model_shutdown_total_s"]),
        "cwsr_session_promotions": int(cwsr["session_promotions"]),
        "release_all_session_promotions": int(release["session_promotions"]),
        "scientific_outputs_identical": bool(row["scientific_outputs_identical"]),
        "cwsr_peak_measured": cwsr["peak_measured"],
        "release_all_peak_measured": release["peak_measured"],
        "transport_startup_s_excluded": float(cwsr["transport_startup_s_excluded"]),
    }


def _write_summary(rows: list[dict[str, Any]]) -> None:
    savings_s = [float(row["seconds_saved_release_minus_cwsr"]) for row in rows]
    fractions = [float(row["fraction_saved_release_minus_cwsr"]) for row in rows]
    cwsr_elapsed = [float(row["cwsr_elapsed_s"]) for row in rows]
    release_elapsed = [float(row["release_all_elapsed_s"]) for row in rows]
    model_loads_saved = [
        int(row["release_all_model_load_count"]) - int(row["cwsr_model_load_count"])
        for row in rows
    ]
    sessions_saved = [
        int(row["release_all_session_promotions"]) - int(row["cwsr_session_promotions"])
        for row in rows
    ]
    payload = {
        "schema_version": "cwsr-real-rag-uniform-all8-v1",
        "status": "PASS"
        if len(rows) == len(FAMILIES)
        and all(bool(row["scientific_outputs_identical"]) for row in rows)
        else "PARTIAL",
        "timing_definition": (
            "BGE transport shell startup excluded; elapsed time includes BGE model "
            "load/unload, Neo4j session materialization, and real query execution."
        ),
        "rows": rows,
        "summary": {
            "completed_family_count": len(rows),
            "seconds_saved_release_minus_cwsr": _summary(savings_s) if rows else None,
            "fraction_saved_release_minus_cwsr": _summary(fractions) if rows else None,
            "cwsr_elapsed_s": _summary(cwsr_elapsed) if rows else None,
            "release_all_elapsed_s": _summary(release_elapsed) if rows else None,
            "model_loads_saved": _summary([float(v) for v in model_loads_saved])
            if rows
            else None,
            "session_promotions_saved": _summary([float(v) for v in sessions_saved])
            if rows
            else None,
            "positive_time_saving_family_count": sum(value > 0 for value in savings_s),
            "all_scientific_outputs_identical": all(
                bool(row["scientific_outputs_identical"]) for row in rows
            ),
        },
    }
    SUMMARY.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    DETAIL.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []

    # Reuse the already completed C1 result if present and current-schema.
    c1_existing = ROOT / "benchmark" / "cwsr_real_rag_uniform_C1.json"
    start_index = 0
    if c1_existing.exists():
        try:
            rows.append(_extract(c1_existing))
            _write_summary(rows)
            start_index = 1
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        except Exception:
            rows.clear()
            start_index = 0

    for family in FAMILIES[start_index:]:
        result = DETAIL / f"{family}.json"
        stdout = DETAIL / f"{family}.stdout.log"
        stderr = DETAIL / f"{family}.stderr.log"
        for path in (result, stdout, stderr):
            if path.exists():
                path.unlink()

        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT.parent)
        cmd = [
            sys.executable,
            str(REPLAY),
            "--family",
            family,
            "--output",
            str(result),
        ]

        with stdout.open("w", encoding="utf-8") as out_fh, stderr.open(
            "w", encoding="utf-8"
        ) as err_fh:
            process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=out_fh,
                stderr=err_fh,
            )
            deadline = time.monotonic() + 240.0
            try:
                while time.monotonic() < deadline:
                    if result.exists() and result.stat().st_size > 0:
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.5)

                if not result.exists() or result.stat().st_size == 0:
                    error_tail = (
                        stderr.read_text(encoding="utf-8", errors="replace")[-5000:]
                        if stderr.exists()
                        else ""
                    )
                    raise RuntimeError(
                        f"family did not produce result: {family}; "
                        f"rc={process.poll()}; stderr={error_tail!r}"
                    )

                row = _extract(result)
                rows.append(row)
                _write_summary(rows)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            finally:
                if process.poll() is None:
                    _kill_tree(process.pid)
                time.sleep(2.0)

    _write_summary(rows)
    return 0 if len(rows) == len(FAMILIES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
