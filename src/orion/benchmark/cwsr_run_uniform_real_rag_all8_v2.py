"""逐任务隔离重跑：Driver 驻留，Session 按查询创建并关闭。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from orion.benchmark.cwsr_run_uniform_real_rag_all8_v1 import FAMILIES, _kill_tree, _summary

ROOT = Path(__file__).resolve().parents[1]
DETAIL = ROOT / "benchmark" / "cwsr_real_rag_uniform_all8_v2"
SUMMARY = ROOT / "benchmark" / "cwsr_real_rag_uniform_all8_v2_result.json"


def _extract(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("schema_version") != "cwsr-real-rag-kg-trace-replay-v2":
        raise RuntimeError(f"corrected family result failed: {path}")
    pair = payload["paired"][0]
    retained, released = pair["cwsr"], pair["release_all"]
    return {
        "family_id": pair["family_id"],
        "operation_count": retained["operation_count"],
        "rag_operation_count": retained["rag_operation_count"],
        "kg_operation_count": retained["kg_operation_count"],
        "cwsr_elapsed_s": retained["elapsed_to_last_result_s"],
        "release_all_elapsed_s": released["elapsed_to_last_result_s"],
        "seconds_saved_release_minus_cwsr": pair["seconds_saved_release_minus_cwsr"],
        "cwsr_model_load_count": retained["model_load_count"],
        "release_all_model_load_count": released["model_load_count"],
        "cwsr_driver_initializations": retained["neo4j_driver_initializations"],
        "release_all_driver_initializations": released["neo4j_driver_initializations"],
        "cwsr_sessions_created": retained["neo4j_sessions_created"],
        "release_all_sessions_created": released["neo4j_sessions_created"],
        "scientific_outputs_identical": pair["scientific_outputs_identical"],
    }


def _write_summary(rows: list[dict]) -> None:
    def metric(key: str) -> dict | None:
        return _summary([float(row[key]) for row in rows]) if rows else None

    payload = {
        "schema_version": "cwsr-real-rag-uniform-all8-v2",
        "status": "PASS" if len(rows) == len(FAMILIES) and all(
            row["scientific_outputs_identical"] and
            row["cwsr_sessions_created"] == row["release_all_sessions_created"]
            for row in rows
        ) else "PARTIAL",
        "timing_definition": "Per-family fresh process; BGE transport startup excluded; Driver/pool may persist across calls, Session is created and closed per query in both policies.",
        "rows": rows,
        "summary": {
            "completed_family_count": len(rows),
            "cwsr_elapsed_s": metric("cwsr_elapsed_s"),
            "release_all_elapsed_s": metric("release_all_elapsed_s"),
            "seconds_saved_release_minus_cwsr": metric("seconds_saved_release_minus_cwsr"),
            "cwsr_model_load_count": sum(row["cwsr_model_load_count"] for row in rows),
            "release_all_model_load_count": sum(row["release_all_model_load_count"] for row in rows),
            "cwsr_driver_initializations": sum(row["cwsr_driver_initializations"] for row in rows),
            "release_all_driver_initializations": sum(row["release_all_driver_initializations"] for row in rows),
            "cwsr_sessions_created": sum(row["cwsr_sessions_created"] for row in rows),
            "release_all_sessions_created": sum(row["release_all_sessions_created"] for row in rows),
            "positive_time_saving_family_count": sum(row["seconds_saved_release_minus_cwsr"] > 0 for row in rows),
        },
    }
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    DETAIL.mkdir(exist_ok=True)
    rows: list[dict] = []
    for family in FAMILIES:
        result = DETAIL / f"{family}.json"
        stdout = DETAIL / f"{family}.stdout.log"
        stderr = DETAIL / f"{family}.stderr.log"
        if result.exists():
            result.unlink()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT.parent)
        cmd = [sys.executable, "-m", "orion.benchmark.cwsr_real_rag_kg_trace_replay_v2", "--family", family, "--output", str(result)]
        with stdout.open("w", encoding="utf-8") as out_fh, stderr.open("w", encoding="utf-8") as err_fh:
            process = subprocess.Popen(cmd, cwd=str(ROOT.parent), env=env, stdout=out_fh, stderr=err_fh)
            deadline = time.monotonic() + 240.0
            try:
                while time.monotonic() < deadline:
                    if result.exists() and result.stat().st_size > 0:
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.5)
                if not result.exists() or result.stat().st_size == 0:
                    tail = stderr.read_text(encoding="utf-8", errors="replace")[-2000:]
                    raise RuntimeError(f"{family}: no result, rc={process.poll()}, stderr={tail!r}")
                rows.append(_extract(result))
                _write_summary(rows)
                print(f"{family}: {rows[-1]['cwsr_elapsed_s']:.3f} / {rows[-1]['release_all_elapsed_s']:.3f} s", flush=True)
            finally:
                if process.poll() is None:
                    _kill_tree(process.pid)
                time.sleep(2.0)
    return 0 if len(rows) == len(FAMILIES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
