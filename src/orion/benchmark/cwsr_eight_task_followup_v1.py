"""原八个电解液任务的修复后随访；不改写冻结 Study4C 或固定检索计时。

每个任务独立进程、空记忆和只读工具运行。输入为原问题，不注入候选或预期答案。
输出逐个保存完整报告、结构化结果、按 tool_call_id 对应的工具输入输出及时间线。
本脚本不评价电导率，不对资源策略作性能比较。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
ARCHIVE = ROOT / "benchmark/self_evolution_study4c_v1_20260822/formal_run_v1/raw"
PROFILE = ROOT / "benchmark/e3_current_snapshot_r1_20260814/tool_profile.json"
ORDER = ("P2", "U2", "U1", "P1", "C1", "C2", "C3", "D1")


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def selected():
    rows = {}
    for path in ARCHIVE.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("arm") == "G_evidence_governed" and data.get("repeat_index") == 1:
            family = data["family_id"]
            short = next(k for k in ORDER if k in family.split("_"))
            if short in rows:
                raise ValueError(f"Duplicate original task: {short}")
            rows[short] = (path, data)
    if set(rows) != set(ORDER):
        raise ValueError(f"Expected original eight, got {sorted(rows)}")
    return rows


def snapshot():
    files = ["main.py", "model_config.py", "llms.py", "core/design_runtime.py",
             "core/design_decision.py", "evaluation/agent_runner.py", "tools/kg/models.py",
             "tools/electrolyte_design/criteria.py", "tools/electrolyte_design/evaluator.py",
             "tools/electrolyte_design/scientific_numeric_text.py", "skills/electrolyte_design/SKILL.md",
             "tools/electrolyte_design/synthesis.py", "tools/electrolyte_design/models.py",
             "tools/electrolyte_design/formulation_evidence.py", "tools/electrolyte_design/formulate.py",
             "task_evidence/scientific_text.py",
             "core/retrieved_evidence.py", "tools/electrolyte_design/screen.py",
             "task_evidence/condition_extraction.py", "task_evidence/evidence_applicability.py",
             "benchmark/cwsr_eight_task_followup_v1.py"]
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}


def run_one(short, output, timeout):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path, source = selected()[short]
    path = output / f"{short}.json"
    if path.exists():
        raise FileExistsError(path)
    # 配置先于 main/llms 导入，密钥沿用项目环境而不写入工件。
    os.environ.update({
        "ORION_AGENT_PROVIDER": "deepseek", "ORION_AGENT_MODEL": "deepseek-v4-flash",
        "ORION_AGENT_TEMPERATURE": "0.0", "ORION_AGENT_TOP_P": "1.0",
        "ORION_AGENT_THINKING": "enabled",
        "ORION_GROUNDING_CHECK": "0", "ORION_LLM_CACHE": "0", "ORION_LLM_STREAM": "0",
        "ORION_REFLECTION_INJECT": "0", "ORION_ABLATION_NO_KG": "0",
        "ORION_ABLATION_NO_FORMULATION": "0", "ORION_DESIGN_REASONING_MODE": "semantic",
        "ORION_RESOURCE_POOL_ENABLED": "1",
        "ORION_RESOURCE_EVENTS_PATH": str(output / f"{short}.resource-events.jsonl"),
        "ORION_EVAL_AGENT_TIMEOUT": str(timeout),
        "ORION_EVAL_LLM_STEP_TIMEOUT": "90", "ORION_EVAL_RECURSION_LIMIT": "100",
        "ORION_EVAL_LLM_REQUEST_MAX_RETRIES": "2", "ORION_LLM_REQUEST_MAX_RETRIES": "2",
    })
    with tempfile.TemporaryDirectory(prefix=f"cwsr-eight-{short}-", dir=output, ignore_cleanup_errors=True) as tmp:
        os.environ["ORION_EVOLUTION_STORE"] = str(Path(tmp) / "empty_evolution_store")
        os.chdir(tmp)
        from langchain_core.messages import AIMessage, ToolMessage
        from orion.evaluation.models import EvaluationSample, ToolCallRecord
        import orion.evaluation.agent_runner as runner
        from orion.benchmark.self_evolution_fresh_v2_development_runtime import _V2IsolatedContextManager
        main = importlib.import_module("orion.main")
        main._context_manager = _V2IsolatedContextManager(memory_service=main._memory_service, model_name="cwsr-eight-followup")
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        allowed = set(profile["allowed_tool_names"])
        if allowed & set(profile["blocked_side_effect_or_out_of_scope_tool_names"]):
            raise ValueError("Unsafe tool profile")
        exposed = {t.name: t for t in main._BASE_TOOLS}
        if allowed - set(exposed):
            raise ValueError(f"Missing tools: {sorted(allowed - set(exposed))}")
        main._BASE_TOOLS = tuple(exposed[name] for name in sorted(allowed))
        main._configure_runtime_tools()
        messages_dump = []

        def paired_calls(messages):
            # 当前历史 runner 的相邻结果启发式不适用于同轮多个工具；按真实 ID 保存。
            messages_dump.extend(m.model_dump(mode="json") for m in messages)
            results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
            records = []
            for message in messages:
                if isinstance(message, AIMessage):
                    for call in message.tool_calls or []:
                        result = results.get(call.get("id"))
                        records.append(ToolCallRecord(name=call["name"], args=call.get("args") or {},
                                                      result=result.content if result is not None else None))
            return records

        runner.extract_tool_calls = paired_calls
        sample = EvaluationSample(question=source["sample"]["question"], task_type="design",
                                  ground_truth="", expected_tools=[], tags=["cwsr_original_eight_followup"],
                                  benchmark_case_id=source["case_id"])
        trace_path = output / f"{short}.events.jsonl"
        def sink(event):
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        raw = runner.run_agent(sample, config=SimpleNamespace(auto_approve=True),
                               agent_factory=lambda: main._build_graph(main.llm_async, rebuild_memory=False), event_sink=sink)
        raw["tool_calls"] = [r.model_dump(mode="json") for r in raw.get("tool_calls", [])]
        runtime = getattr(main, "_default_resource_runtime", None)
        runtime_snapshot = list(runtime.snapshot()) if runtime is not None else []
        if runtime is not None:
            runtime.close()
        resource_event_path = output / f"{short}.resource-events.jsonl"
        record = {
            "schema_version": "cwsr_original_eight_followup_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "short_id": short, "original_family_id": source["family_id"], "original_case_id": source["case_id"],
            "original_raw_path": str(source_path), "original_raw_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "sample": sample.model_dump(mode="json"), "source_sha256": snapshot(),
            "run_configuration": {"model": "deepseek-v4-flash", "temperature": 0, "top_p": 1,
                                  "thinking": os.environ.get("ORION_AGENT_THINKING", "disabled"),
                                  "design_reasoning_mode": os.environ.get("ORION_DESIGN_REASONING_MODE"),
                                  "resource_pool_enabled": os.environ.get("ORION_RESOURCE_POOL_ENABLED"),
                                  "agent_timeout_seconds": timeout, "tool_profile_sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest()},
            "interpretation": {"previously_observed_questions": True, "seed_injection": False,
                               "expected_answer_injection": False, "independent_semantic_validation": False,
                               "resource_policy_comparison": False, "conductivity_prediction": False,
                               "historical_records_overwritten": False},
            "runtime_snapshot_before_close": runtime_snapshot,
            "runtime_resource_events_path": str(resource_event_path),
            "runtime_resource_events_sha256": (
                hashlib.sha256(resource_event_path.read_bytes()).hexdigest()
                if resource_event_path.exists() else None
            ),
            **raw, "messages": messages_dump,
        }
        dump(path, record)
        dump(output / f"{short}.messages.json", messages_dump)
        (output / f"{short}.report.md").write_text(record.get("answer") or "", encoding="utf-8")
        artifact = record.get("task_artifact") or {}
        print(json.dumps({"task": short, "ok": record["ok"], "status": artifact.get("task_status"),
                          "duration": record.get("run_duration_seconds"), "tool_count": len(raw["tool_calls"])}, ensure_ascii=False), flush=True)
        os.chdir(ROOT)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", choices=ORDER)
    parser.add_argument("--timeout", type=int, default=480)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT.parent):
        raise ValueError("Output must stay in project workspace")
    if args.worker:
        if not args.execute:
            raise ValueError("Execution requires --execute")
        run_one(args.worker, output, args.timeout)
        return
    tasks = selected()
    if not args.execute:
        print(json.dumps({k: {"family": tasks[k][1]["family_id"], "question": tasks[k][1]["sample"]["question"]} for k in ORDER}, ensure_ascii=False, indent=2))
        return
    if output.exists():
        raise FileExistsError("Choose a new output directory; historical outputs are immutable")
    with urllib.request.urlopen("http://127.0.0.1:8799/v1/metadata", timeout=30) as r:
        preflight = json.loads(r.read())
    output.mkdir(parents=True)
    dump(output / "manifest.json", {"tasks": list(ORDER), "preflight": preflight, "source_sha256": snapshot(),
                                     "timeout_per_task": args.timeout, "purpose": "Original eight task followup after numeric checks repaired"})
    for short in ORDER:
        print(f"START {short}", flush=True)
        with (output / f"{short}.stdout.log").open("w", encoding="utf-8") as stdout, (output / f"{short}.stderr.log").open("w", encoding="utf-8") as stderr:
            try:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--execute", "--worker", short,
                                          "--output", str(output), "--timeout", str(args.timeout)],
                                         cwd=ROOT, stdout=stdout, stderr=stderr, timeout=args.timeout + 120)
                print(f"END {short} process_exit={process.returncode}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"END {short} process_timeout; inspect logs and saved checkpoints", flush=True)
    print("COLLECTION FINISHED", flush=True)


if __name__ == "__main__":
    main_cli()
