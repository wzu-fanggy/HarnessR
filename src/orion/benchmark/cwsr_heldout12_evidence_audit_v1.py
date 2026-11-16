"""12-formulation live Agent evidence audit (read-only).

Independently inspect raw tool results, runtime resource state, held-out rank,
and user-facing science-report completeness. Does not rewrite historical runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "reports/cwsr_heldout12_agent_pair_20261014_v7"
DATA = ROOT / "reports/cwsr_leca_rf_comparison_20261012_v1"
TEMPERATURES = (-20.0, -10.0, 0.0, 20.0)
LOW = (-20.0, -10.0, 0.0)


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def archived() -> dict[tuple[str, float], float]:
    rows = {}
    with (DATA / "predictions.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] != "rf":
                continue
            key = (row["candidate_id"], float(row["target_temperature_c"]))
            if key in rows:
                raise ValueError("Duplicate frozen RF row: " + str(key))
            rows[key] = float(row["predicted_ms_cm"])
    if len(rows) != 48:
        raise ValueError(f"Expected 48 archived RF predictions, found {len(rows)}")
    return rows


def parse_tool_prediction(row) -> list[dict]:
    value = row.get("result")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    records = []
    for item in value:
        if not isinstance(item, dict):
            continue
        prediction_list = item.get("model_predictions") or []
        if isinstance(prediction_list, dict):
            prediction_list = [prediction_list]
        for prediction in prediction_list:
            if not isinstance(prediction, dict):
                continue
            records.append({"candidate_id": item.get("candidate_id"),
                            "temperature_c": prediction.get("temperature_c"),
                            "numeric_value": prediction.get("numeric_value"),
                            "status": prediction.get("status")})
    return records


def prediction_audit(run: dict, expected: dict) -> dict:
    observed = {}
    requests = []
    duplicate = []
    for call in run.get("tool_calls", []):
        if call.get("name") != "predict_formulation_conductivity":
            continue
        req = (call.get("args") or {}).get("request") or {}
        temperature = req.get("temperature_c")
        requests.append({"temperature_c": temperature,
                         "candidate_count": len(req.get("candidates") or [])})
        for row in parse_tool_prediction(call):
            try:
                key = (row["candidate_id"], float(row["temperature_c"]))
                predicted = float(row["numeric_value"])
            except (TypeError, ValueError):
                continue
            if key in observed:
                duplicate.append(str(key))
            observed[key] = predicted
    matches = sum(k in expected and math.isclose(v, expected[k], rel_tol=1e-9, abs_tol=1e-8)
                  for k, v in observed.items())
    artifact_records = ((run.get("task_artifact") or {}).get("model_comparison") or {}).get("records") or []
    artifact_matches = 0
    artifact_keys = set()
    for row in artifact_records:
        pred = row.get("prediction") or {}
        try:
            key = (row["candidate_id"], float(pred["temperature_c"]))
            val = float(pred["numeric_value"])
        except (KeyError, TypeError, ValueError):
            continue
        if key in expected and math.isclose(val, expected[key], rel_tol=1e-9, abs_tol=1e-8):
            artifact_matches += 1
        artifact_keys.add(key)
    return {"tool_prediction_calls": len(requests), "request_shape": requests,
            "unique_tool_points": len(observed), "matching_tool_points": matches,
            "duplicated_tool_points": duplicate,
            "artifact_points": len(artifact_records), "artifact_unique_points": len(artifact_keys),
            "matching_artifact_points": artifact_matches,
            "raw_and_artifact_points_both_complete": (
                len(observed) == matches == len(expected) == len(artifact_keys) == artifact_matches == 48
                and not duplicate
            )}


def report_audit(run: dict, expected: dict) -> dict:
    answer = str(run.get("answer") or "")
    section = answer.split("### 初筛结果", 1)[-1].split("### 文献依据", 1)[0]
    top4 = sorted({k[0] for k in expected}, key=lambda cid: -min(expected[(cid, t)] for t in LOW))[:4]
    name_listing = bool(re.search(r"前\s*四|前\s*4|四\s*种\s*优先|top\s*[- ]?4", section, flags=re.I))
    explicit_top_four = name_listing and all(cid in section for cid in top4)
    has_validation_plan = bool(
        re.search(r"(后续实验|验证方案|下一步验证|EIS|阻抗谱|恒电流|电芯循环|电导率实测)", answer)
    )
    has_citation = bool(re.search(r"\[.+?::sec\d+::child\d+\]", answer))
    task = run.get("task_artifact") or {}
    literature_claims = [claim for claim in task.get("claims", [])
                         if claim.get("evidence_ids")]
    provenance = task.get("runtime_provenance") or {}
    comparison = task.get("model_comparison") or {}
    declared_n = (comparison.get("request") or {}).get("priority_count", 1)
    submitted_plan = (provenance.get("validation_plan_source") == "agent_submitted"
                      and len(task.get("validation_plan") or []) >= 2)
    selected_top = comparison.get("priority_candidate_ids") or []
    distinct_source_ids = {str(source) for claim in literature_claims
                           for source in (claim.get("evidence_ids") or [])}
    source_kind_warning = "科学文献引文已通过归属检查，尚未人工确认语义完整性"
    issues = []
    if not explicit_top_four or declared_n != 4 or selected_top != top4:
        issues.append("报告及结构化数据未明确列出按预测排序的前四种候选")
    if not has_validation_plan:
        issues.append("缺少明确的后续电导率实测、电芯或界面验证计划")
    if not submitted_plan:
        issues.append("Agent 未提交至少两项独立验证建议；自动补充的通用模板不计作 Agent 自主完成")
    if not has_citation or not literature_claims:
        issues.append("缺少可追溯文献依据")
    if len(distinct_source_ids) < 2:
        issues.append("不同来源的相关文献背景事实不足两项")
    if any(str(claim.get("text") or "").rstrip().lower().endswith("a high") for claim in literature_claims):
        issues.append("至少一条文献摘句在句中截断，不能直接作为完整科学解释")
    return {"has_full_12x4_prediction_table": all(
                re.search(r"(?m)^\\|\\s*" + re.escape(cid) + r"\\s*\\|", answer)
                for cid in sorted({k[0] for k in expected})),
            "priority_top4_expected": top4,
            "explicit_top_four_in_result_section": explicit_top_four,
            "has_followup_validation_plan": has_validation_plan,
            "has_source_citations": has_citation,
            "literature_claim_count": len(literature_claims),
            "distinct_literature_source_count": len(distinct_source_ids),
            "declared_priority_count": declared_n,
            "structured_priority_ids": selected_top,
            "agent_submitted_validation_plan": submitted_plan,
            "validation_plan_source": provenance.get("validation_plan_source"),
            "issues": issues, "full_report_quality_pass": len(issues) == 0,
            "literature_semantic_verification": source_kind_warning}


def rank_audit(expected: dict) -> dict:
    split = read_json(DATA / "split.json")
    ids = [row["candidate_id"] for row in split["held_out_formulations"]]
    observations = {(row["candidate_id"], float(row["target_temperature_c"])): float(row["measured_ms_cm"])
                    for row in split["test_conditions"]}
    if set(expected) != set(observations):
        raise ValueError("Prediction/measurement key mismatch")
    cold_pred = {cid: min(expected[cid, t] for t in LOW) for cid in ids}
    cold_real = {cid: min(observations[cid, t] for t in LOW) for cid in ids}
    predicted_rank = sorted(ids, key=lambda cid: -cold_pred[cid])
    measured_rank = sorted(ids, key=lambda cid: -cold_real[cid])
    rho = float(spearmanr([cold_pred[cid] for cid in ids], [cold_real[cid] for cid in ids]).statistic)
    sigma20_pred = sorted(ids, key=lambda cid: -expected[cid, 20.0])
    low_measured_minima = {cid: min(LOW, key=lambda t: observations[cid, t]) for cid in ids}
    minus20_mae = sum(abs(expected[cid, -20.] - observations[cid, -20.]) for cid in ids) / len(ids)
    rho_minus20 = float(spearmanr([expected[cid, -20.] for cid in ids],
                                 [observations[cid, -20.] for cid in ids]).statistic)
    monotone = all(
        expected[cid, -20.0] <= expected[cid, -10.0] <= expected[cid, 0.0] <= expected[cid, 20.0]
        for cid in ids
    )
    return {"spearman_low_temperature": rho,
            "spearman_at_minus20": rho_minus20, "mae_at_minus20_mS_cm": minus20_mae,
            "measured_minimum_temperature_per_formulation": low_measured_minima,
            "top4_overlap": len(set(predicted_rank[:4]) & set(measured_rank[:4])),
            "predicted_top4": predicted_rank[:4], "measured_top4": measured_rank[:4],
            "predicted_room_top1": sigma20_pred[0],
            "all_low_temperature_minima_occur_at_minus20": all(cold_pred[cid] == expected[cid, -20.] for cid in ids),
            "predicted_conductivity_increases_with_temperature_for_all": monotone,
            "ranked_candidates": [
                {"id": cid, "predicted_worst_low_temp_mS_cm": round(cold_pred[cid], 6),
                 "measured_worst_low_temp_mS_cm": round(cold_real[cid], 6),
                 "predicted_room_mS_cm": round(expected[cid, 20.], 6)}
                for cid in predicted_rank]}


def mechanisms(folder: Path, native: dict, runtime: dict) -> dict:
    base_check = read_json(folder / "T1-r1-native/shared-admission-checks.json")
    rt_check = read_json(folder / "T1-r1-runtime/shared-admission-checks.json")
    events = [json.loads(line) for line in
              (folder / "T1-r1-runtime/runtime-events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    rf_events = [e for e in events if e.get("resource_id") == "process:na-conductivity-rf"]
    rf_reused = max((int(e.get("reuse_after_idle_count") or 0) for e in rf_events), default=0)
    before_close = runtime.get("runtime_snapshot_before_close") or []
    return {
        "native_denied": [c for c in base_check.get("checks", []) if not c.get("admitted")],
        "runtime_denied_count": rt_check.get("refusal_count"),
        "runtime_cost_after_release_before_literature": [
            {"tool": c["tool"], "bge": c.get("observed_bge_resident"),
             "rf": c.get("observed_rf_running"), "cost_mib": c.get("projected_mib")}
            for c in rt_check["checks"] if c["tool"].startswith("search_")],
        "rf_reuse_count": rf_reused,
        "rf_demotion_before_close": [
            s for s in before_close if (
                isinstance(s, dict) and
                (s.get("resource_id") == "process:na-conductivity-rf"
                 or s.get("capability_id") == "process:na-conductivity-rf"))
        ],
        "runtime_completed_without_cap_denial": rt_check.get("refusal_count") == 0,
        "native_budget_mib": base_check["budget_mib"],
        "runtime_budget_mib": rt_check["budget_mib"],
        "same_budget": base_check["budget_mib"] == rt_check["budget_mib"]
    }


def inspect(directory: Path) -> dict:
    manifest = read_json(directory / "manifest.json")
    n = read_json(directory / "T1-r1-native/result.json")
    r = read_json(directory / "T1-r1-runtime/result.json")
    expected = archived()
    model_path = Path(manifest["settings"]["conductivity_model_path"])
    if digest(model_path) != manifest["settings"]["conductivity_model_sha256"]:
        raise ValueError("Model hash mismatch")
    if digest(DATA / "split.json") != manifest["tasks"]["T1"]["raw_sha256"]:
        raise ValueError("Split hash mismatch")
    pred = prediction_audit(r, expected)
    ranking = rank_audit(expected)
    report = report_audit(r, expected)
    mech = mechanisms(directory, n, r)
    native_error = n.get("error_message") or ""
    if not native_error:
        native_error = str(n.get("failure_reasons") or "")
    checks = {"frozen_model": True, "frozen_split": True,
              "native_baseline_capacity_denied": "COMMON_CAPACITY_DENIED" in native_error,
              "runtime_agent_ok": r.get("ok") is True,
              "runtime_task_status": (r.get("task_artifact") or {}).get("task_status"),
              "prediction_audit": pred, "ranking": ranking, "report_audit": report,
              "resource_observations": mech, "native_duration_s": n.get("run_duration_seconds"),
              "runtime_duration_s": r.get("run_duration_seconds"),
              "sample_size_per_arm": 1,
              "invalid_claims": [
                 "Scientific accuracy increase due to runtime",
                 "Observed end-to-end speedup",
                 "Statistically estimated improvement in completion probability"
              ]}
    checks["system_plus_numeric_workflow_pass"] = bool(
        checks["native_baseline_capacity_denied"] and checks["runtime_agent_ok"]
        and checks["prediction_audit"]["raw_and_artifact_points_both_complete"]
        and checks["resource_observations"]["runtime_completed_without_cap_denial"]
        and checks["resource_observations"]["same_budget"])
    checks["full_user_science_requirements_pass"] = bool(
        checks["system_plus_numeric_workflow_pass"] and report["full_report_quality_pass"])
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=str(DEFAULT))
    args = parser.parse_args()
    print(json.dumps(inspect(Path(args.experiment)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
