"""候选批次文件化与批量点预测评价；摘要不冒充全空间或电池性能最优。"""

from __future__ import annotations



from collections import Counter

from copy import deepcopy

import hashlib

import json

import math

from pathlib import Path

import re

from typing import Any



from tools.file_operate import safe_path



SCHEMAS = {"candidates": "candidate_batch_v1", "evaluations": "candidate_batch_evaluation_v1"}





class BatchRequestError(ValueError):

    """可由调用者修正的批处理参数错误。"""





def _integer(value, name, maximum):

    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:

        raise BatchRequestError(f"{name} must be an integer in 1..{maximum}")

    return value





def _write(kind: str, payload: dict) -> str:

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    identifier = hashlib.sha256(raw).hexdigest()

    path = safe_path(f"reports/candidate_{kind}/{identifier}.json")

    path.parent.mkdir(parents=True, exist_ok=True)

    try:

        with path.open("xb") as stream:

            stream.write(raw)

    except FileExistsError:

        if path.read_bytes() != raw:

            raise RuntimeError("candidate artifact content conflict")

    return identifier





def _read(kind: str, identifier: str) -> dict:

    if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{64}", identifier):

        raise BatchRequestError("use the returned 64-character batch/evaluation identifier, not a path")

    raw = safe_path(f"reports/candidate_{kind}/{identifier}.json").read_bytes()

    if hashlib.sha256(raw).hexdigest() != identifier:

        raise BatchRequestError("candidate artifact failed content verification")

    payload = json.loads(raw)

    if payload.get("schema") != SCHEMAS[kind]:

        raise BatchRequestError("candidate artifact schema mismatch")

    return payload





def _error(exc) -> dict:

    return {"status": "invalid_request", "error": str(exc), "recoverable": True}





def save_candidate_batch(result: dict, request: dict) -> dict:

    """完整保存生成结果，只返回小预览；预览不是筛选推荐。"""

    candidates = result["candidates"]

    ids = [c.get("candidate_id") for c in candidates]

    if not candidates or any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):

        raise BatchRequestError("generated candidate ids must be non-empty and unique")

    source = Path(__file__).with_name("candidate_space.py").read_bytes()

    identifier = _write("candidates", {"schema": SCHEMAS["candidates"], "generation_request": request,

        "generator_source_sha256": hashlib.sha256(source).hexdigest(), "result": result})

    return {"status": "ok", "batch_id": identifier,

        "artifact_path": f"reports/candidate_candidates/{identifier}.json",

        "generated_count": len(candidates), "evaluated_count": 0,

        "space": result["space"], "warnings": result.get("warnings", []),

        "preview": [{"candidate_ref": f"{identifier}:{c['candidate_id']}",

                     **{k: c[k] for k in ("candidate_id", "component_mmol", "solvent_ratio_basis") if k in c}}

                    for c in candidates[:2]],

        "preview_note": "仅生成顺序的前两项预览，未预测、未排名；用batch_id进行评价或分页读取。"}





def evaluate_candidate_batch(request: dict, *, config=None) -> dict:

    """对全部或显式预算限定的候选调用既有RF，记录每项状态再按点预测排序。"""

    try:

        if request.get("objective") != "maximise_predicted_conductivity":

            raise BatchRequestError("objective must explicitly be maximise_predicted_conductivity")

        batch = _read("candidates", request["batch_id"])

        candidates = batch["result"]["candidates"]

        top_k = _integer(request.get("top_k", 3), "top_k", 20)

        limit = _integer(request.get("max_evaluations", len(candidates)), "max_evaluations", 5000)

        from .prediction_records import request_context

        context = request_context(request)

    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:

        return _error(exc)

    selected = deepcopy(candidates[:limit])

    from .conductivity_tool import predict_formulation_conductivity

    results = predict_formulation_conductivity.invoke({"request": {

        "candidates": selected, "temperature_c": request["temperature_c"], "conditions": context}},

        **({"config": config} if config is not None else {}))

    if isinstance(results, dict):

        return results

    if (len(results) != len(selected) or

            [c.get("candidate_id") for c in results] != [c["candidate_id"] for c in selected]):

        raise RuntimeError("prediction backend did not return the requested candidate sequence")

    rows = []

    for candidate in results:

        prediction = candidate["model_predictions"][-1]

        if prediction["status"] == "predicted":

            value = prediction["numeric_value"]

            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or not prediction.get("result_binding"):

                raise RuntimeError("prediction backend returned an invalid or unbound numeric result")

        rows.append({"candidate_id": candidate["candidate_id"],

                     "candidate_ref": f"{request['batch_id']}:{candidate['candidate_id']}",

                     "component_mmol": candidate["component_mmol"],

                     "solvent_ratio_basis": "molar", "prediction": prediction})

    predicted = [row for row in rows if row["prediction"]["status"] == "predicted"]

    predicted.sort(key=lambda r: (-r["prediction"]["numeric_value"], r["candidate_id"]))

    status_counts = dict(Counter(row["prediction"]["status"] for row in rows))

    coverage = {"generated_count": len(candidates), "attempted_count": len(rows),

        "predicted_count": len(predicted), "unevaluated_count": len(candidates)-len(rows),

        "attempted_all_generated": len(rows) == len(candidates), "prediction_status_counts": status_counts}

    boundary = predicted[min(top_k, len(predicted))-1]["prediction"]["numeric_value"] if predicted else None

    omitted_ties = sum(math.isclose(r["prediction"]["numeric_value"], boundary, rel_tol=1e-12, abs_tol=1e-12)

                       for r in predicted[top_k:]) if boundary is not None else 0

    identifier = _write("evaluations", {"schema": SCHEMAS["evaluations"], "batch_id": request["batch_id"],

        "request": request, "coverage": coverage, "rows": rows,

        "ranked_predicted_candidate_ids": [r["candidate_id"] for r in predicted]})

    best = []

    for row in predicted[:top_k]:

        amounts = row["component_mmol"]

        solvent_total = sum(amounts.get(name, 0) for name in ("EC", "EMC", "PC"))

        total = sum(amounts.values())

        binding = row["prediction"]["result_binding"]

        best.append({"candidate_id": row["candidate_id"], "candidate_ref": row["candidate_ref"], "component_mmol": amounts,

            "solvent_ratio_basis": "molar", "conductivity_ms_cm": row["prediction"]["numeric_value"],

            "all_component_mole_fractions": {name: amount/total for name, amount in amounts.items()},

            "solvent_molar_shares": {name: amounts.get(name, 0)/solvent_total for name in ("EC", "EMC", "PC")},

            "model_features": row["prediction"].get("features", {}),

            "ensemble_std_log10": row["prediction"].get("ensemble_std_log10"),

            "estimated_factor_1sigma": row["prediction"].get("estimated_factor_1sigma"),

            "composition_id": binding.get("composition_id"), "condition_id": binding.get("condition_id"),

            "record_id": binding["record_id"]})

    return {"status": "evaluated", "evaluation_id": identifier, "batch_id": request["batch_id"],

        "artifact_path": f"reports/candidate_evaluations/{identifier}.json", "coverage": coverage,

        "temperature_c": float(request["temperature_c"]), "objective": request["objective"],

        "selection_scope": "predicted_candidates_in_actual_evaluated_sample",

        "identity_rules": "candidate_id仅为批次内标签，跨批可重名；candidate_ref=批次ID:标签。composition_id标识模型摩尔组成；record_id标识一次计算，改温重算后应不同。温度比较核对composition_id和模型版本，不要求两个record_id相同。",

        "top_candidates": best, "omitted_candidates_tied_at_boundary": omitted_ties,

        "composition_definitions": {"all_component_mole_fractions": "组分/(NaPF6+EC+EMC+PC)",

            "solvent_molar_shares": "溶剂/(EC+EMC+PC)",

            "model_features": "x_NaSalt=NaPF6/(NaPF6+EC+EMC+PC); x_EC=EC/(EC+EMC); x_PC=PC/(EC+EMC+PC)"},

        "uncertainty_note": "树间离散未做实验误差校准，不能换成有覆盖率保证的±误差或据区间重叠判定统计显著性。",

        "warnings": ([] if coverage["attempted_all_generated"] else [

            "预算只允许评价生成顺序的前一部分，不能声称已筛完全部采样。"]),

        "limitations": "仅按点预测排序；并列按ID稳定展示，不表示显著优劣。未评价可配制性或电池性能，不是连续全空间最优。"}





def inspect_candidate_artifact(request: dict) -> dict:

    """按需分页查看完整候选或逐项评价，避免把整批结果放进上下文。"""

    try:

        if ("batch_id" in request) == ("evaluation_id" in request):

            raise BatchRequestError("provide exactly one of batch_id or evaluation_id")

        limit = _integer(request.get("limit", 3), "limit", 20)

        offset = request.get("offset", 0)

        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:

            raise BatchRequestError("offset must be a non-negative integer")

        kind, key = ("candidates", "batch_id") if "batch_id" in request else ("evaluations", "evaluation_id")

        artifact = _read(kind, request[key])

        rows = artifact["result"]["candidates"] if kind == "candidates" else artifact["rows"]

        order = request.get("order", "generation" if kind == "candidates" else "predicted_rank")

        if order not in {"generation", "predicted_rank"} or (kind == "candidates" and order != "generation"):

            raise BatchRequestError("order must be generation, or predicted_rank for an evaluation")

        if order == "predicted_rank":

            by_id = {r["candidate_id"]: r for r in rows}

            ranked = artifact["ranked_predicted_candidate_ids"]

            ranked_set = set(ranked)

            rows = [by_id[cid] for cid in ranked] + [r for r in rows if r["candidate_id"] not in ranked_set]

    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:

        return _error(exc)

    batch_id = request[key] if kind == "candidates" else artifact["batch_id"]

    items = [{**row, "candidate_ref": f"{batch_id}:{row['candidate_id']}"} for row in rows[offset:offset+limit]]

    return {"status": "ok", key: request[key], "batch_id": batch_id,

            "total": len(rows), "offset": offset, "order": order,

            "items": items, "has_more": offset+limit < len(rows)}

