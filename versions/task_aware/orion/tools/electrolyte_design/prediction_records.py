"""保存输运预测的实际输入与结果；同配方温度比较只读取已有计算记录。"""


from __future__ import annotations





from copy import deepcopy


import hashlib


import json


import math


from pathlib import Path


import re


from typing import Any


from uuid import uuid4





from tools.file_operate import safe_path





SCHEMA = "transport_prediction_record_v1"


RECORD_DIR = "reports/transport_predictions"








def _json(value: Any) -> str:


    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)








def _digest(value: Any) -> str:


    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()








def artifact_model_reference(path: Path) -> dict[str, str]:


    """在实际模型加载处记录工件版本，不从Agent参数接受版本声明。"""


    with path.open("rb") as stream:


        artifact = hashlib.file_digest(stream, "sha256").hexdigest()


    implementation = Path(__file__).with_name("conductivity_ml.py").read_bytes()


    return {"artifact_sha256": artifact, "implementation_sha256": hashlib.sha256(implementation).hexdigest()}








def model_reference(model: Any) -> dict[str, str]:


    reference = getattr(model, "_orion_model_reference", None)


    if reference:


        return dict(reference)


    adapter = getattr(model, "adapter", None)


    if adapter is not None:


        # 训练数据启动的历史worker没有可核实的工件版本；保留未知，比较时不放行。


        return dict(getattr(adapter, "prediction_model_reference", None) or {})


    import joblib


    implementation = Path(__file__).with_name("conductivity_ml.py").read_bytes()


    return {"joblib_state_sha1": joblib.hash(model),


            "implementation_sha256": hashlib.sha256(implementation).hexdigest()}








def composition_identity(features: dict[str, float]) -> dict[str, Any]:


    """仅标识该模型实际接受的摩尔组成，不认证原文提取或体积到摩尔换算。"""


    salt, ec, pc = (float(features[k]) for k in ("x_NaSalt", "x_EC", "x_PC"))


    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (salt, ec, pc)):


        raise ValueError("prediction composition features must be finite fractions")


    amounts = {"NaPF6": salt, "EC": (1-salt)*(1-pc)*ec,


               "EMC": (1-salt)*(1-pc)*(1-ec), "PC": (1-salt)*pc}


    # 忽略比例缩放和浮点末位差；12位有效数字是本版本的身份分辨率。


    return {"basis": "mole_fraction_of_all_components", "components": {


        name: format(value, ".12g") for name, value in amounts.items()}}








def request_context(request: dict[str, Any]) -> dict[str, Any]:


    """在调用预测前核对可选条件，避免算完才因元数据冲突退出。"""


    context = deepcopy(request.get("conditions"))


    context = {} if context is None else context


    if not isinstance(context, dict):


        raise ValueError("conditions must be an object")


    temperature = float(request["temperature_c"])


    declared_t = float(context.pop("temperature_c", temperature))


    if not math.isfinite(temperature) or declared_t != temperature:


        raise ValueError("conditions.temperature_c conflicts with prediction temperature")


    _json(context)


    return context








def bind_prediction_results(candidates: list[dict[str, Any]], request: dict[str, Any], *,


                            reference: dict[str, str], tool_run_id: str | None = None,


                            thread_id: str | None = None) -> list[dict[str, Any]]:


    """只绑定这次实际新算的末条结果；历史记录不被重新登记为当前计算。"""


    result = deepcopy(candidates)


    execution_id = uuid4().hex


    context = request_context(request)


    for candidate in result:


        predictions = candidate.get("model_predictions") or []


        if not predictions or predictions[-1].get("status") != "predicted":


            continue


        prediction = predictions[-1]


        identity = composition_identity(prediction["features"])


        conditions = {"temperature_c": prediction["temperature_c"], "declared_context": context}


        # 只保存输入快照，不递归复制候选所携带的历史预测和文献大块。


        snapshot = {k: deepcopy(candidate[k]) for k in (


            "salt", "solvents", "additives", "component_mmol", "molar_ratios",


            "solvent_ratio_basis", "conductivity_features", "composition_inputs") if k in candidate}


        record = {"schema": SCHEMA, "execution_id": execution_id, "tool_run_id": tool_run_id,


                  "thread_id": thread_id, "candidate_id": candidate.get("candidate_id", ""),


                  "candidate_input": snapshot,


                  "composition_input": deepcopy((request.get("composition_inputs") or {}).get(candidate.get("candidate_id"), {})),


                  "composition": identity, "composition_id": _digest(identity),


                  "conditions": conditions, "condition_id": _digest(conditions),


                  "model_reference": reference, "prediction": deepcopy(prediction)}


        raw = _json(record)


        record_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()


        path = safe_path(f"{RECORD_DIR}/{record_id}.json")


        path.parent.mkdir(parents=True, exist_ok=True)


        # 每次execution_id不同；不覆盖既有计算记录。


        with path.open("x", encoding="utf-8", newline="\n") as stream:


            stream.write(raw)


        prediction["result_binding"] = {"schema": SCHEMA, "record_id": record_id,


            "record_path": f"{RECORD_DIR}/{record_id}.json", "execution_id": execution_id,


            "tool_run_id": tool_run_id, "thread_id": thread_id,


            "composition_id": record["composition_id"], "condition_id": record["condition_id"],


            "composition": identity, "conditions": conditions, "model_reference": reference,


            "scope": "model_input_identity_not_source_or_battery_validation"}


    return result








def read_prediction_record(record_id: str) -> dict[str, Any]:


    if not isinstance(record_id, str) or not re.fullmatch(r"[0-9a-f]{64}", record_id):


        raise ValueError("record_id must be a returned 64-character record identifier")


    raw = safe_path(f"{RECORD_DIR}/{record_id}.json").read_bytes()


    if hashlib.sha256(raw).hexdigest() != record_id:


        raise ValueError("saved prediction record failed content verification")


    record = json.loads(raw)


    if record.get("schema") != SCHEMA:


        raise ValueError("unsupported prediction record schema")


    return record








def compare_saved_predictions(request: dict[str, Any]) -> dict[str, Any]:


    """计算同组成、同模型且其它声明条件不变的温度差和预测比值。"""


    try:


        left_id, right_id = request["left_record_id"], request["right_record_id"]


        left, right = read_prediction_record(left_id), read_prediction_record(right_id)


    except (KeyError, ValueError, FileNotFoundError) as exc:


        return {"status": "invalid_request", "error": str(exc), "recoverable": True,


                "guidance": "使用预测返回的record_id；不要在比较请求里补写或重标预测数值。"}


    issues = []


    if left["composition_id"] != right["composition_id"]:


        issues.append("composition_changed")


    if not left["model_reference"] or not right["model_reference"]:


        issues.append("model_version_unknown")


    elif left["model_reference"] != right["model_reference"]:


        issues.append("model_version_changed")


    a, b = left["prediction"], right["prediction"]


    if (a["model_id"], a["property_name"], a["unit"]) != (b["model_id"], b["property_name"], b["unit"]):


        issues.append("prediction_definition_changed")


    if left["conditions"]["declared_context"] != right["conditions"]["declared_context"]:


        issues.append("other_declared_conditions_changed")


    if a["temperature_c"] == b["temperature_c"]:


        issues.append("temperature_not_changed")


    summary = [{"record_id": rid, "candidate_id": r["candidate_id"], "composition_id": r["composition_id"],


                "temperature_c": r["prediction"]["temperature_c"],


                "numeric_value": r["prediction"]["numeric_value"], "unit": r["prediction"]["unit"]}


               for rid, r in ((left_id, left), (right_id, right))]


    if issues:


        return {"status": "not_comparable", "issues": issues, "records": summary,


                "guidance": "保留各自结果；若要研究同配方温度变化，应对同一组成在所需温度重新计算。"}


    value_a, value_b = a["numeric_value"], b["numeric_value"]


    return {"status": "compared", "scope": "same_composition_model_temperature_change",


            "records": summary, "difference_right_minus_left": value_b-value_a,


            "ratio_right_over_left": value_b/value_a if value_a else None,


            "unit": a["unit"],


            "limitations": "只比较同模型输入；未核实的条件保持未知，不是实测温度响应或校准置信区间。"}


