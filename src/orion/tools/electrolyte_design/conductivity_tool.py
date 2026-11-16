"""Workflow-facing sodium-electrolyte conductivity prediction tool.



The predictor is intentionally advisory.  It annotates formulation candidates with a

bounded ML property prediction but does not convert a model estimate into literature

``evidence`` and does not satisfy the evidence-backed hard-constraint gate by itself.



A candidate is eligible only when its chemistry lies inside the model family

(NaPF6 with EC/EMC/PC and no additives) and the caller supplies an exact molar

composition representation.  Salt concentration plus an unlabeled solvent ratio is not

silently converted to mole fractions because the existing literature extractor does not

preserve whether a ratio is v/v, w/w, or molar.

"""



from __future__ import annotations



import os

import tempfile

from pathlib import Path

from threading import Lock

from typing import Any



import joblib



from langchain_core.tools import tool

from langchain_core.runnables import RunnableConfig



from .conductivity_ml import (

    ConductivityFeatures,

    OutOfDomainError,

    SUPPORTED_SALT,

    SUPPORTED_SOLVENTS,

    SodiumConductivityRandomForest,

    download_upstream_na_frames,

    features_from_mmol,

    fit_public_na_model,

    load_upstream_na_frames,

    temperature_c_to_inverse_temperature,

)





MODEL_ID = "na-conductivity-rf-v1"

CONDUCTIVITY_PROCESS_RESOURCE_ID = "process:na-conductivity-rf"

_MODEL: Any | None = None

_MODEL_LOCK = Lock()

_SERIALIZE_LOCK = Lock()





def _fit_public_model_instance() -> SodiumConductivityRandomForest:

    """Build the frozen public-data RF without mutating the process-local cache."""



    local_zip = os.environ.get("ORION_NA_CONDUCTIVITY_DATA_ZIP", "").strip()

    frames = (

        load_upstream_na_frames(Path(local_zip))

        if local_zip

        else download_upstream_na_frames()

    )

    model, _ = fit_public_na_model(frames)

    return model





def _get_public_model() -> Any:

    """Lazily construct the in-process fallback predictor."""



    global _MODEL

    if _MODEL is not None:

        return _MODEL

    with _MODEL_LOCK:

        if _MODEL is None:

            explicit = os.environ.get("ORION_NA_CONDUCTIVITY_MODEL_PATH", str(Path(__file__).resolve().parents[4] / "data" / "cwsr_conductivity_rf_formal_v1.joblib")).strip()

            if explicit:

                # 与进程工具使用同一离线模型，避免把重新训练误算为运行时收益。

                from .prediction_records import artifact_model_reference

                path = Path(explicit).expanduser().resolve()

                reference = artifact_model_reference(path)

                loaded = joblib.load(path)

                if not isinstance(loaded, SodiumConductivityRandomForest) or not loaded.fitted:

                    raise TypeError("serialized model is not a fitted SodiumConductivityRandomForest")

                loaded._orion_model_reference = reference

                _MODEL = loaded

            else:

                _MODEL = _fit_public_model_instance()

        return _MODEL





def ensure_serialized_public_model() -> Path:

    """Return a worker-loadable RF artifact stored outside the Git workspace."""



    explicit = os.environ.get("ORION_NA_CONDUCTIVITY_MODEL_PATH", str(Path(__file__).resolve().parents[4] / "data" / "cwsr_conductivity_rf_formal_v1.joblib")).strip()

    if explicit:

        path = Path(explicit).expanduser().resolve()

        if not path.exists():

            raise FileNotFoundError(path)

        return path



    cache_root = os.environ.get("ORION_RUNTIME_CACHE_DIR", "").strip()

    if cache_root:

        base = Path(cache_root).expanduser()

    else:

        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()

        base = Path(local_appdata) / "Orion" if local_appdata else Path(tempfile.gettempdir()) / "orion"

    path = (base / "runtime-models" / "na_conductivity_rf_v1.joblib").resolve()

    if path.exists():

        return path

    with _SERIALIZE_LOCK:

        if path.exists():

            return path

        path.parent.mkdir(parents=True, exist_ok=True)

        model = _fit_public_model_instance()

        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")

        try:

            joblib.dump(model, temporary, compress=0)

            os.replace(temporary, path)

        finally:

            if temporary.exists():

                temporary.unlink(missing_ok=True)

    return path





def _normalized_component_set(candidate: dict[str, Any]) -> tuple[str, set[str], list[str]]:

    salt = str(candidate.get("salt") or "").replace(" ", "")

    solvents = {

        str(item).strip().upper()

        for item in candidate.get("solvents") or []

        if str(item).strip()

    }

    additives = [

        str(item).strip()

        for item in candidate.get("additives") or []

        if str(item).strip()

    ]

    return salt, solvents, additives





def _prediction_record(

    *,

    status: str,

    temperature_c: float,

    reason: str = "",

    prediction: Any | None = None,

) -> dict[str, Any]:

    record: dict[str, Any] = {

        "property_name": "ionic_conductivity",

        "status": status,

        "numeric_value": None,

        "unit": "mS/cm",

        "temperature_c": float(temperature_c),

        "model_id": MODEL_ID,

        "source_data_doi": "10.5281/zenodo.18429013",

        "ensemble_std_log10": None,

        "estimated_factor_1sigma": None,

        "features": {},

        "reason": reason,

    }

    if prediction is not None:

        record.update(

            {

                "numeric_value": float(prediction.conductivity_ms_cm),

                "ensemble_std_log10": float(prediction.ensemble_std_log10),

                "estimated_factor_1sigma": float(prediction.estimated_factor_1sigma),

                "features": {

                    "inverse_temperature": float(prediction.features.inverse_temperature),

                    "x_NaSalt": float(prediction.features.x_NaSalt),

                    "x_EC": float(prediction.features.x_EC),

                    "x_PC": float(prediction.features.x_PC),

                },

            }

        )

    return record





def _explicit_features(

    candidate: dict[str, Any],

    *,

    temperature_c: float,

    override: dict[str, Any] | None,

) -> ConductivityFeatures | None:

    """核对组成表示一致性后，返回由明确摩尔组成换算的模型特征。"""

    from .conductivity_inputs import resolve_conductivity_inputs



    return resolve_conductivity_inputs(candidate, temperature_c=temperature_c, override=override)





def annotate_candidates_with_conductivity(

    request: dict[str, Any],

    *,

    model: SodiumConductivityRandomForest,

) -> list[dict[str, Any]]:

    """Annotate candidates while keeping ML predictions separate from evidence."""



    candidates = [

        dict(item) for item in request.get("candidates") or [] if isinstance(item, dict)

    ]

    temperature_c = float(request["temperature_c"])

    overrides = request.get("composition_inputs") or {}

    overrides = overrides if isinstance(overrides, dict) else {}

    output: list[dict[str, Any]] = []



    for candidate in candidates:

        candidate_id = str(candidate.get("candidate_id") or "").strip()

        salt, solvents, additives = _normalized_component_set(candidate)

        if salt.casefold() != SUPPORTED_SALT.casefold():

            record = _prediction_record(

                status="out_of_domain",

                temperature_c=temperature_c,

                reason=f"model supports {SUPPORTED_SALT}; candidate salt is {salt or 'unknown'}",

            )

        elif not solvents or not solvents <= set(SUPPORTED_SOLVENTS):

            record = _prediction_record(

                status="out_of_domain",

                temperature_c=temperature_c,

                reason=(

                    "model supports only EC/EMC/PC solvent compositions; candidate solvents="

                    + ",".join(sorted(solvents))

                ),

            )

        elif additives:

            record = _prediction_record(

                status="out_of_domain",

                temperature_c=temperature_c,

                reason="additive-containing formulations are outside the frozen model domain",

            )

        else:

            try:

                features = _explicit_features(

                    candidate,

                    temperature_c=temperature_c,

                    override=(

                        overrides.get(candidate_id)

                        if isinstance(overrides.get(candidate_id), dict)

                        else None

                    ),

                )

            except (TypeError, ValueError) as exc:

                features = None

                feature_error = str(exc)

            else:

                feature_error = ""



            if features is None:

                record = _prediction_record(

                    status="insufficient_composition",

                    temperature_c=temperature_c,

                    reason=(

                        feature_error

                        or "exact molar composition is unavailable; concentration plus an unlabeled solvent ratio is not converted implicitly"

                    ),

                )

            else:

                try:

                    prediction = model.predict_features(features)

                except OutOfDomainError as exc:

                    record = _prediction_record(

                        status="out_of_domain",

                        temperature_c=temperature_c,

                        reason=str(exc),

                    )

                else:

                    record = _prediction_record(

                        status="predicted",

                        temperature_c=temperature_c,

                        prediction=prediction,

                    )



        previous = [

            item

            for item in candidate.get("model_predictions") or []

            if not (

                isinstance(item, dict)

                and item.get("property_name") == "ionic_conductivity"

                and item.get("model_id") == MODEL_ID

                and item.get("temperature_c") == temperature_c

            )

        ]

        candidate["model_predictions"] = [*previous, record]

        output.append(candidate)

    return output





@tool

def predict_formulation_conductivity(request: dict[str, Any], config: RunnableConfig, callbacks=None) -> list[dict[str, Any]] | dict[str, Any]:

    """用轻量 RF 为可映射的 NaPF6–EC/EMC/PC 候选估计离子电导率。



    request 必须包含 ``candidates`` 与 ``temperature_c``。只有候选已有明确摩尔组成，

    或 ``composition_inputs[candidate_id]`` 显式提供 ``component_mmol``、``molar_ratios``

    或明确的模型特征时才预测。优先提供完整 ``molar_ratios``，由工具换算。

    模型特征定义为 x_NaSalt=NaPF6/(NaPF6+EC+EMC+PC)、x_EC=EC/(EC+EMC)、

    x_PC=PC/(EC+EMC+PC)，后两项不是配方中 EC、PC 的全体摩尔份额。

    同时给出完整组成与模型特征时，两种表示必须一致，否则返回组成错误而不预测。

    模型输出写入 ``model_predictions``，不写入

    文献 evidence，也不能单独满足实验事实型硬约束。域外配方和组成不足会显式返回状态。

    新预测带result_binding，保存组成/条件/模型版本与记录ID；同配方温度比较可用Skill脚本compare读取记录，不能重标旧结果。

    """



    if "temperature_c" not in request:

        raise ValueError("predict_formulation_conductivity requires temperature_c")



    from .prediction_records import request_context

    try:

        request_context(request)

    except (TypeError, ValueError) as exc:

        return {"status": "invalid_request", "error": str(exc), "recoverable": True,

                "guidance": "conditions须为JSON对象；其中温度如有声明，必须与temperature_c一致。"}



    from orion.core.langgraph_residency import get_active_runtime_resource



    resource = get_active_runtime_resource(CONDUCTIVITY_PROCESS_RESOURCE_ID)

    if resource is None:

        # 仅保留直接函数调用/旧测试的兼容路径；正常 Agent ToolNode 会先 acquire 静态依赖。

        model = _get_public_model()

    else:

        from .conductivity_process import (

            ConductivityProcessPredictor,

            SodiumConductivityMlProcessAdapter,

        )



        if not isinstance(resource, SodiumConductivityMlProcessAdapter):

            raise RuntimeError(

                "leased conductivity resource has unexpected adapter type: "

                f"{type(resource).__name__}"

            )

        model = ConductivityProcessPredictor(resource)



    from .prediction_records import bind_prediction_results, model_reference



    candidates = annotate_candidates_with_conductivity(request, model=model)

    reference = model_reference(model) if any(

        c.get("model_predictions") and c["model_predictions"][-1].get("status") == "predicted"

        for c in candidates) else {}

    run_id = getattr(callbacks, "parent_run_id", None)

    thread_id = (config.get("configurable") or {}).get("thread_id")

    return bind_prediction_results(candidates, request, reference=reference,

        tool_run_id=str(run_id) if run_id else None, thread_id=str(thread_id) if thread_id else None)

