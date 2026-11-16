"""Fresh v2 development runtime adapter.

Import 本模块不会调用模型或写任何实验状态。真实 provider invocation 只发生在显式函数调用时。
本 runtime 覆盖 neutral health check、native-JSON proposer，以及最多 3 个 LLM turns 的
first-action ORION episode。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from orion.benchmark.epistemic_promotion_live import _message_text, temporary_environment
from orion.benchmark.epistemic_promotion_transport_v3 import invoke_with_preoutput_transport_retry
from orion.benchmark.self_evolution_fresh_formal_runtime import (
    FreshFormalFutureEvidencePacket,
    build_frozen_future_probe_tools,
    development_source_packet,
)
from orion.benchmark.self_evolution_fresh_v2_admission import FreshV2ScientificFamilyContract
from orion.benchmark.self_evolution_fresh_v2_runtime_health import (
    FreshV2RuntimeHealthReceipt,
    run_runtime_health_probe,
    runtime_health_messages,
    sanitize_provider_exception,
)
from orion.benchmark.self_evolution_fresh_v2_scope import FreshPersistentUpdateProposalV2
from orion.benchmark.self_evolution_fresh_v2_treatment_delivery import (
    candidate_sha256_v2,
    mount_v2_candidate_skill,
)
from orion.core.context.manager import ContextManager


ROOT = Path(__file__).resolve().parent.parent
# 跨模型复现（如 Study 3 cross-model trial）通过 ORION_V2_RUNTIME_CONTRACT_PATH
# 选择替代合同；未设置时与历史默认逐字节一致（V0 deepseek 合同）。
CONTRACT_PATH = Path(
    os.environ.get("ORION_V2_RUNTIME_CONTRACT_PATH")
    or (ROOT / "benchmark/self_evolution_fresh_v2/V2_DEVELOPMENT_RUNTIME_CONTRACT_V0_2026-08-18.json")
)
TASKS_PATH = ROOT / "benchmark/self_evolution_fresh_v2/V2_DEVELOPMENT_INSTRUMENT_TASKS_V0_2026-08-18.json"
SCIENTIFIC_CONTRACTS_PATH = ROOT / "benchmark/self_evolution_fresh_v2/V2_DEVELOPMENT_SCIENTIFIC_CONTRACTS_V0_2026-08-18.json"
PROPOSER_PROMPT_PATH = ROOT / "benchmark/self_evolution_fresh_v2/V2_COMMON_PROPOSER_PROMPT_V0_2026-08-18.md"
BASELINE_SKILL_PATH = ROOT / "skills/electrolyte_design/SKILL.md"
JSON_RESPONSE_FORMAT = {"type": "json_object"}


class FreshV2DevelopmentRuntimeError(RuntimeError):
    pass


class FreshV2FirstActionBudgetExceeded(RuntimeError):
    pass


class FreshV2DevelopmentRuntimeBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    temperature: float
    top_p: float
    max_output_tokens: int
    hidden_sdk_retries: int
    endpoint_sha256: str
    api_key_present: bool


class FreshV2DirectCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str
    transport_retry_count: int


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _runtime_contract() -> dict[str, Any]:
    return _load_json(CONTRACT_PATH)


def expected_v2_runtime_environment() -> dict[str, str]:
    return dict(_runtime_contract()["explicit_runtime_override"])


def resolve_v2_runtime_binding() -> FreshV2DevelopmentRuntimeBinding:
    """零调用验证 provider/model/sampling/endpoint 绑定。"""

    import orion.env_utils as env_utils

    contract = _runtime_contract()
    override = contract["explicit_runtime_override"]
    for key, expected in override.items():
        if (os.environ.get(key) or "").strip() != str(expected):
            raise FreshV2DevelopmentRuntimeError(f"v2 runtime env drift: {key}")

    # 端点/密钥默认走历史 DeepSeek 链（零漂移）；合同显式声明 endpoint_env /
    # api_key_env 时（跨模型复现合同）改从 llms 已解析的主模型端点/密钥取值，
    # 保证 binding 与 _direct_model 实际使用的端点一致。
    endpoint_env = str(contract.get("endpoint_env") or "").strip()
    api_key_env = str(contract.get("api_key_env") or "").strip()
    if endpoint_env or api_key_env:
        import orion.llms as llms

        endpoint = str(llms._main_base_url or "").strip().rstrip("/")
        api_key = str(llms._agent_api_key or "")
    else:
        endpoint = str(
            getattr(env_utils, "DEEPSEEK_BASE_URL", "")
            or os.environ.get("DEEPSEEK_BASE_URL")
            or ""
        ).strip().rstrip("/")
        api_key = str(
            getattr(env_utils, "DEEPSEEK_API_KEY", "")
            or os.environ.get("DEEPSEEK_API_KEY")
            or ""
        )
    if not endpoint:
        raise FreshV2DevelopmentRuntimeError("v2 provider endpoint is not configured")
    if not api_key.strip():
        raise FreshV2DevelopmentRuntimeError("v2 provider API key is not present")
    sampling = contract["sampling"]
    return FreshV2DevelopmentRuntimeBinding(
        provider=str(contract["provider"]),
        model=str(contract["model"]),
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        max_output_tokens=int(sampling["max_output_tokens"]),
        hidden_sdk_retries=int(sampling["hidden_sdk_retries"]),
        endpoint_sha256=_sha256_text(endpoint),
        api_key_present=True,
    )


def _direct_model(*, max_tokens: int, request_timeout: int, json_mode: bool):
    binding = resolve_v2_runtime_binding()
    from langchain.chat_models import init_chat_model
    import orion.llms as llms

    if str(llms._agent_provider) != binding.provider:
        raise FreshV2DevelopmentRuntimeError("v2 resolved provider drift")
    if str(llms._agent_model) != binding.model:
        raise FreshV2DevelopmentRuntimeError("v2 resolved model drift")
    if float(llms._agent_temperature) != binding.temperature:
        raise FreshV2DevelopmentRuntimeError("v2 resolved temperature drift")
    if float(llms._agent_top_p) != binding.top_p:
        raise FreshV2DevelopmentRuntimeError("v2 resolved top_p drift")
    if int(getattr(llms.main_model_nonstream, "max_retries", -1)) != binding.hidden_sdk_retries:
        raise FreshV2DevelopmentRuntimeError("v2 SDK retry drift")

    model = init_chat_model(
        model=binding.model,
        model_provider=binding.provider,
        base_url=llms._main_base_url,
        api_key=llms._agent_api_key,
        max_tokens=max_tokens,
        temperature=binding.temperature,
        top_p=binding.top_p,
        request_timeout=request_timeout,
        max_retries=binding.hidden_sdk_retries,
        extra_body=llms._agent_extra_body,
    )
    if not json_mode:
        return model
    bound = model.bind(response_format=JSON_RESPONSE_FORMAT)
    kwargs = getattr(bound, "kwargs", None)
    if not isinstance(kwargs, dict) or kwargs.get("response_format") != JSON_RESPONSE_FORMAT:
        raise FreshV2DevelopmentRuntimeError("v2 provider-native JSON mode not bound")
    return bound


def invoke_v2_runtime_health_once() -> FreshV2RuntimeHealthReceipt:
    """执行一次 task-independent health check；任何 pre-lock runtime/provider failure 都 fail closed。"""

    health = _runtime_contract()["health_check"]
    try:
        model = _direct_model(
            max_tokens=int(health["max_tokens"]),
            request_timeout=int(health["request_timeout_seconds"]),
            json_mode=False,
        )
    except Exception as exc:
        safe = sanitize_provider_exception(exc)
        return FreshV2RuntimeHealthReceipt(
            passed=False,
            response_observed=False,
            sanitized_error_class=safe["error_class"],
            sanitized_http_status=safe["http_status"],
            sanitized_error_code=safe["error_code"],
            reason_codes=["v2_runtime_health_setup_failed"],
        )
    system, user = runtime_health_messages()

    def invoke():
        return model.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    return run_runtime_health_probe(invoke, extract_text=_message_text)


def _family_source_mapping() -> dict[str, str]:
    tasks = _load_json(TASKS_PATH)
    return {
        str(item["family_id"]): str(item["source_family_id"])
        for item in tasks["proposer_stress_families"]
    }


def render_v2_proposer_messages(family_id: str) -> tuple[str, str, int]:
    mapping = _family_source_mapping()
    if family_id not in mapping:
        raise KeyError(f"unknown v2 development family: {family_id}")
    evidence = development_source_packet(mapping[family_id])
    if not evidence:
        raise FreshV2DevelopmentRuntimeError(f"development evidence packet missing: {family_id}")
    system = PROPOSER_PROMPT_PATH.read_text(encoding="utf-8")
    user = "Development-only evidence packet. Return one JSON object only.\n\n" + _canonical(
        {"family_id": family_id, "development_evidence": evidence}
    )
    return system, user, int(_runtime_contract()["proposer_call"]["max_tokens"])


def invoke_v2_proposer_once(*, family_id: str) -> FreshV2DirectCallResult:
    """一次 native-JSON proposer call；只允许 pre-output transport retry。"""

    system, user, max_tokens = render_v2_proposer_messages(family_id)
    config = _runtime_contract()["proposer_call"]
    model = _direct_model(
        max_tokens=max_tokens,
        request_timeout=int(config["request_timeout_seconds"]),
        json_mode=True,
    )

    def invoke():
        return model.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    response, retries = invoke_with_preoutput_transport_retry(
        invoke,
        max_transport_retries=int(config["max_transport_retries_before_response_object"]),
    )
    raw = _message_text(response)
    if not raw.strip():
        raise FreshV2DevelopmentRuntimeError("v2 proposer returned empty content")
    return FreshV2DirectCallResult(raw=raw, transport_retry_count=retries)


def parse_v2_proposer_raw(*, family_id: str, raw: str) -> FreshPersistentUpdateProposalV2:
    """严格解析 proposer JSON；不 strip fence、不 repair。"""

    proposal = FreshPersistentUpdateProposalV2.model_validate_json(raw)
    mapping = _family_source_mapping()
    allowed = {item["source_id"] for item in development_source_packet(mapping[family_id])}
    observed = set(proposal.source_evidence_ids)
    if proposal.proposal_status == "candidate" and (not observed or not observed <= allowed):
        raise ValueError("v2 proposer referenced evidence outside development packet")
    if proposal.proposal_status == "no_update" and observed:
        raise ValueError("v2 no_update may not carry source evidence IDs")
    return proposal


def v2_scientific_contract(family_id: str) -> FreshV2ScientificFamilyContract:
    payload = _load_json(SCIENTIFIC_CONTRACTS_PATH)
    return FreshV2ScientificFamilyContract.model_validate(payload["families"][family_id])


def v2_first_action_case(case_id: str) -> dict[str, Any]:
    payload = _load_json(TASKS_PATH)
    matches = [item for item in payload["first_action_cases"] if item["case_id"] == case_id]
    if len(matches) != 1:
        raise KeyError(f"unknown/non-unique first-action case: {case_id}")
    return dict(matches[0])


class _V2IsolatedContextManager(ContextManager):
    """development first-action probe 禁止自动注入 RAG/memory/reflection。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.assemble_calls = 0
        self.suppressed_evidence_calls = 0
        self.suppressed_memory_calls = 0

    async def _retrieve_evidence(self, goal: str) -> str:
        self.suppressed_evidence_calls += 1
        return ""

    async def _recall_memory(self, session_id: str, goal: str) -> str:
        self.suppressed_memory_calls += 1
        return ""

    def _load_reflections(self, goal: str, task_type: str = "design") -> str:
        return ""

    async def assemble(self, *args: Any, **kwargs: Any):
        messages = await super().assemble(*args, **kwargs)
        self.assemble_calls += 1
        rendered = "\n".join(str(getattr(message, "content", "")) for message in messages)
        if "相关证据：" in rendered or "长期记忆：" in rendered or "历史教训（动态约束）：" in rendered:
            raise FreshV2DevelopmentRuntimeError("v2 first-action context isolation failed")
        return messages


def _v2_first_action_agent_factory(*, max_llm_turns: int, turn_counter: dict[str, int]):
    empty_packet = FreshFormalFutureEvidencePacket(
        family_id="V2_DEVELOPMENT_FIRST_ACTION",
        task_role="positive_transfer",
        source_ids=[],
        chunks=[],
    )

    def factory():
        main = importlib.import_module("orion.main")
        selected = build_frozen_future_probe_tools(empty_packet)
        original_tools = main._BASE_TOOLS
        main._BASE_TOOLS = selected

        async def limited_llm_async(state, config=None):
            turn_counter["count"] = int(turn_counter.get("count", 0)) + 1
            if turn_counter["count"] > max_llm_turns:
                raise FreshV2FirstActionBudgetExceeded(
                    f"v2 first-action LLM turn budget exceeded: {max_llm_turns}"
                )
            return await main.llm_async(state, config)

        try:
            main._configure_runtime_tools()
            return main._build_graph(limited_llm_async, rebuild_memory=False)
        finally:
            main._BASE_TOOLS = original_tools

    return factory


def run_v2_first_action_probe_once(
    *,
    case: dict[str, Any],
    proposal: FreshPersistentUpdateProposalV2 | None,
    expected_baseline_skill_sha256: str,
) -> dict[str, Any]:
    """运行一个 development-only first-action episode；调用者必须已持有 semantic lock。"""

    resolve_v2_runtime_binding()
    from orion.evaluation.agent_runner import run_agent
    from orion.evaluation.models import EvaluationSample

    main = importlib.import_module("orion.main")
    import orion.llms as llms

    binding = resolve_v2_runtime_binding()
    if str(llms._agent_model) != binding.model:
        raise FreshV2DevelopmentRuntimeError("v2 first-action model drift")
    if float(llms._agent_temperature) != binding.temperature:
        raise FreshV2DevelopmentRuntimeError("v2 first-action temperature drift")
    if float(llms._agent_top_p) != binding.top_p:
        raise FreshV2DevelopmentRuntimeError("v2 first-action top_p drift")

    config = _runtime_contract()["first_action_episode"]
    sample = EvaluationSample(
        question=str(case["prompt"]),
        task_type=str(case["task_type"]),
        ground_truth="",
        expected_tools=[],
    )
    env = {
        "ORION_GROUNDING_CHECK": "0",
        "ORION_LLM_CACHE": "0",
        "ORION_LLM_STREAM": "0",
        "ORION_REFLECTION_INJECT": "0",
        "ORION_EVAL_AGENT_TIMEOUT": str(config["agent_timeout_seconds"]),
        "ORION_EVAL_LLM_STEP_TIMEOUT": str(config["llm_step_timeout_seconds"]),
        "ORION_EVAL_LLM_REQUEST_MAX_RETRIES": str(config["llm_request_max_retries"]),
        "ORION_LLM_REQUEST_MAX_RETRIES": str(config["llm_request_max_retries"]),
        "ORION_EVAL_RECURSION_LIMIT": "16",
    }

    original_context = main._context_manager
    isolated = _V2IsolatedContextManager(
        memory_service=main._memory_service,
        model_name=binding.model,
    )
    turn_counter = {"count": 0}

    with tempfile.TemporaryDirectory(prefix=f"orion-fresh-v2-{case['case_id']}-") as tmp:
        tmp_path = Path(tmp)
        if proposal is not None:
            store = tmp_path / "candidate_evolution_store"
            mount_v2_candidate_skill(
                store,
                proposal,
                expected_baseline_skill_sha256=expected_baseline_skill_sha256,
            )
            env["ORION_EVOLUTION_STORE"] = str(store)
        else:
            env["ORION_EVOLUTION_STORE"] = str(tmp_path / "empty_evolution_store")

        original_cwd = Path.cwd()
        with temporary_environment(env):
            main._context_manager = isolated
            try:
                os.chdir(tmp_path)
                raw = run_agent(
                    sample,
                    config=SimpleNamespace(auto_approve=True),
                    agent_factory=_v2_first_action_agent_factory(
                        max_llm_turns=int(config["max_llm_turns"]),
                        turn_counter=turn_counter,
                    ),
                )
            finally:
                os.chdir(original_cwd)
                main._context_manager = original_context
                main._configure_runtime_tools()

    tool_calls: list[dict[str, Any]] = []
    for call in raw.get("tool_calls") or []:
        if hasattr(call, "model_dump"):
            tool_calls.append(call.model_dump(mode="json"))
        elif isinstance(call, dict):
            tool_calls.append(call)
        else:
            tool_calls.append(
                {
                    "name": str(getattr(call, "name", "")),
                    "args": getattr(call, "args", {}),
                    "result": getattr(call, "result", None),
                }
            )

    return {
        "case_id": str(case["case_id"]),
        "answer": raw.get("answer", ""),
        "tool_calls": tool_calls,
        "failure_reasons": list(raw.get("failure_reasons") or []),
        "ok": bool(raw.get("ok", True)),
        "error_type": raw.get("error_type"),
        "error_message": raw.get("error_message"),
        "evaluation_timed_out": bool(raw.get("evaluation_timed_out")),
        "llm_turn_count": int(turn_counter["count"]),
        "candidate_seal_sha256": candidate_sha256_v2(proposal) if proposal is not None else None,
        "formal_context_isolation": {
            "assemble_calls": isolated.assemble_calls,
            "suppressed_evidence_calls": isolated.suppressed_evidence_calls,
            "suppressed_memory_calls": isolated.suppressed_memory_calls,
            "passed": isolated.assemble_calls > 0,
        },
        "production_persistence_used": False,
    }


__all__ = [
    "FreshV2DevelopmentRuntimeBinding",
    "FreshV2DevelopmentRuntimeError",
    "FreshV2DirectCallResult",
    "FreshV2FirstActionBudgetExceeded",
    "expected_v2_runtime_environment",
    "invoke_v2_proposer_once",
    "invoke_v2_runtime_health_once",
    "parse_v2_proposer_raw",
    "render_v2_proposer_messages",
    "resolve_v2_runtime_binding",
    "run_v2_first_action_probe_once",
    "v2_first_action_case",
    "v2_scientific_contract",
]
