"""Concrete inert runtime adapter for the fresh ORION confirmatory study.

Importing this module performs no semantic model call, network request, formal
lock acquisition or production persistence.  The functions below define the
exact model/evidence/sandbox interfaces that the hash-bound formal runner will
invoke only after external authorization.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from orion.benchmark.epistemic_promotion_baselines import PairedGenericQualityJudgment
from orion.benchmark.epistemic_promotion_live import _message_text, temporary_environment
from orion.benchmark.epistemic_promotion_transport_v3 import invoke_with_preoutput_transport_retry
from orion.benchmark.self_evolution_fresh_formal_inputs import (
    assemble_formal_generic_c_spec,
    assemble_formal_source_bundle,
)
from orion.benchmark.self_evolution_fresh_generic_c import (
    FreshGenericCTrajectoryOutput,
    render_generic_c_trajectory_packet,
    score_generic_c_fixed_contract,
    score_generic_c_holdout,
)
from orion.benchmark.self_evolution_fresh_retention import score_retention_output
from orion.benchmark.self_evolution_fresh_retention_static import RetentionStaticOutput
from orion.benchmark.self_evolution_fresh_scope import FreshPersistentUpdateProposal
from orion.core.context.manager import ContextManager
from orion.evolution.assets.skill_asset import SkillAsset
from orion.evolution.assets.storage import FileSystemAssetStorage


ROOT = Path(__file__).resolve().parent.parent
FORMAL_INPUTS_PATH = ROOT / "benchmark/self_evolution_fresh_v1/FRESH_FORMAL_INPUTS_V1_2026-08-18.json"
RUNTIME_CONTRACT_PATH = ROOT / "benchmark/self_evolution_fresh_v1/FORMAL_RUNTIME_CONTRACT_V1_2026-08-18.json"
FUTURE_EVIDENCE_RELEASE_PATH = ROOT / "benchmark/self_evolution_fresh_v1/FRESH_FUTURE_EVIDENCE_RELEASE_V1_2026-08-18.json"
GENERIC_TRAJECTORY_PROMPT_PATH = ROOT / "benchmark/self_evolution_fresh_v1/FRESH_GENERIC_TRAJECTORY_PROMPT_V1_2026-08-18.md"
BASELINE_SKILL_PATH = ROOT / "skills/electrolyte_design/SKILL.md"
RETENTION_SOURCE_BANK_PATH = ROOT / "benchmark/self_evolution_fresh_v1/RETENTION_BANK_V1_2026-08-18.json"
JSON_RESPONSE_FORMAT = {"type": "json_object"}


class FreshFormalRuntimeError(RuntimeError):
    pass


class FreshFormalRuntimeBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    temperature: float
    top_p: float
    max_output_tokens: int
    hidden_sdk_retries: int
    native_json_response_format: dict[str, str]
    endpoint_sha256: str
    api_key_present: bool


class FreshFormalDirectCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str
    transport_retry_count: int


class FreshFormalFutureEvidencePacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_id: str
    task_role: Literal["positive_transfer", "out_of_scope_control"]
    source_ids: list[str]
    chunks: list[dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized_endpoint(value: str) -> str:
    return value.strip().rstrip("/")


def _formal_inputs() -> dict[str, Any]:
    return _load_json(FORMAL_INPUTS_PATH)


def _runtime_contract() -> dict[str, Any]:
    return _load_json(RUNTIME_CONTRACT_PATH)


def expected_runtime_environment() -> dict[str, str]:
    contract = _runtime_contract()
    return dict(contract["explicit_runtime_override"])


def resolve_formal_runtime_binding() -> FreshFormalRuntimeBinding:
    """Resolve and verify the active model binding without making a model call."""

    # env_utils loads the repository .env without importing the model client.
    import orion.env_utils as env_utils

    contract = _runtime_contract()
    override = contract["explicit_runtime_override"]
    observed_provider = (os.environ.get("ORION_AGENT_PROVIDER") or "").strip()
    observed_model = (os.environ.get("ORION_AGENT_MODEL") or "").strip()
    observed_temp = (os.environ.get("ORION_AGENT_TEMPERATURE") or "").strip()
    observed_top_p = (os.environ.get("ORION_AGENT_TOP_P") or "").strip()
    if observed_provider != override["ORION_AGENT_PROVIDER"]:
        raise FreshFormalRuntimeError("formal provider override is missing or drifted")
    if observed_model != override["ORION_AGENT_MODEL"]:
        raise FreshFormalRuntimeError("formal model override is missing or drifted")
    if observed_temp != override["ORION_AGENT_TEMPERATURE"]:
        raise FreshFormalRuntimeError("formal temperature override is missing or drifted")
    if observed_top_p != override["ORION_AGENT_TOP_P"]:
        raise FreshFormalRuntimeError("formal top_p override is missing or drifted")

    endpoint = _normalized_endpoint(str(getattr(env_utils, "DEEPSEEK_BASE_URL", "") or os.environ.get("DEEPSEEK_BASE_URL") or ""))
    api_key = str(getattr(env_utils, "DEEPSEEK_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY") or "")
    if not endpoint:
        raise FreshFormalRuntimeError("formal DeepSeek endpoint is not configured")
    if not api_key.strip():
        raise FreshFormalRuntimeError("formal DeepSeek API key is not present")

    sampling = contract["sampling"]
    return FreshFormalRuntimeBinding(
        provider=str(contract["provider"]),
        model=str(contract["model"]),
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        max_output_tokens=int(sampling["max_output_tokens"]),
        hidden_sdk_retries=int(sampling["hidden_sdk_retries"]),
        native_json_response_format=JSON_RESPONSE_FORMAT,
        endpoint_sha256=_sha256_text(endpoint),
        api_key_present=True,
    )


def _json_model(*, max_tokens: int):
    """Build the exact direct-call client after runtime overrides are already active."""

    binding = resolve_formal_runtime_binding()
    from langchain.chat_models import init_chat_model
    import orion.llms as llms

    if str(llms._agent_provider) != binding.provider:
        raise FreshFormalRuntimeError("resolved llms provider drift")
    if str(llms._agent_model) != binding.model:
        raise FreshFormalRuntimeError("resolved llms model drift")
    if float(llms._agent_temperature) != binding.temperature:
        raise FreshFormalRuntimeError("resolved llms temperature drift")
    if float(llms._agent_top_p) != binding.top_p:
        raise FreshFormalRuntimeError("resolved llms top_p drift")
    if int(llms.AGENT_MAX_OUTPUT_TOKENS) != binding.max_output_tokens:
        raise FreshFormalRuntimeError("resolved llms max_output_tokens drift")

    model = init_chat_model(
        model=binding.model,
        model_provider=binding.provider,
        base_url=llms._main_base_url,
        api_key=llms.DEEPSEEK_API_KEY,
        max_tokens=max_tokens,
        temperature=binding.temperature,
        top_p=binding.top_p,
        request_timeout=int(_runtime_contract()["direct_json_calls"]["request_timeout_seconds"]),
        max_retries=binding.hidden_sdk_retries,
        extra_body=llms._agent_extra_body,
    )
    bound = model.bind(response_format=JSON_RESPONSE_FORMAT)
    kwargs = getattr(bound, "kwargs", None)
    if not isinstance(kwargs, dict) or kwargs.get("response_format") != JSON_RESPONSE_FORMAT:
        raise FreshFormalRuntimeError("provider-native JSON response_format was not bound")
    return bound


def invoke_formal_json_once(*, system: str, user: str, max_tokens: int) -> FreshFormalDirectCallResult:
    """One direct semantic call with only the frozen pre-output transport retry policy."""

    model = _json_model(max_tokens=max_tokens)

    def invoke():
        return model.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    response, retries = invoke_with_preoutput_transport_retry(
        invoke,
        max_transport_retries=int(_runtime_contract()["transport_policy"]["max_transport_retries_before_response_object"]),
    )
    raw = _message_text(response)
    if not raw.strip():
        raise FreshFormalRuntimeError("provider returned an empty formal semantic response")
    return FreshFormalDirectCallResult(raw=raw, transport_retry_count=retries)


def development_source_packet(family_id: str) -> list[dict[str, Any]]:
    manifest = _formal_inputs()
    bundle = assemble_formal_source_bundle(root=ROOT, manifest=manifest)
    sources = list((bundle.get("families", {}).get(family_id) or {}).get("sources") or [])
    allowed: list[dict[str, Any]] = []
    for source in sources:
        role = str(source.get("role") or "")
        if not role.startswith("development"):
            continue
        allowed.append(
            {
                "source_id": str(source["source_id"]),
                "source_identity": str(source.get("source_identity") or ""),
                "title": str(source.get("title") or ""),
                "facts": list(source.get("facts") or []),
                "explicit_absences": list(source.get("explicit_absences") or []),
            }
        )
    return allowed


def render_formal_proposer_messages(family_id: str) -> tuple[str, str, int]:
    manifest = _formal_inputs()
    prompt = (ROOT / manifest["common_proposer"]["prompt"]).read_text(encoding="utf-8")
    payload = {
        "family_id": family_id,
        "development_evidence": development_source_packet(family_id),
        "information_wall": "No matched boundary, future evaluator source/task, arm label or desired result is supplied.",
    }
    return prompt, "Complete this frozen development packet and return JSON only.\n\n" + _canonical(payload), 2200


def parse_formal_proposer_raw(*, family_id: str, raw: str) -> FreshPersistentUpdateProposal:
    proposal = FreshPersistentUpdateProposal.model_validate_json(raw)
    allowed = {item["source_id"] for item in development_source_packet(family_id)}
    observed = set(proposal.source_evidence_ids)
    if proposal.proposal_status == "candidate" and (not observed or not observed <= allowed):
        raise ValueError("formal proposer referenced evidence outside development-only packet")
    if proposal.proposal_status == "no_update" and observed:
        raise ValueError("no_update formal proposal may not carry source evidence IDs")
    return proposal


def candidate_sha256(proposal: FreshPersistentUpdateProposal) -> str:
    return _sha256_text(_canonical(proposal.model_dump(mode="json")))


def render_fresh_candidate_skill(
    proposal: FreshPersistentUpdateProposal,
    *,
    expected_baseline_skill_sha256: str,
) -> str:
    if proposal.proposal_status != "candidate" or proposal.claimed_scope is None:
        raise ValueError("only a scientific candidate can be mounted")
    baseline = BASELINE_SKILL_PATH.read_text(encoding="utf-8")
    observed = hashlib.sha256(BASELINE_SKILL_PATH.read_bytes()).hexdigest()
    if observed != expected_baseline_skill_sha256:
        raise FreshFormalRuntimeError("baseline electrolyte-design Skill drift before sandbox mount")
    append = (
        "\n\n## Frozen research-guidance candidate (benchmark sandbox only)\n\n"
        "This block is persistent guidance, **not current-task scientific evidence**. "
        "Use it only to prioritize evidence acquisition within its stated scope. "
        "Never cite this block as proof, never infer missing conditions, and let current-task evidence override it.\n\n"
        f"- Guidance: {proposal.proposed_content.strip()}\n"
        f"- Claimed scope (canonical JSON): `{_canonical(proposal.claimed_scope.model_dump(mode='json'))}`\n"
        f"- Development-visible limitations (canonical JSON): `{_canonical(proposal.limitations_visible_from_development)}`\n"
        f"- Candidate seal: `{candidate_sha256(proposal)}`\n"
    )
    return baseline.rstrip() + append


def mount_fresh_candidate_skill(
    store_root: str | Path,
    proposal: FreshPersistentUpdateProposal,
    *,
    expected_baseline_skill_sha256: str,
) -> str:
    content = render_fresh_candidate_skill(
        proposal,
        expected_baseline_skill_sha256=expected_baseline_skill_sha256,
    )
    storage = FileSystemAssetStorage(Path(store_root))
    storage.save(SkillAsset(asset_id="electrolyte_design", content=content))
    loaded = storage.load("electrolyte_design", "skill", SkillAsset)
    if loaded is None or str(loaded.active_content) != content:
        raise FreshFormalRuntimeError("fresh candidate Skill sandbox round-trip mismatch")
    return content


def _formal_generic_spec_and_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _formal_inputs()
    return (
        assemble_formal_generic_c_spec(root=ROOT, manifest=manifest),
        assemble_formal_source_bundle(root=ROOT, manifest=manifest),
    )


def render_formal_generic_trajectory_messages(
    *,
    family_id: str,
    condition: Literal["baseline", "candidate"],
    proposal: FreshPersistentUpdateProposal | None = None,
) -> tuple[str, str, int]:
    spec, sources = _formal_generic_spec_and_sources()
    packet = render_generic_c_trajectory_packet(
        spec=spec,
        source_bundle=sources,
        family_id=family_id,
        condition=condition,
        candidate=proposal,
    )
    system = GENERIC_TRAJECTORY_PROMPT_PATH.read_text(encoding="utf-8")
    return system, "Complete this frozen proxy packet. Return JSON only.\n\n" + _canonical(packet), 1800


def parse_formal_generic_trajectory_raw(raw: str) -> FreshGenericCTrajectoryOutput:
    return FreshGenericCTrajectoryOutput.model_validate_json(raw)


def score_formal_generic_trajectory(
    *,
    family_id: str,
    output: FreshGenericCTrajectoryOutput,
) -> dict[str, Any]:
    spec, sources = _formal_generic_spec_and_sources()
    packet = render_generic_c_trajectory_packet(
        spec=spec,
        source_bundle=sources,
        family_id=family_id,
        condition="baseline",
    )
    allowed = [str(item["source_id"]) for item in packet["evidence"]]
    return score_generic_c_fixed_contract(output=output, allowed_evidence_refs=allowed).model_dump(mode="json")


def render_formal_generic_judge_messages(
    *,
    family_id: str,
    baseline: FreshGenericCTrajectoryOutput,
    candidate: FreshGenericCTrajectoryOutput,
) -> tuple[str, str, int]:
    spec, _ = _formal_generic_spec_and_sources()
    task = spec["family_tasks"][family_id]
    payload = {
        "task_id": task["task_id"],
        "user_task": task["user_task"],
        "generic_task_contract": task["generic_task_contract"],
        "baseline_output": baseline.model_dump(mode="json"),
        "candidate_output": candidate.model_dump(mode="json"),
    }
    judge_path = ROOT / spec["judge_prompt_path"]
    return judge_path.read_text(encoding="utf-8"), "Audit this frozen pair. Return JSON only.\n\n" + _canonical(payload), 1000


def parse_formal_generic_judge_raw(raw: str) -> PairedGenericQualityJudgment:
    return PairedGenericQualityJudgment.model_validate_json(raw)


def render_formal_generic_holdout_messages(
    *,
    proposal: FreshPersistentUpdateProposal,
) -> tuple[str, str, int]:
    spec, _ = _formal_generic_spec_and_sources()
    holdout = spec["isolated_generic_holdout"]
    packet = {
        "task_id": holdout["task_id"],
        "user_task": holdout["user_task"],
        "generic_task_contract": holdout["generic_task_contract"],
        "evidence": list(holdout["evidence"]),
        "sealed_candidate": proposal.model_dump(mode="json"),
    }
    system = GENERIC_TRAJECTORY_PROMPT_PATH.read_text(encoding="utf-8")
    return system, "Complete this frozen generic holdout. Return JSON only.\n\n" + _canonical(packet), 1400


def score_formal_generic_holdout(output: FreshGenericCTrajectoryOutput) -> dict[str, Any]:
    spec, _ = _formal_generic_spec_and_sources()
    allowed = [str(item["evidence_id"]) for item in spec["isolated_generic_holdout"]["evidence"]]
    return score_generic_c_holdout(output=output, allowed_evidence_refs=allowed).model_dump(mode="json")


def _retention_task(task_id: str) -> dict[str, Any]:
    bank = _load_json(RETENTION_SOURCE_BANK_PATH)
    for task in bank["tasks"]:
        if task["task_id"] == task_id:
            return task
    raise KeyError(f"unknown formal retention task: {task_id}")


def formal_candidate_aliases(family_id: str) -> list[str]:
    manifest = _formal_inputs()
    task_path = manifest["task_bundles"].get(family_id)
    if not task_path:
        return []
    payload = _load_json(ROOT / task_path)
    aliases: list[str] = []
    for task in payload["tasks"]:
        for alias in task.get("candidate_aliases") or []:
            if alias not in aliases:
                aliases.append(str(alias))
    return aliases


def render_formal_retention_messages(
    *,
    family_id: str,
    task_id: str,
    proposal: FreshPersistentUpdateProposal,
) -> tuple[str, str, int]:
    task = _retention_task(task_id)
    system = (
        "You are completing a frozen scientific retention check. Use only the supplied Record evidence. "
        "The optional persistent scientific candidate is advice, not evidence: do not put candidate content into direct_facts "
        "or missing_facts unless the current Record itself states it. Return JSON only with exactly decision, direct_facts, "
        "missing_facts, evidence_refs_used, unsupported_claims. decision must be answer or hold; unsupported_claims must be []."
    )
    payload = {
        "family_id": family_id,
        "task_id": task_id,
        "evidence": task["evidence"],
        "user_task": task["user_task"],
        "persistent_candidate_guidance_not_evidence": {
            "proposed_content": proposal.proposed_content,
            "claimed_scope": proposal.claimed_scope.model_dump(mode="json") if proposal.claimed_scope else None,
            "limitations": proposal.limitations_visible_from_development,
        },
    }
    return system, "Complete this frozen retention packet. Return JSON only.\n\n" + _canonical(payload), 1000


def parse_and_score_formal_retention(
    *,
    family_id: str,
    task_id: str,
    raw: str,
) -> dict[str, Any]:
    output = RetentionStaticOutput.model_validate_json(raw)
    score = score_retention_output(
        task=_retention_task(task_id),
        output=output.model_dump(mode="json"),
        persistent_candidate_aliases=formal_candidate_aliases(family_id),
    )
    return {
        "output": output.model_dump(mode="json"),
        "score": score.model_dump(mode="json"),
    }


def formal_future_evidence_packet(
    *,
    family_id: str,
    task_role: Literal["positive_transfer", "out_of_scope_control"],
) -> FreshFormalFutureEvidencePacket:
    release = _load_json(FUTURE_EVIDENCE_RELEASE_PATH)
    family_rule = release["families"][family_id]
    key = "positive_transfer_source_ids" if task_role == "positive_transfer" else "out_of_scope_control_source_ids"
    source_ids = list(family_rule.get(key) or [])
    manifest = _formal_inputs()
    bundle = assemble_formal_source_bundle(root=ROOT, manifest=manifest)
    index: dict[str, dict[str, Any]] = {}
    for family in bundle.get("families", {}).values():
        for source in family.get("sources", []):
            index[str(source["source_id"])] = source
    chunks: list[dict[str, Any]] = []
    for source_id in source_ids:
        source = index[source_id]
        chunks.append(
            {
                "chunk_id": f"{source_id}::frozen-facts",
                "paper_id": source_id,
                "source_id": source_id,
                "source_identity": str(source.get("source_identity") or ""),
                "paper_title": str(source.get("title") or ""),
                "content": " ".join(str(item) for item in source.get("facts") or []),
                "evidence_kind": "frozen_future_primary_source_facts",
            }
        )
    return FreshFormalFutureEvidencePacket(
        family_id=family_id,
        task_role=task_role,
        source_ids=source_ids,
        chunks=chunks,
    )


def _tool_by_name(tools: tuple[Any, ...] | list[Any], name: str) -> Any:
    for item in tools:
        if str(getattr(item, "name", "")) == name:
            return item
    raise FreshFormalRuntimeError(f"required ORION tool missing: {name}")


def build_frozen_future_probe_tools(packet: FreshFormalFutureEvidencePacket) -> tuple[Any, ...]:
    """Expose production tool names but only a frozen current-task evidence oracle."""

    main = importlib.import_module("orion.main")
    base = tuple(main._BASE_TOOLS)
    original_chunks = _tool_by_name(base, "search_literature_chunks")
    original_multiview = _tool_by_name(base, "search_literature_multiview")
    original_paper = _tool_by_name(base, "search_paper_chunks")
    original_query = _tool_by_name(base, "query_formulation_space")
    original_screen = _tool_by_name(base, "screen_formulations")
    original_report = _tool_by_name(base, "synthesize_design_report")
    original_load_skill = _tool_by_name(base, "load_skill")

    frozen = [dict(item) for item in packet.chunks]

    def search_literature_chunks(
        query: str,
        top_k: int = 8,
        mode: str = "hybrid",
        sections: list[str] | None = None,
        paper_ids: list[str] | None = None,
        entities: list[str] | None = None,
        exclude_low_value: bool = True,
        expand_hops: int = 0,
        include_parent: bool = True,
    ) -> Any:
        return frozen[: max(0, int(top_k))]

    def search_literature_multiview(
        queries: list[str],
        top_k: int = 10,
        per_query_top_k: int = 10,
        weights: list[float] | None = None,
        mode: str = "hybrid",
        sections: list[str] | None = None,
        paper_ids: list[str] | None = None,
        entities: list[str] | None = None,
        max_per_paper: int = 2,
        rrf_k: int = 60,
    ) -> Any:
        return frozen[: max(0, int(top_k))]

    def search_paper_chunks(
        paper_id: str,
        query: str,
        top_k: int = 8,
        sections: list[str] | None = None,
    ) -> Any:
        allowed = [item for item in frozen if paper_id in {item["paper_id"], item["source_id"], item["source_identity"]}]
        if not allowed:
            return {"error": "paper_not_in_frozen_future_evidence_packet", "paper_id": paper_id, "results": []}
        return {"paper_id": paper_id, "results": allowed[: max(0, int(top_k))]}

    def query_formulation_space(request: Any) -> Any:
        return {
            "candidates": [],
            "source": "fresh_formal_frozen_future_no_global_kg",
            "request_received": request,
            "decision": "hold_for_current_task_evidence",
        }

    def screen_formulations(request: Any) -> Any:
        return {
            "status": "hold",
            "source": "fresh_formal_deterministic_screen",
            "request_received": request,
            "reason": "frozen_probe_does_not_authorize_production_recommendation",
        }

    def synthesize_design_report(request: Any, state: dict[str, Any]) -> Any:
        return {
            "decision": "hold",
            "source": "fresh_formal_deterministic_report",
            "request_received": request,
            "state_received": state,
            "reason": "frozen_probe_does_not_authorize_production_recommendation",
        }

    return (
        original_load_skill,
        StructuredTool.from_function(func=search_literature_chunks, name="search_literature_chunks", description=original_chunks.description, args_schema=original_chunks.args_schema),
        StructuredTool.from_function(func=search_literature_multiview, name="search_literature_multiview", description=original_multiview.description, args_schema=original_multiview.args_schema),
        StructuredTool.from_function(func=search_paper_chunks, name="search_paper_chunks", description=original_paper.description, args_schema=original_paper.args_schema),
        StructuredTool.from_function(func=query_formulation_space, name="query_formulation_space", description=original_query.description, args_schema=original_query.args_schema),
        StructuredTool.from_function(func=screen_formulations, name="screen_formulations", description=original_screen.description, args_schema=original_screen.args_schema),
        StructuredTool.from_function(func=synthesize_design_report, name="synthesize_design_report", description=original_report.description, args_schema=original_report.args_schema),
    )


class _FormalIsolatedContextManager(ContextManager):
    """Disable automatic RAG/memory/reflection; the model must call frozen tools itself."""

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
            raise FreshFormalRuntimeError("formal future context isolation failed")
        return messages


def _formal_future_agent_factory(packet: FreshFormalFutureEvidencePacket):
    def factory():
        main = importlib.import_module("orion.main")
        selected = build_frozen_future_probe_tools(packet)
        original = main._BASE_TOOLS
        main._BASE_TOOLS = selected
        try:
            main._configure_runtime_tools()
            return main._build_graph(main.llm_async, rebuild_memory=False)
        finally:
            main._BASE_TOOLS = original
    return factory


def run_formal_future_probe_once(
    *,
    task: dict[str, Any],
    proposal: FreshPersistentUpdateProposal | None,
    expected_baseline_skill_sha256: str,
) -> dict[str, Any]:
    """Run one hidden future episode. Caller must already hold the exact formal slot lock."""

    resolve_formal_runtime_binding()
    from orion.evaluation.agent_runner import run_agent
    from orion.evaluation.models import EvaluationSample

    main = importlib.import_module("orion.main")
    import orion.llms as llms
    if str(llms._agent_model) != _runtime_contract()["model"]:
        raise FreshFormalRuntimeError("future agent model was imported under a different runtime")
    binding = resolve_formal_runtime_binding()
    if float(llms._agent_temperature) != binding.temperature:
        raise FreshFormalRuntimeError("future agent temperature drift")
    if float(llms._agent_top_p) != binding.top_p:
        raise FreshFormalRuntimeError("future agent top_p drift")
    if int(llms.AGENT_MAX_OUTPUT_TOKENS) != binding.max_output_tokens:
        raise FreshFormalRuntimeError("future agent max_output_tokens drift")
    if int(getattr(llms.main_model_nonstream, "max_retries", -1)) != binding.hidden_sdk_retries:
        raise FreshFormalRuntimeError("future agent SDK retry drift")

    packet = formal_future_evidence_packet(family_id=str(task["family_id"]), task_role=str(task["role"]))  # type: ignore[arg-type]
    sample = EvaluationSample(
        question=str(task["prompt"]),
        task_type=str(task["task_type"]),
        ground_truth="",
        expected_tools=[],
    )
    env = {
        "ORION_GROUNDING_CHECK": "0",
        "ORION_LLM_CACHE": "0",
        "ORION_LLM_STREAM": "0",
        "ORION_REFLECTION_INJECT": "0",
        "ORION_EVAL_AGENT_TIMEOUT": str(_runtime_contract()["agent_calls"]["agent_timeout_seconds"]),
        "ORION_EVAL_LLM_STEP_TIMEOUT": str(_runtime_contract()["agent_calls"]["llm_step_timeout_seconds"]),
        "ORION_EVAL_LLM_REQUEST_MAX_RETRIES": "1",
        "ORION_LLM_REQUEST_MAX_RETRIES": "1",
    }

    original_context = main._context_manager
    isolated = _FormalIsolatedContextManager(
        memory_service=main._memory_service,
        model_name=str(_runtime_contract()["model"]),
    )

    with tempfile.TemporaryDirectory(prefix=f"orion-fresh-formal-{task['task_id']}-") as tmp:
        tmp_path = Path(tmp)
        if proposal is not None:
            store = tmp_path / "candidate_evolution_store"
            mount_fresh_candidate_skill(
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
                    agent_factory=_formal_future_agent_factory(packet),
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
            tool_calls.append({"name": str(getattr(call, "name", "")), "args": getattr(call, "args", {}), "result": getattr(call, "result", None)})

    return {
        "task_id": str(task["task_id"]),
        "sample": sample.model_dump(mode="json"),
        "answer": raw.get("answer", ""),
        "contexts": [],
        "retrieved_contexts": raw.get("contexts", []),
        "tool_calls": tool_calls,
        "failure_reasons": raw.get("failure_reasons", []),
        "ok": raw.get("ok", True),
        "error_type": raw.get("error_type"),
        "error_message": raw.get("error_message"),
        "evaluation_timed_out": bool(raw.get("evaluation_timed_out")),
        "task_artifact": raw.get("task_artifact"),
        "llm_request_max_retries": raw.get("llm_request_max_retries"),
        "llm_transport_retry_count": int(raw.get("llm_transport_retry_count") or 0),
        "formal_context_isolation": {
            "assemble_calls": isolated.assemble_calls,
            "suppressed_evidence_calls": isolated.suppressed_evidence_calls,
            "suppressed_memory_calls": isolated.suppressed_memory_calls,
            "passed": isolated.assemble_calls > 0 and isolated.suppressed_evidence_calls > 0 and isolated.suppressed_memory_calls > 0,
        },
        "frozen_future_evidence_source_ids": packet.source_ids,
        "production_persistence_used": False,
    }


__all__ = [
    "FreshFormalDirectCallResult",
    "FreshFormalFutureEvidencePacket",
    "FreshFormalRuntimeBinding",
    "FreshFormalRuntimeError",
    "build_frozen_future_probe_tools",
    "candidate_sha256",
    "development_source_packet",
    "expected_runtime_environment",
    "formal_candidate_aliases",
    "formal_future_evidence_packet",
    "invoke_formal_json_once",
    "mount_fresh_candidate_skill",
    "parse_and_score_formal_retention",
    "parse_formal_generic_judge_raw",
    "parse_formal_generic_trajectory_raw",
    "parse_formal_proposer_raw",
    "render_formal_generic_holdout_messages",
    "render_formal_generic_judge_messages",
    "render_formal_generic_trajectory_messages",
    "render_formal_proposer_messages",
    "render_formal_retention_messages",
    "render_fresh_candidate_skill",
    "resolve_formal_runtime_binding",
    "run_formal_future_probe_once",
    "score_formal_generic_holdout",
    "score_formal_generic_trajectory",
]
