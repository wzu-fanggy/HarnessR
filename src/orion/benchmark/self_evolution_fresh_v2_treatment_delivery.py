"""Fresh v2 treatment delivery instrumentation.

本模块只负责把已经封存的 candidate Skill 写入临时 evolution store，并通过
生产 ``load_skill`` 路径验证 treatment 确实送达。它不调用模型、不写生产 store，
也不修改 frozen v1 资产。
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.benchmark.self_evolution_fresh_v2_scope import FreshPersistentUpdateProposalV2
from orion.evolution.assets.skill_asset import SkillAsset
from orion.evolution.assets.storage import FileSystemAssetStorage


ROOT = Path(__file__).resolve().parent.parent
BASELINE_SKILL_PATH = ROOT / "skills/electrolyte_design/SKILL.md"


class FreshV2TreatmentDeliveryReceipt(BaseModel):
    """零模型 treatment-delivery probe 的可审计结果。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["orion_self_evolution_fresh_v2_treatment_delivery_v1"] = (
        "orion_self_evolution_fresh_v2_treatment_delivery_v1"
    )
    passed: bool
    skill_name: str
    canonical_asset_id: str
    candidate_seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    static_loaded_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_loaded_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    static_equals_frozen_baseline: bool
    candidate_contains_frozen_baseline: bool
    candidate_contains_seal: bool
    candidate_alias_visible_only_in_candidate: bool
    production_load_path_used: bool
    temporary_store_only: bool
    reason_codes: list[str] = Field(default_factory=list)


def canonical_skill_asset_id(skill_name: str) -> str:
    """把 production ``load_skill(name=...)`` 名称映射为唯一 Skill asset ID。

    生产 loader 当前按 ``<skill_name>.skill`` 查询，因此 v2 writer 也只允许这一种
    canonical identity。传入已经带 ``.skill`` 的名称时保持幂等。
    """

    name = skill_name.strip()
    if not name:
        raise ValueError("skill_name may not be empty")
    if any(token in name for token in ("/", "\\", "..")):
        raise ValueError("skill_name may not contain path components")
    return name if name.endswith(".skill") else f"{name}.skill"


@contextmanager
def _temporary_evolution_store(path: Path) -> Iterator[None]:
    """临时覆盖 evolution store，并在退出时恢复调用者环境。"""

    key = "ORION_EVOLUTION_STORE"
    previous = os.environ.get(key)
    os.environ[key] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _canonical(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate_sha256_v2(proposal: FreshPersistentUpdateProposalV2) -> str:
    """对 sealed v2 proposal 的 canonical JSON 计算 treatment identity。"""

    return _sha256_text(_canonical(proposal.model_dump(mode="json")))


def render_v2_candidate_skill(
    proposal: FreshPersistentUpdateProposalV2,
    *,
    expected_baseline_skill_sha256: str,
) -> str:
    """把 v2 typed candidate 渲染为 benchmark sandbox 内的 persistent Skill block。"""

    if proposal.proposal_status != "candidate" or proposal.claimed_scope is None:
        raise ValueError("only a v2 scientific candidate can be mounted")
    baseline = BASELINE_SKILL_PATH.read_text(encoding="utf-8")
    observed = hashlib.sha256(BASELINE_SKILL_PATH.read_bytes()).hexdigest()
    if observed != expected_baseline_skill_sha256:
        raise RuntimeError("baseline electrolyte-design Skill drift before v2 sandbox mount")
    seal = candidate_sha256_v2(proposal)
    append = (
        "\n\n## Frozen research-guidance candidate (benchmark sandbox only)\n\n"
        "This block is persistent guidance, **not current-task scientific evidence**. "
        "Use it only to prioritize evidence acquisition within its stated scope. "
        "Never cite this block as proof, never infer missing conditions, and let current-task evidence override it.\n\n"
        f"- Guidance: {proposal.proposed_content.strip()}\n"
        f"- Claimed scope (canonical JSON): `{_canonical(proposal.claimed_scope.model_dump(mode='json'))}`\n"
        f"- Development-visible limitations (canonical JSON): `{_canonical(proposal.limitations_visible_from_development)}`\n"
        f"- Candidate seal: `{seal}`\n"
    )
    return baseline.rstrip() + append


def mount_v2_candidate_skill(
    store_root: str | Path,
    proposal: FreshPersistentUpdateProposalV2,
    *,
    expected_baseline_skill_sha256: str,
    skill_name: str = "electrolyte_design",
) -> str:
    """按 production canonical asset ID 把 candidate Skill 写入临时 store。"""

    content = render_v2_candidate_skill(
        proposal,
        expected_baseline_skill_sha256=expected_baseline_skill_sha256,
    )
    asset_id = canonical_skill_asset_id(skill_name)
    storage = FileSystemAssetStorage(Path(store_root))
    storage.save(SkillAsset(asset_id=asset_id, content=content))
    loaded = storage.load(asset_id, "skill", SkillAsset)
    if loaded is None or str(loaded.active_content) != content:
        raise RuntimeError("v2 candidate Skill sandbox round-trip mismatch")
    return content


def production_load_skill(skill_name: str, *, store_root: str | Path) -> str:
    """通过真实 production ``load_skill`` tool 路径读取 Skill。"""

    from orion.tools.skill_tools import load_skill as load_skill_tool

    with _temporary_evolution_store(Path(store_root)):
        result = load_skill_tool.invoke({"name": skill_name})
    if not isinstance(result, str):
        raise TypeError("production load_skill did not return text")
    return result


def audit_v2_treatment_delivery(
    *,
    static_store_root: str | Path,
    candidate_store_root: str | Path,
    proposal: FreshPersistentUpdateProposalV2,
    candidate_alias: str,
    expected_baseline_skill_sha256: str,
    skill_name: str = "electrolyte_design",
) -> FreshV2TreatmentDeliveryReceipt:
    """证明 Static 与 candidate condition 在 production loader 上只有 treatment 差异。"""

    alias = candidate_alias.strip()
    if not alias:
        raise ValueError("candidate_alias may not be empty")
    baseline = BASELINE_SKILL_PATH.read_text(encoding="utf-8")
    baseline_sha = hashlib.sha256(BASELINE_SKILL_PATH.read_bytes()).hexdigest()
    if baseline_sha != expected_baseline_skill_sha256:
        raise RuntimeError("baseline Skill hash drift before v2 delivery audit")

    static_store = Path(static_store_root)
    candidate_store = Path(candidate_store_root)
    static_store.mkdir(parents=True, exist_ok=True)
    candidate_store.mkdir(parents=True, exist_ok=True)

    mounted = mount_v2_candidate_skill(
        candidate_store,
        proposal,
        expected_baseline_skill_sha256=expected_baseline_skill_sha256,
        skill_name=skill_name,
    )
    static_loaded = production_load_skill(skill_name, store_root=static_store)
    candidate_loaded = production_load_skill(skill_name, store_root=candidate_store)

    seal = candidate_sha256_v2(proposal)
    static_alias = alias.casefold() in static_loaded.casefold()
    candidate_alias_seen = alias.casefold() in candidate_loaded.casefold()
    checks = {
        "static_equals_frozen_baseline": static_loaded == baseline,
        "candidate_contains_frozen_baseline": candidate_loaded.startswith(baseline.rstrip()),
        "candidate_contains_seal": seal in candidate_loaded,
        "candidate_alias_visible_only_in_candidate": (not static_alias) and candidate_alias_seen,
        "candidate_round_trip_matches_renderer": candidate_loaded == mounted,
    }
    passed = all(checks.values())
    return FreshV2TreatmentDeliveryReceipt(
        passed=passed,
        skill_name=skill_name,
        canonical_asset_id=canonical_skill_asset_id(skill_name),
        candidate_seal_sha256=seal,
        baseline_sha256=baseline_sha,
        static_loaded_sha256=_sha256_text(static_loaded),
        candidate_loaded_sha256=_sha256_text(candidate_loaded),
        static_equals_frozen_baseline=checks["static_equals_frozen_baseline"],
        candidate_contains_frozen_baseline=checks["candidate_contains_frozen_baseline"],
        candidate_contains_seal=checks["candidate_contains_seal"],
        candidate_alias_visible_only_in_candidate=checks[
            "candidate_alias_visible_only_in_candidate"
        ],
        production_load_path_used=True,
        temporary_store_only=True,
        reason_codes=(
            ["v2_treatment_delivery_passed"]
            if passed
            else [key for key, value in checks.items() if not value]
        ),
    )


__all__ = [
    "FreshV2TreatmentDeliveryReceipt",
    "audit_v2_treatment_delivery",
    "candidate_sha256_v2",
    "canonical_skill_asset_id",
    "mount_v2_candidate_skill",
    "production_load_skill",
    "render_v2_candidate_skill",
]
