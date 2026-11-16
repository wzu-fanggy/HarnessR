from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from orion.core.claim_contract import ClaimContract
from orion.core.reflection_entry import ReflectionEntry


_MINERS = []


def _miner(fn):
    _MINERS.append(fn)
    return fn


@_miner
def _mine_missing_cell_configuration(
    claims: list[ClaimContract],
    task_type: str,
) -> list[ReflectionEntry]:
    entries = []
    for claim in claims:
        if claim.claim_type != "recommendation":
            continue
        text = claim.claim_text.lower()
        if "hard carbon" in text and ("-20" in text or "-40" in text or "low temp" in text):
            electrode = None
            if claim.condition_scope:
                electrode = claim.condition_scope.get("electrode")
            if not electrode:
                entries.append(
                    ReflectionEntry(
                        reflection_id=f"ref-{uuid.uuid4().hex[:12]}",
                        created_at=datetime.now(timezone.utc).isoformat(),
                        domain_tags=["sodium-ion", "low-temp", "hard-carbon"],
                        task_types=[task_type],
                        trigger_claim_type="recommendation",
                        failure_pattern="低温硬碳设计缺少半电池构型证据",
                        pattern_code="missing_cell_configuration",
                        corrective_constraint="若 claim 涉及 hard carbon / -20C 及以下，必须绑定 Na||HC 半电池构型证据且 evidence_binding >= condition_bound",
                        supporting_claim_ids=[claim.claim_id],
                        confidence=0.3,
                    )
                )
    return entries


@_miner
def _mine_quote_bound_numeric(
    claims: list[ClaimContract],
    task_type: str,
) -> list[ReflectionEntry]:
    entries = []
    for claim in claims:
        if claim.evidence_binding == "quote_bound" and re.search(
            r"\d+\.?\d*\s*(?:°C|C|M|V|mAh|%)", claim.claim_text
        ):
            entries.append(
                ReflectionEntry(
                    reflection_id=f"ref-{uuid.uuid4().hex[:12]}",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    domain_tags=["claim-quality"],
                    task_types=[task_type],
                    trigger_claim_type=claim.claim_type,
                    failure_pattern="quote_bound 声明包含带单位数值",
                    pattern_code="quote_bound_numeric",
                    corrective_constraint="带单位数值的声明必须升级到 condition_bound，不能仅依赖 quote_bound",
                    supporting_claim_ids=[claim.claim_id],
                    confidence=0.3,
                )
            )
    return entries


def mine_reflections(
    claims: list[ClaimContract],
    task_type: str,
) -> list[ReflectionEntry]:
    """从 claims 中挖掘结构化失败模式。"""
    entries = []
    for miner in _MINERS:
        entries.extend(miner(claims, task_type))
    return entries
