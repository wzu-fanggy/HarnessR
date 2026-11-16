from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReflectionEntry(BaseModel):
    reflection_id: str
    created_at: str = ""
    superseded_at: str | None = None
    status: Literal["active", "superseded", "deprecated"] = "active"
    domain_tags: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    trigger_claim_type: str
    failure_pattern: str
    pattern_code: str
    condition_signature: dict[str, Any] | None = None
    corrective_constraint: str
    supporting_claim_ids: list[str] = Field(default_factory=list)
    supporting_run_ids: list[str] = Field(default_factory=list)
    observed_count: int = 1
    confidence: float = 0.2
