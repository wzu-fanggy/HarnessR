"""任务级证据 sidecar 的可机读数据契约。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


TaskType = Literal["design", "screening", "inverse_design"]
TaskTerminalStatus = Literal[
    "completed_validated",
    "completed_proposed_only",
    "completed_validated_adjustment",
    "completed_hypothesis_only",
    "completed",
    "not_comparable",
    "insufficient_evidence",
    "invalid_input",
    "tool_failure",
]
ClaimType = Literal[
    "validated_performance",
    "component_evidence",
    "mechanism_evidence",
    "hypothesis",
    "comparison",
]


class SourceSpan(BaseModel):
    """不复制全文的最小可审计文献定位。

    ``mineru_block`` 可核验 MinerU 原文块，但不表示已映射到现役 vector child。
    """

    paper_id: str = Field(min_length=1)
    parent_chunk_id: str | None = None
    child_chunk_id: str = ""
    source_kind: Literal["vector_child", "mineru_block"] = "vector_child"
    source_locator: str | None = None
    page_idx: int | None = Field(default=None, ge=0)
    bbox: list[float] | None = None
    chunk_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_span_order(self) -> "SourceSpan":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self


class Observation(BaseModel):
    """实验中观察到的单项性能或表征结果。"""

    metric: str = Field(min_length=1)
    value: float | str | None = None
    unit: str | None = None
    qualifier: str | None = None


class ExperimentRecord(BaseModel):
    """论文中单次可审计实验的结构化记录。"""

    experiment_id: str = Field(min_length=1)
    formulation: dict[str, object] = Field(default_factory=dict)
    cell_context: dict[str, object] = Field(default_factory=dict)
    test_conditions: dict[str, object] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list)
    evidence_ref: SourceSpan
    completeness: Literal["complete", "partial", "ambiguous"]


class EvidenceAssertion(BaseModel):
    """任务输出中的一项可追溯事实或假设。"""

    claim_type: ClaimType
    text: str = Field(min_length=1)
    experiment_id: str | None = None
    evidence_ref: SourceSpan | None = None


class TaskContract(BaseModel):
    """某类任务的可机读输入/证据/终态约束。"""

    contract_version: str = Field(min_length=1)
    task_type: TaskType
    required_evidence_kinds: list[str] = Field(default_factory=list)
    allowed_terminal_statuses: list[TaskTerminalStatus] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)


class TaskRunArtifact(BaseModel):
    """离线保存的任务级证据 artifact，不替代 Agent state。"""

    contract_version: str = Field(min_length=1)
    task_type: TaskType
    status: TaskTerminalStatus
    benchmark_case_id: str | None = None
    input: dict[str, object] = Field(default_factory=dict)
    candidates: list[dict[str, object]] = Field(default_factory=list)
    experiment_records: list[ExperimentRecord] = Field(default_factory=list)
    assertions: list[EvidenceAssertion] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    provenance: dict[str, object] = Field(default_factory=dict)

    def validate_claim_bindings(self) -> list[str]:
        """报告性能主张的实验绑定缺口，不在构造时抛错以便离线评分。"""
        experiment_ids = {record.experiment_id for record in self.experiment_records}
        errors: list[str] = []
        for assertion in self.assertions:
            if assertion.claim_type != "validated_performance":
                continue
            if not assertion.experiment_id:
                errors.append("missing_experiment_id")
            elif assertion.experiment_id not in experiment_ids:
                errors.append(f"unknown_experiment_id:{assertion.experiment_id}")
        return errors
