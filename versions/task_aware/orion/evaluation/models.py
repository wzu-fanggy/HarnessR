from __future__ import annotations



from pydantic import BaseModel, Field, model_validator



from orion.task_evidence.contracts_v2 import (
    TASK_TYPE_SKILLS,
    TaskType,
    canonical_task_type,
    expected_skill_for_task_type,
)




# 任务类型到 Skill 的映射；用于从 Skill frontmatter 读取 required workflow steps，

# 而不是依赖可能包含可选工具的 expected_tools 列表。

_TASK_TYPE_TO_SKILL = TASK_TYPE_SKILLS




def _required_tool_names_for_task_type(task_type: str) -> list[str] | None:

    """从对应 Skill 的 workflow.steps 中提取 required=true 的工具名。



    找不到 Skill 或没有 workflow 时返回 None，调用方应回退到 sample.expected_tools。

    """

    skill_name = expected_skill_for_task_type(task_type)
    if not skill_name:

        return None

    # 延迟导入，避免在模型模块顶层加载 Skill 扫描器。

    from orion.tools.skill_loader import get_skill_workflow



    steps = get_skill_workflow(skill_name)

    if not steps:

        return None

    return [step["tool"] for step in steps if step.get("required")]





class EvaluationSample(BaseModel):

    """一条评估样本。"""



    question: str

    task_type: TaskType

    source_task_type: str | None = None

    expected_tools: list[str] = Field(default_factory=list)

    ground_truth: str

    tags: list[str] = Field(default_factory=list)

    benchmark_case_id: str | None = None

    contract_version: str | None = None



    @model_validator(mode="before")

    @classmethod

    def _normalize_legacy_task_type(cls, value):

        """加载冻结基线时映射旧标签，同时保留原标签供审计。"""



        if not isinstance(value, dict):

            return value

        payload = dict(value)

        raw = payload.get("task_type")

        canonical = canonical_task_type(

            raw,

            question=str(payload.get("question") or ""),

        )

        if raw and raw != canonical:

            payload.setdefault("source_task_type", str(raw))

        payload["task_type"] = canonical

        return payload





class ToolCallRecord(BaseModel):

    """Agent 实际调用的工具记录。"""



    name: str

    args: dict = Field(default_factory=dict)

    # 工具返回值形态不限：dict / str / list（如 formulate_candidates 返回 []）

    result: dict | str | list | None = None





class EvaluationResult(BaseModel):

    """单条样本的评估结果。"""



    sample: EvaluationSample

    answer: str

    contexts: list[str] = Field(default_factory=list)

    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    tool_messages: list[str] = Field(default_factory=list)

    metrics: dict[str, float] = Field(default_factory=dict)

    failure_reasons: list[str] = Field(default_factory=list)

    task_artifact: dict | None = None

    grounding_status: str | None = None

    grounding_error: str | None = None

    grounding_duration_seconds: float | None = None

    grounding_fallback_used: bool = False

    grounding_primary_error: str | None = None

    claim_contract_summary: dict | None = None

    evaluation_timed_out: bool = False

    evaluation_timeout_seconds: float | None = None

    run_duration_seconds: float | None = None
    ok: bool = True
    error_type: str | None = None
    error_message: str | None = None
    # 不含密钥的可复现实验标识；必须随 raw JSONL 一起持久化，不能只依赖文件名。

    provenance: dict[str, str | int | float | bool | None] = Field(default_factory=dict)



    @property

    def needs_more_evidence(self) -> bool:

        return self.metrics.get("context_recall", 1.0) < 0.5



    @property

    def has_hallucination_risk(self) -> bool:

        return self.metrics.get("faithfulness", 1.0) < 0.5



    @property

    def expected_tools_missing(self) -> list[str]:

        """缺失的工具列表：优先按 Skill workflow 的 required steps 判定。"""

        actual = {tc.name for tc in self.tool_calls}

        # 用 Skill required workflow 替代可能包含可选工具的 expected_tools

        required = _required_tool_names_for_task_type(self.sample.task_type)

        if required is None:

            required = self.sample.expected_tools

        missing = []

        for exp in required:

            if exp in actual:

                continue

            # named-study 下论文内聚焦检索是通用 focused chunks 的更具体替代；

            # multiview 仍是独立 required step，不能被任意检索调用掩盖。

            if exp == "search_literature_chunks" and "search_paper_chunks" in actual:

                continue

            missing.append(exp)

        return missing





class AggregateReport(BaseModel):

    """聚合评估报告。"""



    total: int

    success: int

    failed: int

    run_total: int = 0

    run_success: int = 0

    average_metrics: dict[str, float] = Field(default_factory=dict)

    by_task_type: dict[str, dict[str, float]] = Field(default_factory=dict)

    failure_cases: list[dict] = Field(default_factory=list)

    recommendations: list[str] = Field(default_factory=list)

