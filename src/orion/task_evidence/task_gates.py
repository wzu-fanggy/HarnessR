"""四类任务中的纯确定性门禁。



本模块只消费结构化输入和调用方显式提供的 evidence id，不访问 KG/RAG/LLM，

也不从自由文本中猜测实验条件或根因。报告编译器和离线指标应共同消费这里的

评估结果，避免各自实现一套隐含规则。

"""



from __future__ import annotations



from typing import Any, Literal



from pydantic import BaseModel, Field, field_validator, model_validator





ComparisonField = Literal[

    "temperature_c",

    "electrode",

    "cell_configuration",

    "voltage_window_v",

    "rate",

    "cycle_count",

    "measurement_definition",

]



DEFAULT_COMPARABILITY_FIELDS: tuple[ComparisonField, ...] = (

    "temperature_c",

    "electrode",

    "voltage_window_v",

    "rate",

    "cycle_count",

    "measurement_definition",

)





class ComparisonConditions(BaseModel):

    """单个候选的一组结构化实验条件。



    字段故意保持基础且严格：调用方应先从原文抽取并规范化，而不是让门禁对自然

    语言做模糊匹配。``electrode`` 应包含足以区分电池构型的信息，例如

    ``HC || Na half-cell``。

    """



    temperature_c: float | None = None

    electrode: str | None = None

    cell_configuration: str | None = None

    voltage_window_v: tuple[float, float] | None = None

    rate: str | None = None

    cycle_count: int | None = Field(default=None, ge=0)

    measurement_definition: str | None = None



    @field_validator("electrode", "cell_configuration", "rate", "measurement_definition")

    @classmethod

    def _strip_optional_text(cls, value: str | None) -> str | None:

        if value is None:

            return None

        normalized = " ".join(value.split())

        return normalized or None



    @model_validator(mode="after")

    def _validate_voltage_window(self) -> "ComparisonConditions":

        if (

            self.voltage_window_v is not None

            and self.voltage_window_v[0] >= self.voltage_window_v[1]

        ):

            raise ValueError("voltage_window_v must be ordered as [lower, upper]")

        return self





class CandidateComparisonRecord(BaseModel):

    """候选及其被原文证据绑定的可比条件。"""



    candidate_id: str = Field(min_length=1)

    conditions: ComparisonConditions = Field(default_factory=ComparisonConditions)

    evidence_ids: list[str] = Field(default_factory=list)





class GateIssue(BaseModel):

    """阻止强终态的机器可读原因。"""



    code: str = Field(min_length=1)

    description: str = Field(min_length=1)

    item_id: str | None = None





class ComparabilityAssessment(BaseModel):

    """候选条件可比性的确定性判定。"""



    status: Literal["comparable", "not_comparable", "unknown"]

    required_fields: list[ComparisonField] = Field(default_factory=list)

    candidate_ids: list[str] = Field(default_factory=list)

    checked_fields: list[ComparisonField] = Field(default_factory=list)

    missing_fields: dict[str, list[ComparisonField]] = Field(default_factory=dict)

    mismatches: dict[str, dict[str, Any]] = Field(default_factory=dict)

    unbound_evidence_ids: dict[str, list[str]] = Field(default_factory=dict)

    issues: list[GateIssue] = Field(default_factory=list)





class HypothesisRecord(BaseModel):

    """逆向设计中的一个可证伪假设及其证据账本。"""



    hypothesis_id: str = Field(min_length=1)

    text: str = Field(min_length=1)

    supporting_evidence_ids: list[str] = Field(default_factory=list)

    counter_evidence_ids: list[str] = Field(default_factory=list)

    missing_evidence: list[str] = Field(default_factory=list)

    confounders: list[str] = Field(default_factory=list)





class DiscriminatingTest(BaseModel):

    """能够对至少两个假设产生不同预期结果的验证实验。"""



    test_id: str = Field(min_length=1)

    description: str = Field(min_length=1)

    distinguishes_hypothesis_ids: list[str] = Field(default_factory=list)

    expected_outcomes: dict[str, str] = Field(default_factory=dict)

    controlled_variables: list[str] = Field(default_factory=list)





class InverseHypothesisAssessment(BaseModel):

    """竞争假设与区分实验的确定性完备性评估。"""



    status: Literal[

        "diagnosis_supported",

        "diagnosis_hypothesis_only",

        "insufficient_evidence",

    ]

    valid_hypothesis_ids: list[str] = Field(default_factory=list)

    supported_hypothesis_ids: list[str] = Field(default_factory=list)

    adequate_test_ids: list[str] = Field(default_factory=list)

    issues: list[GateIssue] = Field(default_factory=list)





def _normalized_condition_value(field: ComparisonField, value: Any) -> Any:

    if value is None:

        return None

    if field == "temperature_c":

        return round(float(value), 6)

    if field == "voltage_window_v":

        return tuple(round(float(item), 6) for item in value)

    if field == "cycle_count":

        return int(value)

    text = " ".join(str(value).split()).casefold()

    if field == "rate":

        return text.replace(" ", "")

    return text





def evaluate_candidate_comparability(

    records: list[CandidateComparisonRecord | dict[str, Any]],

    *,

    expected_candidate_ids: list[str] | None = None,

    available_evidence_ids: set[str] | None = None,

    required_fields: tuple[ComparisonField, ...] = DEFAULT_COMPARABILITY_FIELDS,

) -> ComparabilityAssessment:

    """严格比较候选工况；缺失信息返回 unknown，显式冲突返回 not_comparable。"""



    parsed = [

        item

        if isinstance(item, CandidateComparisonRecord)

        else CandidateComparisonRecord.model_validate(item)

        for item in records

    ]

    expected = list(

        dict.fromkeys(

            item

            for item in (expected_candidate_ids or [row.candidate_id for row in parsed])

            if item

        )

    )

    issues: list[GateIssue] = []

    seen: dict[str, CandidateComparisonRecord] = {}

    duplicate_ids: set[str] = set()

    for row in parsed:

        if row.candidate_id in seen:

            duplicate_ids.add(row.candidate_id)

        else:

            seen[row.candidate_id] = row

    for candidate_id in sorted(duplicate_ids):

        issues.append(

            GateIssue(

                code="duplicate_candidate_record",

                description="同一候选存在多份条件记录，无法确定权威值。",

                item_id=candidate_id,

            )

        )



    if len(expected) < 2:

        issues.append(

            GateIssue(

                code="insufficient_candidates",

                description="候选比较至少需要两个不同候选。",

            )

        )



    missing_records = [item for item in expected if item not in seen]

    for candidate_id in missing_records:

        issues.append(

            GateIssue(

                code="missing_candidate_record",

                description="候选缺少结构化实验条件记录。",

                item_id=candidate_id,

            )

        )



    unexpected = sorted(set(seen) - set(expected)) if expected_candidate_ids else []

    for candidate_id in unexpected:

        issues.append(

            GateIssue(

                code="unexpected_candidate_record",

                description="条件记录不属于本次待比较候选。",

                item_id=candidate_id,

            )

        )



    missing_fields: dict[str, list[ComparisonField]] = {}

    unbound_evidence_ids: dict[str, list[str]] = {}

    for candidate_id in expected:

        row = seen.get(candidate_id)

        if row is None:

            continue

        missing = [

            field

            for field in required_fields

            if getattr(row.conditions, field) is None

        ]

        if missing:

            missing_fields[candidate_id] = missing

        if not row.evidence_ids:

            issues.append(

                GateIssue(

                    code="missing_condition_evidence",

                    description="候选实验条件未绑定本轮原文证据。",

                    item_id=candidate_id,

                )

            )

        elif available_evidence_ids is not None:

            unbound = sorted(set(row.evidence_ids) - available_evidence_ids)

            if unbound:

                unbound_evidence_ids[candidate_id] = unbound



    if missing_fields:

        issues.append(

            GateIssue(

                code="incomplete_comparison_conditions",

                description="至少一个候选缺少必需实验条件，当前不能判断可比性。",

            )

        )

    if unbound_evidence_ids:

        issues.append(

            GateIssue(

                code="unbound_condition_evidence",

                description="至少一个条件引用不在本轮权威 evidence ledger 中。",

            )

        )



    blocking_unknown = bool(

        issues or missing_fields or unbound_evidence_ids or duplicate_ids

    )

    if blocking_unknown:

        return ComparabilityAssessment(

            status="unknown",

            required_fields=list(required_fields),

            candidate_ids=expected,

            missing_fields=missing_fields,

            unbound_evidence_ids=unbound_evidence_ids,

            issues=issues,

        )



    mismatches: dict[str, dict[str, Any]] = {}

    checked_fields: list[ComparisonField] = []

    for field in required_fields:

        values = {

            candidate_id: _normalized_condition_value(

                field,

                getattr(seen[candidate_id].conditions, field),

            )

            for candidate_id in expected

        }

        checked_fields.append(field)

        if len(set(values.values())) > 1:

            mismatches[field] = values



    if mismatches:

        return ComparabilityAssessment(

            status="not_comparable",

            required_fields=list(required_fields),

            candidate_ids=expected,

            checked_fields=checked_fields,

            mismatches=mismatches,

            issues=[

                GateIssue(

                    code="condition_mismatch",

                    description="候选在一个或多个必需实验条件上不一致。",

                )

            ],

        )

    return ComparabilityAssessment(

        status="comparable",

        required_fields=list(required_fields),

        candidate_ids=expected,

        checked_fields=checked_fields,

    )





def evaluate_inverse_hypotheses(

    hypotheses: list[HypothesisRecord | dict[str, Any]],

    *,

    confounders: list[str],

    tests: list[DiscriminatingTest | dict[str, Any]],

    available_evidence_ids: set[str] | None = None,

) -> InverseHypothesisAssessment:

    """校验竞争假设、证据账本与区分实验，不把实验计划当作已验证根因。"""



    parsed_hypotheses = [

        item

        if isinstance(item, HypothesisRecord)

        else HypothesisRecord.model_validate(item)

        for item in hypotheses

    ]

    parsed_tests = [

        item

        if isinstance(item, DiscriminatingTest)

        else DiscriminatingTest.model_validate(item)

        for item in tests

    ]

    issues: list[GateIssue] = []

    by_id: dict[str, HypothesisRecord] = {}

    duplicates: set[str] = set()

    for hypothesis in parsed_hypotheses:

        if hypothesis.hypothesis_id in by_id:

            duplicates.add(hypothesis.hypothesis_id)

        else:

            by_id[hypothesis.hypothesis_id] = hypothesis

    for hypothesis_id in sorted(duplicates):

        issues.append(

            GateIssue(

                code="duplicate_hypothesis_id",

                description="竞争假设的 hypothesis_id 必须唯一。",

                item_id=hypothesis_id,

            )

        )



    valid_ids: list[str] = []

    for hypothesis_id, hypothesis in by_id.items():

        has_inventory = bool(

            hypothesis.supporting_evidence_ids

            or hypothesis.counter_evidence_ids

            or hypothesis.missing_evidence

        )

        if not has_inventory:

            issues.append(

                GateIssue(

                    code="missing_hypothesis_evidence_inventory",

                    description="假设必须显式记录支持、反证或缺失证据。",

                    item_id=hypothesis_id,

                )

            )

            continue

        referenced = {

            *hypothesis.supporting_evidence_ids,

            *hypothesis.counter_evidence_ids,

        }

        if available_evidence_ids is not None:

            unbound = sorted(referenced - available_evidence_ids)

            if unbound:

                issues.append(

                    GateIssue(

                        code="unbound_hypothesis_evidence",

                        description="假设引用了本轮 evidence ledger 之外的证据。",

                        item_id=hypothesis_id,

                    )

                )

                continue

        valid_ids.append(hypothesis_id)



    if len(valid_ids) < 2:

        issues.append(

            GateIssue(

                code="missing_competing_hypotheses",

                description="逆向设计至少需要两个具有证据账本的可区分假设。",

            )

        )

    if not [item for item in confounders if item.strip()]:

        issues.append(

            GateIssue(

                code="missing_confounders",

                description="尚未记录可能改变诊断结论的混杂因素。",

            )

        )



    valid_id_set = set(valid_ids)

    adequate_test_ids: list[str] = []

    seen_test_ids: set[str] = set()

    duplicate_test_ids: set[str] = set()

    for test in parsed_tests:

        if test.test_id in seen_test_ids:

            duplicate_test_ids.add(test.test_id)

        seen_test_ids.add(test.test_id)

    for test_id in sorted(duplicate_test_ids):

        issues.append(

            GateIssue(

                code="duplicate_discriminating_test_id",

                description="区分实验的 test_id 必须唯一。",

                item_id=test_id,

            )

        )

    for test in parsed_tests:

        if test.test_id in duplicate_test_ids:

            continue

        targets = list(dict.fromkeys(test.distinguishes_hypothesis_ids))

        target_set = set(targets)

        if len(target_set) < 2:

            issues.append(

                GateIssue(

                    code="test_does_not_distinguish_competing_hypotheses",

                    description="区分实验必须同时覆盖至少两个竞争假设。",

                    item_id=test.test_id,

                )

            )

            continue

        if not target_set <= valid_id_set:

            issues.append(

                GateIssue(

                    code="test_references_invalid_hypothesis",

                    description="区分实验引用了不存在或证据账本不完整的假设。",

                    item_id=test.test_id,

                )

            )

            continue

        outcomes = [

            " ".join(test.expected_outcomes.get(item, "").split()).casefold()

            for item in targets

        ]

        if any(not item for item in outcomes) or len(set(outcomes)) < 2:

            issues.append(

                GateIssue(

                    code="test_expected_outcomes_not_discriminating",

                    description="区分实验必须为各假设给出不同且非空的预期结果。",

                    item_id=test.test_id,

                )

            )

            continue

        if not [item for item in test.controlled_variables if item.strip()]:

            issues.append(

                GateIssue(

                    code="test_controls_missing",

                    description="区分实验必须显式记录受控变量。",

                    item_id=test.test_id,

                )

            )

            continue

        adequate_test_ids.append(test.test_id)



    if not adequate_test_ids:

        issues.append(

            GateIssue(

                code="missing_adequate_discriminating_test",

                description="尚无能够区分竞争假设且记录受控变量的实验。",

            )

        )



    blocking_codes = {

        "duplicate_hypothesis_id",

        "missing_hypothesis_evidence_inventory",

        "unbound_hypothesis_evidence",

        "missing_competing_hypotheses",

        "missing_confounders",

        "duplicate_discriminating_test_id",

        "missing_adequate_discriminating_test",

    }

    if any(issue.code in blocking_codes for issue in issues):

        return InverseHypothesisAssessment(

            status="insufficient_evidence",

            valid_hypothesis_ids=valid_ids,

            adequate_test_ids=adequate_test_ids,

            issues=issues,

        )



    supported_ids = [

        hypothesis_id

        for hypothesis_id in valid_ids

        if by_id[hypothesis_id].supporting_evidence_ids

        and not by_id[hypothesis_id].missing_evidence

    ]

    diagnosis_supported = (

        len(supported_ids) == 1

        and all(

            by_id[hypothesis_id].counter_evidence_ids

            for hypothesis_id in valid_ids

            if hypothesis_id != supported_ids[0]

        )

    )

    return InverseHypothesisAssessment(

        status=(

            "diagnosis_supported"

            if diagnosis_supported

            else "diagnosis_hypothesis_only"

        ),

        valid_hypothesis_ids=valid_ids,

        supported_hypothesis_ids=supported_ids,

        adequate_test_ids=adequate_test_ids,

        issues=issues,

    )

