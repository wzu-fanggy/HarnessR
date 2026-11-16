"""失败分类与处置策略（Learn Claude Code s11 的领域化版本）。



教学版 s11 讲的是"工具报错怎么办"：分类、退避、重试、升级。但配方筛选任务的

失败大多不是异常，而是**语义失败**——没报错、结果为空、或者数值算出来了但不在

可信范围内。这类失败如果一律重试，就是在烧预算做无用功：



- 适用域外的配方，重试 10 次还是域外；

- 约束互相冲突（要求 −40 ℃ 下 σ ≥ 10 mS/cm），重试不会让约束变得可满足；

- 文献里确实没人测过这个体系，换关键词只是换一种"查不到"。



所以这里把失败分成两类处置：**可重试（且必须换策略）** 与 **不可重试（立即转人）**。

"重试但用同样方法"一律不允许——``retry_strategy`` 是强制字段，写不出具体改法

就不该重试。

"""



from __future__ import annotations



import re

from dataclasses import asdict, dataclass

from enum import Enum





class FailureKind(str, Enum):

    """失败的语义类别。前五类是领域性的，后两类是通用性的。"""



    NOT_FOUND = "not_found"

    OUT_OF_DOMAIN = "out_of_domain"

    INSUFFICIENT_INPUT = "insufficient_input"

    CONSTRAINT_CONFLICT = "constraint_conflict"

    EVIDENCE_WEAK = "evidence_weak"

    TOOL_ERROR = "tool_error"

    UNKNOWN = "unknown"





@dataclass(frozen=True)

class FailurePolicy:

    """某一类失败的处置方式。"""



    kind: FailureKind

    label: str

    retryable: bool

    retry_strategy: str

    terminal_reason: str

    human_question: str



    def to_dict(self) -> dict:

        d = asdict(self)

        d["kind"] = self.kind.value

        return d





POLICIES: dict[FailureKind, FailurePolicy] = {

    FailureKind.NOT_FOUND: FailurePolicy(

        kind=FailureKind.NOT_FOUND,

        label="查不到",

        retryable=True,

        retry_strategy="换检索路径：先放宽条件（温度±5 ℃、浓度±0.1），再改用盐名/溶剂全称，最后查同体系邻近点",

        terminal_reason="not_found_exhausted",

        human_question="{subject}：放宽条件、换名称、查邻近点三条路径都试过，仍查不到"

                       "（最近一次：{hint}）。要放宽到相邻体系取证，还是这条按「无数据」记录？",

    ),

    FailureKind.OUT_OF_DOMAIN: FailurePolicy(

        kind=FailureKind.OUT_OF_DOMAIN,

        label="适用域外",

        retryable=False,

        retry_strategy="",

        terminal_reason="out_of_domain",

        human_question="{subject}：落在适用域外，按规矩不能给数值（判断依据：{hint}）。"

                       "要改用文献证据清单形式回答（只给实测值、不做预测），还是直接告知无数据？",

    ),

    FailureKind.INSUFFICIENT_INPUT: FailurePolicy(

        kind=FailureKind.INSUFFICIENT_INPUT,

        label="输入不足（组成/条件缺失）",

        retryable=True,

        retry_strategy="补输入：从配方解析器或原文取摩尔组成（x_NaSalt/x_EC/x_PC）、配比基准或温度，"

                       "补不齐则按域外处理，只给文献证据不做预测",

        terminal_reason="insufficient_input_exhausted",

        human_question="{subject}：缺必要的组成或条件信息，模型无法评估（{hint}）。"

                       "要按「无预测、只给文献证据」处理，还是你能补齐组成后再算？",

    ),

    FailureKind.CONSTRAINT_CONFLICT: FailurePolicy(

        kind=FailureKind.CONSTRAINT_CONFLICT,

        label="约束冲突",

        retryable=False,

        retry_strategy="",

        terminal_reason="constraint_conflict",

        human_question="{subject}：约束无法同时满足（{hint}）。要放宽哪一条？请指定优先级。",

    ),

    FailureKind.EVIDENCE_WEAK: FailurePolicy(

        kind=FailureKind.EVIDENCE_WEAK,

        label="证据不足或不可分辨",

        retryable=True,

        retry_strategy="补证据：扩到相邻温度窗口、同体系其他浓度点，或按可分辨性判据判断前 k 名跨度是否小于误差",

        terminal_reason="evidence_weak_exhausted",

        human_question="{subject}：证据只能支持到「方向」层面，排不出可靠名次（{hint}）。"

                       "要按并列处理（只报区间和方向），还是继续补数据到能分辨为止？",

    ),

    FailureKind.TOOL_ERROR: FailurePolicy(

        kind=FailureKind.TOOL_ERROR,

        label="工具异常",

        retryable=True,

        retry_strategy="原样重试一次；仍失败则换等价工具或拆小输入（分批查询、缩小 top-k）",

        terminal_reason="tool_error_exhausted",

        human_question="{subject}：工具连续失败，已重试（最近一次：{hint}）。"

                       "要跳过这一步（标记为未验证）继续后面的流程，还是中断整个任务？",

    ),

    FailureKind.UNKNOWN: FailurePolicy(

        kind=FailureKind.UNKNOWN,

        label="未分类失败",

        retryable=True,

        retry_strategy="下次调用必须显式给 kind（not_found / out_of_domain / constraint_conflict / "

                       "evidence_weak / tool_error），否则不能重试",

        terminal_reason="unclassified",

        human_question="{subject}：失败原因未分类（{hint}）。"

                       "请说明这属于「查不到」「域外」「约束冲突」「证据不足」还是「工具异常」。",

    ),

}





# 工具返回的 ``status`` 直接映射成失败类别，不让模型从自然语言里猜。

# 来源：tools/electrolyte_design/conductivity_tool.py 的 record["status"]。

# 这样模型只需把工具返回的 status 原样传给 board_fail(kind=...)，语义判断由系统做。

# 这些是"成功"状态。模型误把它们当失败上报时要明确拒绝——

# 把 predicted 记成失败，会让看板和失败统计同时失真。

SUCCESS_STATUSES: frozenset[str] = frozenset({"predicted", "ok", "success", "completed"})



TOOL_STATUS_MAP: dict[str, FailureKind] = {

    "out_of_domain": FailureKind.OUT_OF_DOMAIN,

    "insufficient_composition": FailureKind.INSUFFICIENT_INPUT,

    "insufficient_input": FailureKind.INSUFFICIENT_INPUT,

    "missing_composition": FailureKind.INSUFFICIENT_INPUT,

    "not_found": FailureKind.NOT_FOUND,

    "empty": FailureKind.NOT_FOUND,

    "no_match": FailureKind.NOT_FOUND,

    "constraint_conflict": FailureKind.CONSTRAINT_CONFLICT,

    "evidence_weak": FailureKind.EVIDENCE_WEAK,

    "timeout": FailureKind.TOOL_ERROR,

    "error": FailureKind.TOOL_ERROR,

    "failed": FailureKind.TOOL_ERROR,

}





# 关键词只用于"调用方没给 kind 时的兜底推断"，命中不了就落 UNKNOWN 并要求补 kind。

# 顺序有意义：先判领域性的，再判通用性的，最后兜底。

_PATTERNS: list[tuple[FailureKind, re.Pattern]] = [

    (

        FailureKind.CONSTRAINT_CONFLICT,

        re.compile(

            r"约束.{0,6}冲突|冲突|矛盾|无法同时满足|不可同时满足|no feasible|infeasible|"

            r"violat|constraint.{0,10}(fail|conflict|unsatisf)",

            re.I,

        ),

    ),

    (

        FailureKind.INSUFFICIENT_INPUT,

        re.compile(

            r"组成.{0,8}(不足|缺失|不全|不完整)|缺.{0,6}(组成|配比|温度)|"

            r"insufficient (composition|input)|missing (composition|input|ratio)|"

            r"insufficient_composition",

            re.I,

        ),

    ),

    (

        FailureKind.OUT_OF_DOMAIN,

        re.compile(

            r"域外|适用域|超出.{0,4}范围|不支持.{0,6}(体系|配方|溶剂|盐)|"

            r"out[ _-]?of[ _-]?domain|applicability|unsupported|not covered|no training data",

            re.I,

        ),

    ),

    (

        FailureKind.EVIDENCE_WEAK,

        re.compile(

            r"证据不足|不可分辨|无法分辨|样本.{0,4}(太少|不足)|点数.{0,4}(太少|不足)|"

            r"insufficient evidence|not resolvable|too few|low confidence",

            re.I,

        ),

    ),

    (

        FailureKind.NOT_FOUND,

        re.compile(

            # 中文里"无…记录"中间的成分可能很长（"仍无 NaPF6|G2 的 −20 ℃ 记录"），

            # 用同句内有限跨度匹配，避免把"无显著差异"这类误判成查不到。

            r"查不到|找不到|未找到|检索无结果|没有[^。；\n]{0,12}(数据|记录|结果|资料)|"

            r"无[^。；\n]{0,25}(数据|记录|结果)|"

            r"not found|no (data|record|result|match)|nothing (found|here|matches)|"

            r"finding nothing|empty result|missing",

            re.I,

        ),

    ),

    (

        FailureKind.TOOL_ERROR,

        re.compile(

            r"timeout|timed out|超时|connection|连接|network|网络|refused|"

            r"traceback|exception|错误码|status[ _-]?code[ =:]*[45]\d\d|5\d\d (server|error)|"

            r"tool.{0,10}error|crash|崩溃",

            re.I,

        ),

    ),

]





def infer_kind(error: str) -> FailureKind:

    """没给 kind 时，从错误文本推断类别。推不出来就是 UNKNOWN。"""

    text = error or ""

    for kind, pat in _PATTERNS:

        if pat.search(text):

            return kind

    return FailureKind.UNKNOWN





def classify(error: str, kind: FailureKind | str | None = None) -> FailurePolicy:

    """返回该类失败的处置策略。



    ``kind`` 可由调用方显式给出（优先），否则按 ``error`` 文本推断。

    """

    if kind is None:

        resolved = infer_kind(error)

    elif isinstance(kind, FailureKind):

        resolved = kind

    else:

        raw = str(kind).strip().lower()

        if raw in SUCCESS_STATUSES:

            raise ValueError(

                f"{kind!r} 是成功状态，不需要记失败。只有真正没拿到结果时才调 board_fail。"

            )

        if raw in TOOL_STATUS_MAP:

            # 模型直接把工具返回的 status 贴过来即可，语义判断在系统侧完成

            resolved = TOOL_STATUS_MAP[raw]

        else:

            try:

                resolved = FailureKind(raw)

            except ValueError as exc:  # 拼错枚举要立刻报错，不要静默降级成 UNKNOWN

                raise ValueError(

                    f"未知的失败类别 {kind!r}，可选：{[k.value for k in FailureKind]}；"

                    f"也可以直接传工具返回的 status：{sorted(TOOL_STATUS_MAP)}"

                ) from exc

    return POLICIES[resolved]





def render_question(policy: FailurePolicy, subject: str, hint: str = "") -> str:

    """把策略里的问句模板填成一句能直接回答的问题。"""

    return policy.human_question.replace("{subject}", subject).replace(

        "{hint}", hint or "具体原因见失败记录"

    )





def summarize_failures(records: list[dict]) -> dict[str, int]:

    """按类别统计失败次数——第 5 章消融要看失败分布，不是只看总次数。"""

    out: dict[str, int] = {}

    for r in records:

        k = str(r.get("kind") or FailureKind.UNKNOWN.value)

        out[k] = out.get(k, 0) + 1

    return out

