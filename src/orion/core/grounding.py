"""答案接地校验（grounding check）——evaluator 回路。



动机（v20–v21 归因结论）：vanilla faithfulness 缺口中约 25–45% 是真实质量问题

（过度概括、参数知识混入），prompt 级引用纪律对长报告生成不咬文（v21 实测

"（一般知识）"标注 0 次出现）。有效杠杆是在生成后加一道架构级校验：



1. 从最终答案抽取**事实性论断**（数值/组成/性能/机理/归属）；

2. 只依据本次运行检索到的证据 contexts 逐条判定可否直接推出；

3. 未通过论断做**外科式最小编辑**（replace/delete/annotate，字符串级应用到

   原文），其余内容逐字节保留；

4. 每个版本都重新校验，出厂版本 = 各已校验版本中未通过论断最少者

   （best-version selection）。



v23–v24 教训（决定 3–4 的设计）：最初修订是让 LLM **全篇重写**报告，结果

校验越严指标越差——全篇重写会把原本有证据的表述漂移成合取/概括表述

（revision churn），且按 attempts 上限放行时最终稿必然未经校验（开环尾段）。

短答案上严格校验单调改善（FEC 0.74→0.97），长报告上单调退化（高电压

0.53→0.26）。结论：瓶颈是修订保真度，不是校验器强度。故改为：编辑操作

结构化输出 + 程序应用 + 每个版本都过校验 + 出厂最优已检版本，从构造上

消除 churn 和未检尾段。



设计约束：

- 校验 LLM 由调用方注入：生产默认独立模型（llms.get_grounding_model，

  与生成模型不同源），规避"自己审自己"的自检盲点——v22 归因显示同模型

  校验对过度概括/参数知识两类判不出来（合计约占剩余 MISS 的 45%）；

  未配置独立模型时回退主模型（行为同 v22）；

- 证据抽取复用 orion.core.evidence_extract（与评估侧同口径）；

- 只校验事实性论断，设计综合内容（候选方案、推荐、下一步）不在修订范围

  ——评估侧用 synthesis_ratio 监控该回路是否误伤综合层；

- 普通自由文本报告的 LLM/解析异常保留 fail-open；ClaimContract 路径 fail-closed；

- 节点为同步实现（llm.invoke），兼容 sync/async 两种 graph 调用方式；

  代价是最终回答前多 2–8 次阻塞调用（校验 2 次 + 每修订轮 3 次）。

"""



from __future__ import annotations



from orion.core.task_scope import current_task_state, is_user_input



import json

import os

import re

import threading

import time

from dataclasses import dataclass, field

from typing import Any



from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage

from langchain_core.runnables import RunnableConfig



from orion.core.claim_contract import ClaimContract

from orion.core.evidence_extract import extract_contexts

from orion.task_evidence.semantic_support import enforce_quote_identity_v2



_PARAMETER_RE = re.compile(

    r"(\d+\.?\d*)\s*(°C|cP|mPa·s|mS\s*cm-1|mAh\s*g-1|kJ\s*mol-1|V|nm|wt%|vol%|cycles?|%)"

)





def _is_parameter_claim(claim: str) -> bool:

    return bool(_PARAMETER_RE.search(claim))





def _retrieve_for_claims(claims: list[str]) -> list[str]:

    """For ungrounded claims, run narrow retrieval and return text snippets.



    Parameter claims go through verify_parameter_claim (entity+number aware).

    Non-parameter claims are searched directly with the claim text as query.

    """

    from tools.rag.verify_tools import verify_parameter_claim

    from tools.rag.kg_rag_tools import _get_rag_service_client



    extra_contexts: list[str] = []

    seen_chunk_ids: set[str] = set()



    try:

        rag_client = _get_rag_service_client()

    except Exception:

        rag_client = None



    for claim in claims[:5]:  # cap cost

        sources: list[dict] = []

        if _is_parameter_claim(claim):

            try:

                result = verify_parameter_claim.invoke({"claim": claim, "top_k": 5})

                verdict = result.get("verdict")

                # Accept supported verdict; also accept missing verdict with sources for robustness.

                if verdict == "supported" or (verdict is None and result.get("sources")):

                    sources = result.get("sources", [])

            except Exception:

                pass

        elif rag_client is not None:

            # 非数值论断直接经本地 RAG 服务检索，避免 API 进程加载模型或 Neo4j driver。

            try:

                parsed = rag_client.call(

                    "hybrid_search_filtered",

                    {

                        "query": claim,

                        "top_k": 5,

                        "mode": "vector",

                        "exclude_low_value": True,

                    },

                )

                chunks = parsed.get("chunks", []) if isinstance(parsed, dict) else parsed

                sources = chunks if isinstance(chunks, list) else []

            except Exception:

                pass



        for src in sources:

            cid = src.get("chunk_id", "")

            if cid and cid in seen_chunk_ids:

                continue

            if cid:

                seen_chunk_ids.add(cid)

            text = src.get("content", "") or src.get("text", "")

            if text:

                title = src.get("paper_title") or "unknown"

                extra_contexts.append(f"[{title}] {text}")



    return extra_contexts





# 同一回答最多修订几轮（每轮 3 次校验模型调用：编辑生成 + 抽取 + 校验）。

# v23 教训：修订本身会引入新的未校验内容（溶剂物性表数值等参数知识），

# 只修 1 轮时这些内容绕过校验直接出厂；v25 起每个版本都会重新校验，

# 该上限只控制成本（最坏 2 + 3*MAX 次调用），不再造成未检尾段。

GROUNDING_MAX_ATTEMPTS = int(os.environ.get("ORION_GROUNDING_MAX_ATTEMPTS", "1"))

# 低于该长度的最终答案不校验（寒暄/简答没有可校验的事实论断）。

_MIN_ANSWER_CHARS = 200

# 抽取的论断数上限，控制校验 prompt 规模。

# v31：改用"全部语句 + 分类"模式，避免漏抽表格/候选方案中的事实性内容。

# v61：从 60 降到 30，缩短最终 grounding 阶段耗时，避免前端长时间无响应。

_MAX_CLAIMS = int(os.environ.get("ORION_GROUNDING_MAX_CLAIMS", "30"))

# 接地校验总时间预算（秒）。 grounding 是最终回答前的阻塞节点，

# 超长报告+大量未通过论断时容易耗尽子进程 1800s 预算；设上限后 fail-open。

# v61：从 120s 降到 60s，在线对话场景优先保证响应速度。

_GROUNDING_BUDGET_SECONDS = float(os.environ.get("ORION_GROUNDING_BUDGET_SECONDS", "60"))

# 评测模式必须在外层 sample timeout 前主动结束 grounding，并为 checkpoint / 指标

# 序列化留出足够余量。若剩余预算小于完整 grounding 的经验下限，则明确标记

# skipped_budget，而不是启动一个注定被外层 timeout 截断的同步校验线程。

_GROUNDING_EVAL_RESERVE_SECONDS = float(

    os.environ.get("ORION_GROUNDING_EVAL_RESERVE_SECONDS", "20")

)

_GROUNDING_EVAL_MIN_FULL_BUDGET_SECONDS = float(

    os.environ.get("ORION_GROUNDING_EVAL_MIN_FULL_BUDGET_SECONDS", "90")

)

# fallback 检索最多处理几条未通过论断，控制 verify_parameter_claim / hybrid_search 调用次数。

_GROUNDING_FALLBACK_CLAIM_CAP = int(os.environ.get("ORION_GROUNDING_FALLBACK_CLAIM_CAP", "1"))



_EXTRACT_PROMPT = """\

从下面的【报告】中抽取所有可被外部证据判断真假的陈述，并分类为：

- factual：可被本次检索到的文献/图谱证据直接支持或证伪的事实陈述。

  包括：数值、配方组成、性能数据、机理断言、材料性质、作者/文献归属。

- synthesis：设计综合层内容，对检索证据不可直接推断。

  包括：候选方案标签绑定（如"候选方案A使用...""C1采用..."）、推荐排序、

        置信度、证据缺口评估、下一步建议、任务元描述、过渡句、假设说明。



要求：

- 逐条抽取，不要遗漏表格、列表、候选方案中的内容；

- 每条论断独立成句、自包含（补全主语，不用代词指代）；

- 优先保留含数值、组成、性能、文献归属的 factual 论断；

- factual 论断最多保留 {max_claims} 条，超出时优先保留含数值/来源的论断。



只输出 JSON：{{"claims": [{{"text": "论断1", "type": "factual|synthesis"}}, ...]}}



【报告】

{answer}

"""



_VERIFY_PROMPT = """\

你是严格的证据校验员。只依据【证据片段】判断每条【论断】能否被直接推出，

禁止使用你自己的知识。



判定规则（宁严勿宽）：

- verdict=1：证据片段明确、直接地推出该论断，关键事实（数值、组成、机理、归属）都能在证据中找到对应；

- verdict=0：证据片段只相关但未直接支持、关键信息缺失、证据不足、或论断包含证据中没有的推断。



只输出 JSON：{{"results": [{{"claim": "原文照抄", "verdict": 1, "reason": "一句简短理由"}}]}}



【证据片段】

{contexts}



【论断】

{claims}

"""





_EDIT_PROMPT = """\

你是修订助手。【回答】中有若干事实性论断未通过证据校验（无检索证据支持或与证据冲突）。

只依据【证据片段】，为每条未通过论断给出**最小编辑操作**。



操作类型：

- replace：用证据片段中的正确信息改写该句，或软化为方向性/假设性表述（如"有研究报道...""可能..."）（new 必填）；

- delete：删除该句（无证据支持时优先删除）。



禁止操作：

- 不要添加"（一般知识，未检证）"之类的标注，因为评估器仍会把原句当事实判为幻觉；

- 不要新增证据中没有的数值、表格、文献归属。



要求：

- find 必须是【回答】中**逐字存在**的连续文本（取论断所在的完整句子，含句末标点）；

- 只处理未通过论断：不修改任何其它内容；

- 每条论断至多一条编辑；

- 无证据支持时优先 delete，不要 soft annotate。



只输出 JSON：{{"edits": [{{"find": "...", "action": "replace|delete", "new": "..."}}]}}



【证据片段】

{contexts}



【未通过论断】

{listing}



【回答】

{answer}

"""





@dataclass

class GroundingReport:

    """一次接地校验的结果。"""



    total_claims: int = 0

    verified: int = 0

    ungrounded: list[dict[str, str]] = field(default_factory=list)

    skipped: str = ""  # 非空表示未执行校验及原因





class GroundingTimeoutError(TimeoutError):

    """grounding 的墙钟预算耗尽。



    Python 线程无法安全取消已经发出的 HTTP 请求；该异常只保证主执行链

    不再等待，并由调用方如实记录 fail-open 状态。

    """





def _check_claim_contracts(

    claim_contracts: list[ClaimContract],

    contexts: list[str],

) -> dict[str, Any]:

    """对结构化 claim contract 做轻量证据绑定检查，返回汇总结果。



    该检查不替代 LLM grounding，只作为 claim-level auditability 的元数据：

    - quote_bound 必须能在 contexts 中命中 raw_quote；

    - recommendation 类型 claim  evidence_binding 低于 condition_bound 时标记警告。

    """

    if not claim_contracts:

        return {"checked": False, "reason": "no_claim_contracts"}



    contexts_text = "\n".join(contexts)

    results = []

    warnings = []

    for claim in claim_contracts:

        result: dict[str, Any] = {

            "claim_id": claim.claim_id,

            "claim_type": claim.claim_type,

            "evidence_binding": claim.evidence_binding,

        }

        if claim.evidence_binding == "quote_bound":

            quote = claim.evidence_quote

            if quote and contexts_text.count(quote) == 1:

                result["quote_unique_hit"] = True

            elif quote:

                result["quote_unique_hit"] = False

                warnings.append(f"{claim.claim_id}: quote_not_unique")

            else:

                result["quote_unique_hit"] = False

                warnings.append(f"{claim.claim_id}: missing_quote")



        if claim.claim_type == "recommendation" and claim.evidence_binding in ("quote_bound",):

            warnings.append(f"{claim.claim_id}: recommendation_only_quote_bound")



        results.append(result)



    return {

        "checked": True,

        "total": len(claim_contracts),

        "results": results,

        "warnings": warnings,

    }





def _run_with_timeout(operation: Any, timeout_seconds: float) -> Any:

    """在不阻塞主执行链的前提下执行一次不可中断的外部调用。



    ``llm.invoke`` 是同步调用，单靠调用前后的 deadline 检查无法阻止一次

    调用越过总预算。这里使用守护线程实施真实墙钟上限；超时后的请求可能

    仍在服务端完成，因此绝不能把它视为已通过 grounding。

    """

    if timeout_seconds <= 0:

        raise GroundingTimeoutError("grounding budget exhausted before call")



    outcome: dict[str, Any] = {}



    def _target() -> None:

        try:

            outcome["value"] = operation()

        except BaseException as exc:  # 需把原始 LLM 异常回传给主线程

            outcome["error"] = exc



    worker = threading.Thread(target=_target, daemon=True)

    worker.start()

    worker.join(timeout_seconds)

    if worker.is_alive():

        raise GroundingTimeoutError("grounding call exceeded wall-clock budget")

    if "error" in outcome:

        raise outcome["error"]

    return outcome.get("value")





def _safe_error_summary(exc: BaseException) -> str:

    """返回不含请求头、URL 参数或凭据的异常类型链。"""

    parts: list[str] = []

    current: BaseException | None = exc

    seen: set[int] = set()

    while current is not None and id(current) not in seen and len(parts) < 5:

        seen.add(id(current))

        label = type(current).__name__

        status_code = getattr(current, "status_code", None)

        if status_code is not None:

            label += f"(status={status_code})"

        parts.append(label)

        current = current.__cause__ or current.__context__

    return " -> ".join(parts) or "UnknownError"





def _extract_json(text: str) -> dict | None:

    """从模型输出中稳健提取第一个 JSON 对象。"""

    start = text.find("{")

    end = text.rfind("}")

    if start < 0 or end <= start:

        return None

    try:

        data = json.loads(text[start : end + 1])

    except Exception:

        return None

    return data if isinstance(data, dict) else None





def check_answer_grounding(

    answer: str, contexts: list[str], llm: Any, max_claims: int = _MAX_CLAIMS

) -> GroundingReport:

    """抽取答案事实论断并逐条对照证据校验（2 次 LLM 调用）。异常向上抛。"""

    extract_out = llm.invoke(

        _EXTRACT_PROMPT.format(answer=answer, max_claims=max_claims)

    ).content

    data = _extract_json(str(extract_out))

    raw_claims = (data or {}).get("claims", [])

    factual_claims: list[str] = []

    for c in raw_claims:

        if isinstance(c, dict):

            text = str(c.get("text") or "").strip()

            ctype = str(c.get("type") or "factual").strip().lower()

        elif isinstance(c, str):

            text = c.strip()

            ctype = "factual"

        else:

            continue

        if text and ctype == "factual":

            factual_claims.append(text)

    if not factual_claims:

        return GroundingReport(skipped="no_claims")

    factual_claims = factual_claims[:max_claims]



    verify_out = llm.invoke(

        _VERIFY_PROMPT.format(

            contexts="\n\n".join(contexts),

            claims="\n".join(f"{i + 1}. {c}" for i, c in enumerate(factual_claims)),

        )

    ).content

    vdata = _extract_json(str(verify_out))

    results = (vdata or {}).get("results")

    if not isinstance(results, list):

        raise ValueError("grounding verify returned no results")



    report = GroundingReport(total_claims=len(factual_claims))

    verdicts: dict[str, str] = {}

    for item in results:

        if not isinstance(item, dict):

            continue

        claim = str(item.get("claim") or "").strip()

        if not claim:

            continue

        verdicts[claim] = "1" if str(item.get("verdict")) in ("1", "1.0", "true") else "0"

        if verdicts[claim] == "0":

            report.ungrounded.append(

                {"claim": claim, "reason": str(item.get("reason") or "")}

            )

    # 校验器漏判的论断按未通过计（宁严勿宽），reason 标注

    judged = set(verdicts)

    for c in factual_claims:

        if c not in judged:

            report.ungrounded.append({"claim": c, "reason": "校验器未覆盖该论断"})

    report.verified = report.total_claims - len(report.ungrounded)

    return report





def check_claim_contract_grounding(

    claim_contracts: list[ClaimContract],

    contexts: list[str],

    llm: Any,

) -> GroundingReport:

    """直接对结构化 ClaimContract 做一次语义校验，避免重新抽取整篇报告。



    contract 已携带 source_refs/raw_quote，因此优先只把绑定原文送给独立校验模型；

    只有 contract 缺少 quote 时才回退本轮完整 contexts。该路径把结构化任务从

    “整篇抽取 + 校验”两次 LLM 调用降为一次校验调用，并显著缩短 prompt。

    """



    contracts = [

        claim for claim in claim_contracts

        if claim.claim_text and claim.claim_text.strip()

    ]

    if not contracts:

        return GroundingReport(skipped="no_claim_contracts")



    report = GroundingReport(total_claims=len(contracts))

    llm_contracts: list[ClaimContract] = []

    # quote/condition-bound contract 必须已经由 v2 compiler 收缩成逐字引文。

    # 自由改写不能再由随机 LLM “补票”；更强的 computation/human binding 才保留

    # 独立语义 verifier 路径。

    for contract in contracts:

        claim_text = contract.claim_text.strip()

        raw_quotes = [

            str(source.raw_quote or "").strip()

            for source in contract.source_refs

            if str(source.raw_quote or "").strip()

        ]

        if contract.evidence_binding in {"quote_bound", "condition_bound"}:

            if len(raw_quotes) != 1:

                report.ungrounded.append(

                    {

                        "claim": claim_text,

                        "reason": (

                            "missing_authoritative_quote"

                            if not raw_quotes

                            else "multiple_quotes_require_explicit_computation_binding"

                        ),

                    }

                )

                continue

            semantic = enforce_quote_identity_v2(

                claim_text=claim_text,

                evidence_quote=raw_quotes[0],

                claim_type=contract.claim_type,

            )

            if semantic.accepted and claim_text == semantic.canonical_claim_text:

                report.verified += 1

            else:

                report.ungrounded.append(

                    {

                        "claim": claim_text,

                        "reason": semantic.reason or "quote_bound_claim_not_canonical",

                    }

                )

            continue

        if any(claim_text == quote for quote in raw_quotes):

            report.verified += 1

        else:

            llm_contracts.append(contract)



    if not llm_contracts:

        return report



    claims = [claim.claim_text.strip() for claim in llm_contracts]

    contract_contexts: list[str] = []

    for claim in llm_contracts:

        for source in claim.source_refs:

            quote = str(source.raw_quote or "").strip()

            if not quote:

                continue

            label = source.chunk_id or source.paper_id or "claim-evidence"

            contract_contexts.append(f"[{label}] {quote}")

    evidence_contexts = list(dict.fromkeys(contract_contexts)) or contexts



    verify_out = llm.invoke(

        _VERIFY_PROMPT.format(

            contexts="\n\n".join(evidence_contexts),

            claims="\n".join(f"{i + 1}. {claim}" for i, claim in enumerate(claims)),

        )

    ).content

    vdata = _extract_json(str(verify_out))

    results = (vdata or {}).get("results")

    if not isinstance(results, list):

        raise ValueError("claim-contract grounding verify returned no results")



    llm_ungrounded: list[dict[str, str]] = []

    # 独立校验模型偶尔会改动回显 claim 的标点/空格；当结果条数与 contract 数

    # 完全一致时按请求顺序绑定回权威 claim_text，只采信 verdict/reason，避免

    # 把格式性回显差异误判为“未覆盖”。结果缺失/多出时仍退回严格文本匹配。

    if len(results) == len(claims) and all(isinstance(item, dict) for item in results):

        for original_claim, item in zip(claims, results, strict=True):

            verdict = "1" if str(item.get("verdict")) in ("1", "1.0", "true") else "0"

            if verdict == "0":

                llm_ungrounded.append(

                    {

                        "claim": original_claim,

                        "reason": str(item.get("reason") or ""),

                    }

                )

        report.ungrounded.extend(llm_ungrounded)

        report.verified += len(claims) - len(llm_ungrounded)

        return report



    verdicts: dict[str, str] = {}

    for item in results:

        if not isinstance(item, dict):

            continue

        claim = str(item.get("claim") or "").strip()

        if not claim:

            continue

        verdicts[claim] = "1" if str(item.get("verdict")) in ("1", "1.0", "true") else "0"

        if verdicts[claim] == "0":

            llm_ungrounded.append(

                {"claim": claim, "reason": str(item.get("reason") or "")}

            )

    judged = set(verdicts)

    for claim in claims:

        if claim not in judged:

            llm_ungrounded.append({"claim": claim, "reason": "校验器未覆盖该 contract"})

    report.ungrounded.extend(llm_ungrounded)

    report.verified += len(claims) - len(llm_ungrounded)

    return report





def _generate_edits(

    answer: str, report: GroundingReport, contexts: list[str], llm: Any

) -> list[dict]:

    """让 LLM 为未通过论断生成最小编辑操作（1 次调用）。异常向上抛。"""

    listing = "\n".join(

        f"- {u['claim']}（{u['reason']}）" for u in report.ungrounded[:15]

    )

    out = llm.invoke(

        _EDIT_PROMPT.format(

            contexts="\n\n".join(contexts), listing=listing, answer=answer

        )

    ).content

    data = _extract_json(str(out))

    edits = (data or {}).get("edits")

    return [e for e in edits if isinstance(e, dict)] if isinstance(edits, list) else []





def _cleanup_empty_bullets(answer: str) -> str:

    """删除 grounding 编辑后可能留下的空 bullet（只有 [chunk_id] 无文本）。"""

    # 匹配行首 `- ` 后可选空白，接着一个或多个 `[chunk_id]`，行尾无其他文本。

    cleaned = re.sub(r"(?m)^\s*-\s*(\[\S+?\]\s*)+\s*$", "", answer)

    # 折叠连续空行

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip("\n")





def _apply_edits(answer: str, edits: list[dict]) -> tuple[str, int]:

    """把编辑操作以字符串替换应用到原文；未命中的操作跳过（fail-soft）。



    返回 (新文本, 实际应用条数)。未涉及内容逐字节保留——这是消除

    revision churn 的关键：修订不经过全篇重写。

    """

    applied = 0

    for e in edits:

        find = str(e.get("find") or "").strip()

        action = str(e.get("action") or "").strip()

        if not find or find not in answer:

            continue

        if action == "replace":

            new = str(e.get("new") or "").strip()

            if not new or new == find:

                continue

            answer = answer.replace(find, new, 1)

        elif action == "delete":

            answer = answer.replace(find, "", 1)

        else:

            continue

        applied += 1

    answer = _cleanup_empty_bullets(answer)

    return answer, applied





def _is_auth_error(exc: Exception) -> bool:

    """判断异常是否由鉴权/令牌失效导致，用于 grounding 模型降级。"""

    msg = str(exc).lower()

    markers = [

        "401", "402", "403", "authentication", "unauthorized", "forbidden",

        "token", "api key", "apikey", "expired", "invalid key",

        "insufficient", "balance", "quota", "limit",

        "令牌", "过期", "无权限", "余额", "不足", "额度",

    ]

    return any(m in msg for m in markers)





def _is_transient_transport_error(exc: Exception) -> bool:

    """识别 grounding 服务的瞬时网络/协议故障，允许一次模型降级。"""

    markers = {

        "APIConnectionError",

        "APITimeoutError",

        "ConnectError",

        "ConnectTimeout",

        "ReadError",

        "ReadTimeout",

        "RemoteProtocolError",

        "TimeoutException",

    }

    current: BaseException | None = exc

    seen: set[int] = set()

    while current is not None and id(current) not in seen:

        seen.add(id(current))

        if type(current).__name__ in markers:

            return True

        current = current.__cause__ or current.__context__

    return False





def deliver_terminal_result(state: dict, config: RunnableConfig | None = None):

    """只交付当前回合的成对结果；语义审查由主 Agent 或显式子任务完成。



    保留历史 grounding 接口供独立审查和回放使用。本出口不调用模型，不编辑

    科学结论，也不把格式/合同检查记为语义验证通过。

    """

    from langchain_core.messages import AIMessage

    from langgraph.graph import END

    from langgraph.types import Command

    from pydantic import TypeAdapter

    from orion.core.design_runtime import terminal_result

    from orion.task_evidence.contracts_v2 import TaskArtifact



    state = current_task_state(state)

    terminal = terminal_result(state)

    update = {

        "grounding_status": "not_requested",

        "grounding_error": None,

        "grounding_duration_seconds": 0.0,

        "grounding_fallback_used": False,

        "grounding_primary_error": None,

        "claim_contract_summary": {"semantic": {"mode": "not_requested"}},

    }

    if terminal is not None:

        body, artifact, call_id = terminal

        # 旧协议允许无 v2 artifact；一旦声明 v2，就必须通过已有科学合同。

        if isinstance(artifact, dict) and artifact.get("schema_version") == "2.0":

            TypeAdapter(TaskArtifact).validate_python(artifact)

        update.update(last_report=body, task_artifact=artifact,

                      terminal_committed=True, terminal_commit_tool_call_id=call_id)

        messages = state.get("messages") or []

        if not any(isinstance(m, AIMessage) and str(m.content or "") == body

                   and not m.tool_calls for m in messages):

            update["messages"] = [AIMessage(content=body)]

    return Command(update=update, goto=END)





# LangGraph 检查实际注解对象；恢复旧节点时仍须正确注入 RunnableConfig。

deliver_terminal_result.__annotations__["config"] = RunnableConfig





def make_grounding_node(llm: Any, fallback_llm: Any | None = None):

    """构造 grounding 节点（闭包注入校验 LLM，便于测试替换）。



    路由：review 判定"最终回答"后进入本节点；校验-编辑-复检循环全部在

    节点内闭环，结束后一律 → END（出厂的是校验分最优的版本）。



    参数：

        llm: 首选接地校验模型（推荐与生成模型不同源）。

        fallback_llm: 首选模型鉴权失败或出现瞬时网络/协议故障时的单次降级模型；

            未提供或降级也失败时按 fail-open 处理。

    """



    def grounding(state: dict, config: RunnableConfig | None = None):

        from langgraph.types import Command

        from langgraph.graph import END

        from langchain_core.messages import AIMessage as _AIMessage



        state = current_task_state(state)

        started_at = time.monotonic()

        claim_contract_summary: dict[str, Any] = {}

        claim_contracts: list[ClaimContract] = []

        claim_contract_parse_error: str | None = None

        fallback_used = False

        primary_error: str | None = None



        def _status_update(status: str, error: str | None = None, **extra: Any) -> dict[str, Any]:

            """为每个可观测 grounding 终态附加墙钟耗时与降级审计信息。"""

            result = {

                **extra,

                "grounding_status": status,

                "grounding_error": error,

                "grounding_duration_seconds": round(time.monotonic() - started_at, 3),

                "grounding_fallback_used": fallback_used,

                "grounding_primary_error": primary_error,

            }

            if claim_contract_summary:

                result["claim_contract_summary"] = claim_contract_summary

            return result



        if os.environ.get("ORION_GROUNDING_CHECK", "1").lower() in ("0", "false", "no"):

            # 即使跳过 grounding 校验，也要把 last_report 作为最终答案写回 messages，

            # 否则评估/前端拿到的 answer 会退化为中间过渡消息（v58 实测）。

            last_report = state.get("last_report")

            if last_report and len(str(last_report)) >= _MIN_ANSWER_CHARS:

                messages = list(state.get("messages", []))

                # 若 messages 最后一条就是该报告对应的 AI message，则无需重复追加。

                already_in_messages = any(

                    isinstance(m, _AIMessage) and str(m.content or "") == str(last_report)

                    for m in messages

                )

                if not already_in_messages:

                    return Command(

                        update=_status_update(

                            "disabled",

                            None,

                            messages=[_AIMessage(content=str(last_report))],

                        ),

                        goto=END,

                    )

            return Command(

                update=_status_update("disabled"),

                goto=END,

            )



        # grounding 预算不能超过评测样本剩余墙钟时间。旧 canary 曾配置

        # sample_timeout=180s、grounding_budget=300s，导致 grounding 尚未返回时

        # 外层先杀掉整条 graph，最终只剩空 agent_error。保留 5s 给 checkpoint/

        # 结果序列化，确保 fail-open 状态能被评测读取。

        effective_budget = _GROUNDING_BUDGET_SECONDS

        evaluation_budget_limited = False

        configurable = (config or {}).get("configurable", {}) if config is not None else {}

        config_started = (

            configurable.get("evaluation_started_at_monotonic")

            if isinstance(configurable, dict)

            else None

        )

        config_timeout = (

            configurable.get("evaluation_timeout_seconds")

            if isinstance(configurable, dict)

            else None

        )

        # 优先读取 context_prepare 已固化到 GraphState 的评测边界；同步 node

        # 在线程执行时 RunnableConfig 的自定义 configurable 可能被裁剪，但 state

        # 会随 checkpoint 稳定传递。

        eval_started = state.get("evaluation_started_at_monotonic")

        eval_timeout = state.get("evaluation_timeout_seconds")

        if eval_started is None:

            eval_started = config_started

        if eval_timeout is None:

            eval_timeout = config_timeout

        if eval_started is not None and eval_timeout is not None:

            elapsed = max(0.0, time.monotonic() - float(eval_started))

            sample_remaining = max(0.0, float(eval_timeout) - elapsed)

            evaluation_budget = max(

                0.0,

                sample_remaining - _GROUNDING_EVAL_RESERVE_SECONDS,

            )

            if evaluation_budget < effective_budget:

                evaluation_budget_limited = True

            effective_budget = min(effective_budget, evaluation_budget)

        deadline = time.monotonic() + effective_budget



        def _budget_terminal_status() -> tuple[str, str]:

            if evaluation_budget_limited:

                return "skipped_budget", "EvaluationGroundingBudgetExhausted"

            return "failed", "GroundingTimeoutError"



        def _budget_ok() -> bool:

            return time.monotonic() < deadline



        def _remaining_budget() -> float:

            return max(0.0, deadline - time.monotonic())



        def _run_budgeted(operation: Any) -> Any:

            """把整个节点的剩余预算传给每个不可中断的外部 LLM 调用。"""

            return _run_with_timeout(operation, _remaining_budget())



        current_llm = llm



        def _check(answer: str, contexts: list[str]) -> GroundingReport:

            """首选校验失败时，对鉴权或瞬时传输故障做一次可审计的 fallback。"""

            nonlocal current_llm, fallback_used, primary_error

            try:

                return check_answer_grounding(answer, contexts, current_llm)

            except Exception as exc:

                should_fallback = _is_auth_error(exc) or _is_transient_transport_error(exc)

                if fallback_llm is not None and should_fallback and current_llm is not fallback_llm:

                    primary_error = _safe_error_summary(exc)

                    fallback_used = True

                    print(

                        f"[grounding] 首选校验模型失败（{primary_error}），降级到 fallback 模型继续校验",

                        flush=True,

                    )

                    current_llm = fallback_llm

                    return check_answer_grounding(answer, contexts, current_llm)

                raise



        def _check_contracts(contexts: list[str]) -> GroundingReport:

            """结构化任务直接校验 claim contracts；异常降级语义与旧整篇路径一致。"""

            nonlocal current_llm, fallback_used, primary_error

            try:

                return check_claim_contract_grounding(claim_contracts, contexts, current_llm)

            except Exception as exc:

                should_fallback = _is_auth_error(exc) or _is_transient_transport_error(exc)

                if fallback_llm is not None and should_fallback and current_llm is not fallback_llm:

                    primary_error = _safe_error_summary(exc)

                    fallback_used = True

                    print(

                        f"[grounding] 首选 contract 校验模型失败（{primary_error}），降级到 fallback 模型继续校验",

                        flush=True,

                    )

                    current_llm = fallback_llm

                    return check_claim_contract_grounding(

                        claim_contracts, contexts, current_llm

                    )

                raise



        def _build_reflect_payload() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

            """把当前 messages 转为 reflect_and_update 可消费的简化格式。"""

            reflect_messages: list[dict[str, Any]] = []

            reflect_tool_calls: list[dict[str, Any]] = []

            pending_tool: dict[str, Any] | None = None

            for m in messages:

                if is_user_input(m):

                    reflect_messages.append({"role": "user", "content": str(m.content or "")})

                elif isinstance(m, AIMessage):

                    reflect_messages.append({"role": "assistant", "content": str(m.content or "")})

                    for tc in getattr(m, "tool_calls", []) or []:

                        name = tc.get("name") or (tc.get("function") or {}).get("name")

                        pending_tool = {"tool_name": name, "output": None}

                        reflect_tool_calls.append(pending_tool)

                elif isinstance(m, ToolMessage):

                    reflect_messages.append({"role": "tool", "content": str(m.content or "")})

                    if pending_tool is not None and pending_tool.get("output") is None:

                        pending_tool["output"] = str(m.content or "")

                        pending_tool = None

            return reflect_messages, reflect_tool_calls



        def _persist_reflection() -> None:

            """ grounding 出厂后自动沉淀反思到 Wiki 与长期记忆（fail-soft）。"""

            try:

                from orion.core.memory.wiki_reflection import reflect_and_update



                thread_id = (

                    (config or {}).get("configurable", {}).get("thread_id", "default")

                    if config is not None

                    else "default"

                )

                run_id = getattr(messages[last_human], "id", "") if last_human >= 0 else thread_id

                if not run_id:

                    run_id = thread_id

                reflect_messages, reflect_tool_calls = _build_reflect_payload()

                reflect_and_update(

                    session_id=thread_id,

                    run_id=run_id,

                    messages=reflect_messages,

                    tool_calls=reflect_tool_calls,

                )

                print(f"[grounding] reflection persisted for {thread_id}/{run_id}", flush=True)

            except Exception as exc:

                print(f"[grounding] reflection persistence failed: {exc}", flush=True)



        def _end(update: dict[str, Any] | None = None) -> Command:

            """返回 END，并按配置异步持久化非关键反思。"""

            # 评测/回归场景必须能显式关闭自动反思，避免每次 canary 都改写 wiki。

            # 默认仍保持生产行为（开启）；仅 ORION_AUTO_REFLECT=0/false/no 时跳过。

            auto_reflect = os.environ.get("ORION_AUTO_REFLECT", "1").lower() not in {

                "0",

                "false",

                "no",

            }

            if auto_reflect:

                # Wiki reflection 会额外发起一次 LLM 调用。它不能延长 grounding

                # 预算耗尽后的用户可见延迟，也不应阻塞 LangGraph 的 terminal event。

                threading.Thread(

                    target=_persist_reflection,

                    name="orion-wiki-reflection",

                    daemon=True,

                ).start()

            return Command(update=update or {}, goto=END)



        def _emit_best(

            best_answer: str,

            best_report: GroundingReport,

            attempts: int,

            reason: str,

            answer_msg_id: str,

            *,

            status: str = "completed",

            error: str | None = None,

        ):

            print(

                f"[grounding] {reason}: final {best_report.verified}/{best_report.total_claims} "

                f"claims verified after {attempts} edit round(s)",

                flush=True,

            )

            update: dict[str, Any] = {

                "grounding_attempts": attempts,

                **_status_update(status, error),

            }

            report_in_messages = any(

                isinstance(message, AIMessage) and str(message.content or "") == best_answer

                for message in messages

            )

            if best_answer != answer:

                if answer_from_last_report:

                    update["last_report"] = best_answer

                if answer_msg_id:

                    # 替换原答案消息，避免历史中出现多份重复报告

                    update["messages"] = [

                        RemoveMessage(id=answer_msg_id),

                        AIMessage(content=best_answer),

                    ]

                else:

                    # answer 来自 state.last_report，没有对应 AI 消息可替换，直接追加最终答案

                    update["messages"] = [AIMessage(content=best_answer)]

            elif answer_from_last_report and not report_in_messages:

                # review 会移除 LLM 对工具报告的重复复述；即使无需编辑，也必须将

                # 校验后的终端工具报告写回消息历史，供前端/CLI 读取最终答案。

                update["messages"] = [AIMessage(content=best_answer)]

            return _end(update)



        def _emit_contract_hold(

            report: GroundingReport,

            reason: str,

            answer_msg_id: str,

            *,

            error: str,

        ):

            """ClaimContract 未通过时替换原答案，禁止失败后继续出厂。"""



            hold_answer = (

                "## 证据核验暂停\n\n"

                "至少一条 ClaimContract 未通过确定性或独立语义支持门禁；"

                "原结论已停止输出。请补充可逐字绑定的证据，或将推断明确降级为待验证假设。"

            )

            return _emit_best(

                hold_answer,

                report,

                0,

                reason,

                answer_msg_id,

                status="failed",

                error=error,

            )



        # 校验对象：优先使用 state 中保存的终端工具报告（如 synthesize_design_report 产出），

        # 避免 LLM 在 ToolMessage 后又重复输出一次报告导致重复。

        messages = list(state.get("messages", []))

        last_report = state.get("last_report")

        from langchain_core.messages import AIMessage



        answer_msg_id = ""

        answer = ""

        answer_from_last_report = bool(

            last_report and len(str(last_report)) >= _MIN_ANSWER_CHARS

        )

        if answer_from_last_report:

            answer = str(last_report)

            # 尽量把 last_report 挂载到本轮最长的 AI 消息上，便于后续替换该消息。

            last_human = max(

                (i for i, m in enumerate(messages) if is_user_input(m)),

                default=-1,

            )

            candidates = [

                m

                for m in messages[last_human + 1 :]

                if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None)

            ]

            if candidates:

                answer_msg = max(candidates, key=lambda m: len(str(m.content)))

                answer_msg_id = getattr(answer_msg, "id", "")

        else:

            last_human = max(

                (i for i, m in enumerate(messages) if is_user_input(m)),

                default=-1,

            )

            candidates = [

                m

                for m in messages[last_human + 1 :]

                if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None)

            ]

            if not candidates:

                return Command(update=_status_update("skipped"), goto=END)

            answer_msg = max(candidates, key=lambda m: len(str(m.content)))

            answer_msg_id = getattr(answer_msg, "id", "")

            answer = str(answer_msg.content)

            if len(answer) < _MIN_ANSWER_CHARS:

                return Command(update=_status_update("skipped"), goto=END)



        # Layer 1 auditability: 若 state 中有结构化 claim_contracts，做轻量绑定检查。

        contexts = extract_contexts(messages, answer=answer)

        task_artifact = state.get("task_artifact")

        if task_artifact:

            try:

                raw_claims = task_artifact.get("claim_contracts") if isinstance(task_artifact, dict) else getattr(task_artifact, "claim_contracts", None)

                if raw_claims:

                    claim_contracts = [

                        ClaimContract.model_validate(c) if isinstance(c, dict) else c

                        for c in raw_claims

                    ]

                    claim_contract_summary = _check_claim_contracts(claim_contracts, contexts)

            except Exception as exc:

                claim_contract_parse_error = _safe_error_summary(exc)

                claim_contract_summary = {

                    "checked": False,

                    "reason": claim_contract_parse_error,

                }



        if claim_contract_parse_error:

            return _emit_contract_hold(

                GroundingReport(skipped="invalid_claim_contract"),

                "claim-contract parsing failed",

                answer_msg_id,

                error="ClaimContractParseError",

            )

        if not contexts and not claim_contracts:

            return Command(update=_status_update("skipped"), goto=END)



        # v2 compiler 输出的 claim_text 与 raw_quote 逐字相同，可在本地确定性完成；

        # 该路径既不需要外部 contexts，也不应被 LLM 预算门禁误伤。

        contracts_are_verbatim = bool(claim_contracts) and all(

            any(

                claim.claim_text.strip() == str(source.raw_quote or "").strip()

                for source in claim.source_refs

                if str(source.raw_quote or "").strip()

            )

            for claim in claim_contracts

        )

        if contracts_are_verbatim:

            best_report = _check_contracts(contexts)

            claim_contract_summary["semantic"] = {

                "total": best_report.total_claims,

                "verified": best_report.verified,

                "ungrounded": best_report.ungrounded,

                "mode": "contract_first_deterministic",

            }

            if best_report.ungrounded:

                return _emit_contract_hold(

                    best_report,

                    "verbatim claim contract policy rejected output",

                    answer_msg_id,

                    error="ClaimContractSemanticVerificationFailed",

                )

            return _emit_best(

                answer,

                best_report,

                0,

                "verbatim claim contracts verified deterministically",

                answer_msg_id,

            )



        if (

            evaluation_budget_limited

            and effective_budget < _GROUNDING_EVAL_MIN_FULL_BUDGET_SECONDS

        ):

            status, error = _budget_terminal_status()

            if claim_contracts:

                return _emit_contract_hold(

                    GroundingReport(skipped="insufficient_evaluation_budget"),

                    "claim-contract verification budget unavailable",

                    answer_msg_id,

                    error="ClaimContractVerificationUnavailable",

                )

            return _emit_best(

                answer,

                GroundingReport(skipped="insufficient_evaluation_budget"),

                0,

                "evaluation budget too small for full grounding",

                answer_msg_id,

                status=status,

                error=error,

            )



        if not _budget_ok():

            status, error = _budget_terminal_status()

            if claim_contracts:

                return _emit_contract_hold(

                    GroundingReport(skipped="grounding_timeout"),

                    "claim-contract verification budget exhausted",

                    answer_msg_id,

                    error="ClaimContractVerificationUnavailable",

                )

            return Command(

                update=_status_update(status, error),

                goto=END,

            )



        try:

            if claim_contracts:

                best_report = _run_budgeted(lambda: _check_contracts(contexts))

            else:

                best_report = _run_budgeted(lambda: _check(answer, contexts))

        except GroundingTimeoutError:

            status, error = _budget_terminal_status()

            if claim_contracts:

                return _emit_contract_hold(

                    GroundingReport(skipped="grounding_timeout"),

                    "claim-contract semantic check timed out",

                    answer_msg_id,

                    error="ClaimContractVerificationUnavailable",

                )

            return _emit_best(

                answer,

                GroundingReport(skipped="grounding_timeout"),

                0,

                "grounding check budget exhausted",

                answer_msg_id,

                status=status,

                error=error,

            )

        except Exception as exc:  # noqa: BLE001 — 普通报告保留旧降级，contract 必须 fail-closed

            error_summary = _safe_error_summary(exc)

            if claim_contracts:

                return _emit_contract_hold(

                    GroundingReport(skipped="grounding_error"),

                    "claim-contract semantic check failed",

                    answer_msg_id,

                    error="ClaimContractVerificationUnavailable",

                )

            print(

                f"[grounding] check failed, pass through: {error_summary}",

                flush=True,

            )

            return Command(

                update=_status_update("failed", error_summary),

                goto=END,

            )



        # 若无需修订，直接出厂原答案（不新增消息）

        if not best_report.ungrounded:

            if claim_contracts:

                claim_contract_summary["semantic"] = {

                    "total": best_report.total_claims,

                    "verified": best_report.verified,

                    "ungrounded": [],

                    "mode": "contract_first",

                }

            return _emit_best(answer, best_report, 0, "no ungrounded claims", answer_msg_id)



        # 结构化报告由 deterministic compiler + ClaimContract 约束生成。独立语义校验

        # 仍有未通过 contract 时必须 fail-closed；不能用整篇重抽取或跨文献检索“救活”

        # 已绑定到特定 source_ref 的 contract。

        if claim_contracts:

            claim_contract_summary["semantic"] = {

                "total": best_report.total_claims,

                "verified": best_report.verified,

                "ungrounded": best_report.ungrounded,

                "mode": "contract_first",

            }

            return _emit_contract_hold(

                best_report,

                "claim-contract semantic check rejected output",

                answer_msg_id,

                error="ClaimContractSemanticVerificationFailed",

            )



        # Fallback retrieval for ungrounded parameter claims

        if best_report.ungrounded and _budget_ok():

            param_claims = [u["claim"] for u in best_report.ungrounded if _is_parameter_claim(u["claim"])]

            if param_claims:

                extra = _retrieve_for_claims(param_claims[:_GROUNDING_FALLBACK_CLAIM_CAP])

                if extra:

                    print(f"[grounding] fallback retrieval added {len(extra)} contexts", flush=True)

                    contexts = contexts + extra

                    if _budget_ok():

                        try:

                            best_report = _run_budgeted(lambda: _check(answer, contexts))

                        except GroundingTimeoutError:

                            status, error = _budget_terminal_status()

                            return _emit_best(

                                answer,

                                best_report,

                                0,

                                "budget exhausted during fallback re-check",

                                answer_msg_id,

                                status=status,

                                error=error,

                            )

                        except Exception as exc:

                            print(f"[grounding] fallback re-check failed: {exc}", flush=True)



        best_answer = answer

        attempts = 0

        while best_report.ungrounded and attempts < GROUNDING_MAX_ATTEMPTS and _budget_ok():

            print(

                f"[grounding] {len(best_report.ungrounded)}/{best_report.total_claims} "

                f"ungrounded claims, edit round #{attempts + 1}",

                flush=True,

            )

            try:

                edits = _run_budgeted(

                    lambda: _generate_edits(best_answer, best_report, contexts, current_llm)

                )

                new_answer, applied = _apply_edits(best_answer, edits)

                if not applied:

                    print("[grounding] no edits applicable, stop", flush=True)

                    break

                attempts += 1

                if not _budget_ok():

                    best_answer, best_report = new_answer, best_report

                    break

                report = _run_budgeted(lambda: _check(new_answer, contexts))

            except GroundingTimeoutError:

                status, error = _budget_terminal_status()

                return _emit_best(

                    best_answer,

                    best_report,

                    attempts,

                    "budget exhausted during edit round",

                    answer_msg_id,

                    status=status,

                    error=error,

                )

            except Exception as exc:  # noqa: BLE001 — 中途失败保留当前最优

                print(f"[grounding] edit round failed, keep best: {exc}", flush=True)

                break

            improved = len(report.ungrounded) < len(best_report.ungrounded)

            print(

                f"[grounding] round {attempts}: {len(report.ungrounded)}/{report.total_claims} "

                f"ungrounded ({'accept' if improved else 'reject'}, "

                f"best so far {len(best_report.ungrounded)})",

                flush=True,

            )

            if improved:

                best_answer, best_report = new_answer, report



        if not _budget_ok():

            status, error = _budget_terminal_status()

            return _emit_best(

                best_answer,

                best_report,

                attempts,

                "budget exhausted",

                answer_msg_id,

                status=status,

                error=error,

            )

        return _emit_best(best_answer, best_report, attempts, "complete", answer_msg_id)



    return grounding

