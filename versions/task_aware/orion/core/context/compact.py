from __future__ import annotations



import os



from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage





def _is_protected_context_message(message: AnyMessage) -> bool:

    """Return whether a message must survive context compaction unchanged."""



    metadata = getattr(message, "additional_kwargs", {}) or {}

    return bool(metadata.get("protected_context", False))





# 各 provider 的"每字符 token 数"经验系数：(其他字符, CJK 字符)。

# - deepseek：官方文档明确"1 个英文字符 ≈ 0.3 token，1 个中文字符 ≈ 0.6 token"；

# - openai：cl100k/o200k 对英文约 4 字符/token，对 CJK 约 1 token/字；

# - anthropic：官方称 1 token ≈ 3.5 个英文字符，CJK 密度与 OpenAI 同量级；

# - google：官方称 1 token ≈ 4 字符，CJK 同量级。

# 未知 provider 用通用兜底（偏保守：宁可高估，早触发压缩比晚触发安全）。

_PROVIDER_RATIOS: dict[str, tuple[float, float]] = {

    "deepseek": (0.3, 0.6),

    "openai": (0.25, 1.0),

    "anthropic": (0.286, 1.0),

    "google": (0.25, 1.0),

}

_DEFAULT_RATIOS: tuple[float, float] = (0.25, 1.0)





def _is_cjk(char: str) -> bool:

    """CJK 及东亚文字（中/日/韩）：token 密度显著高于拉丁字母，需单独计权。"""

    code = ord(char)

    return (

        0x4E00 <= code <= 0x9FFF      # CJK 统一表意文字（中文）

        or 0x3400 <= code <= 0x4DBF   # CJK 扩展 A

        or 0x3040 <= code <= 0x30FF   # 日文平假名 / 片假名

        or 0xAC00 <= code <= 0xD7AF   # 韩文谚文

        or 0xF900 <= code <= 0xFAFF   # CJK 兼容表意文字

    )





def estimate_token_count(

    messages: list[AnyMessage], provider: str | None = None

) -> int:

    """按当前 provider 校准的 token 估算（区分 CJK 与其他字符）。



    早期实现是"字符数 // 4"，对英文尚可，对中文严重低估约 2.4 倍（DeepSeek

    官方口径中文 0.6 token/字，而 //4 相当于 0.25）。这里改为按 provider 选

    系数、并对 CJK（中/日/韩）单独计权，使估算跟随可切换的模型配置

    （``ORION_AGENT_PROVIDER``，默认 deepseek，与 llms.py 的默认一致）。



    仍是估算而非精确分词；精确计数需目标模型的 tokenizer。对"是否触发压缩"

    的阈值判断而言此精度足够（阈值本身留 20% 余量）。

    """

    name = (

        provider or os.environ.get("ORION_AGENT_PROVIDER") or "deepseek"

    ).strip().lower()

    other_ratio, cjk_ratio = _PROVIDER_RATIOS.get(name, _DEFAULT_RATIOS)

    total = 0.0

    for message in messages:

        text = str(getattr(message, "content", ""))

        cjk_count = sum(1 for char in text if _is_cjk(char))

        total += cjk_count * cjk_ratio + (len(text) - cjk_count) * other_ratio

    return max(1, int(round(total)))





def compact_messages(

    messages: list[AnyMessage],

    token_count: int,

    context_limit: int,

    keep_last_turns: int = 4,

    preserve_evidence: bool = True,

) -> tuple[list[AnyMessage], int]:

    """

    压缩历史消息，保留系统提示、首条用户消息、最近 N 轮以及带引用的证据。

    返回 (new_messages, new_token_estimate)。

    """

    if token_count < context_limit * 0.8 or len(messages) <= 2 + keep_last_turns * 2:

        # 未触发压缩阈值或消息太少，直接返回

        return messages, token_count



    # 1. prefix: system + first user

    prefix: list[AnyMessage] = []

    idx = 0

    while idx < len(messages) and len(prefix) < 2:

        if isinstance(messages[idx], SystemMessage):

            prefix.append(messages[idx])

        elif isinstance(messages[idx], HumanMessage) and not any(

            isinstance(m, HumanMessage) for m in prefix

        ):

            prefix.append(messages[idx])

        idx += 1



    # 2. suffix: last keep_last_turns turns

    suffix: list[AnyMessage] = []

    turn_count = 0

    for m in reversed(messages):

        suffix.insert(0, m)

        if isinstance(m, HumanMessage):

            turn_count += 1

            if turn_count >= keep_last_turns:

                break



    # 3. summarizable: messages between prefix and suffix. Capability-specific

    # stage context can be marked as protected by the caller; such messages are

    # excluded from summarization and restored unchanged below.

    prefix_len = len(prefix)

    suffix_len = len(suffix)

    middle = messages[prefix_len : len(messages) - suffix_len]

    protected = [m for m in messages if _is_protected_context_message(m)]

    middle = [m for m in middle if not _is_protected_context_message(m)]



    # 4. preserve evidence tool messages in middle, but fold them into summary text

    #    rather than keeping raw ToolMessage, to avoid orphan tool_call sequences.

    evidence_texts: list[str] = []

    evidence: list[AnyMessage] = []

    if preserve_evidence:

        for m in middle:

            if isinstance(m, ToolMessage) and (

                "[" in str(m.content) or "chunk_id" in str(m.content)

            ):

                evidence.append(m)

                excerpt = str(getattr(m, "content", ""))[:400].replace("\n", " ")

                name = getattr(m, "name", None) or "evidence"

                evidence_texts.append(f"- {name}: {excerpt}")



    # 5. Build summary text from middle (excluding preserved evidence)

    middle_to_summarize = [m for m in middle if m not in evidence]

    full_summary = "\n".join(

        f"{m.type}: {str(m.content)[:500]}" for m in middle_to_summarize

    )

    middle_chars = sum(len(str(getattr(m, "content", ""))) for m in middle_to_summarize)



    # 仅当详细摘要真正比原始内容更简短时才使用；否则折叠为计数提示，

    # 确保压缩后的 token 数严格低于原始消息。

    if full_summary.strip() and len(full_summary) < middle_chars:

        summary_text = full_summary

    elif middle_to_summarize:

        summary_text = f"（已折叠 {len(middle_to_summarize)} 条早期消息）"

    else:

        summary_text = "（无额外历史内容）"



    if evidence_texts:

        summary_text += "\n\n保留的原文证据：\n" + "\n".join(evidence_texts)



    summary_msg = SystemMessage(content=f"历史对话摘要：\n{summary_text}")



    new_messages = prefix + [summary_msg] + suffix



    # Protected stage context (for example the capability blocks required by the

    # current call) must not disappear merely because ordinary history is compacted.

    existing_ids = {id(m) for m in new_messages}

    to_restore = [m for m in protected if id(m) not in existing_ids]

    if to_restore:

        insert_at = 1 if new_messages and isinstance(new_messages[0], SystemMessage) else 0

        new_messages[insert_at:insert_at] = to_restore



    new_count = estimate_token_count(new_messages)

    return new_messages, new_count

