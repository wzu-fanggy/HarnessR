"""科学对象身份的高精度、确定性解析。

本模块只从原文中的显式 formulation/variant 标签建立对象身份。它不调用 LLM、
RAG 或 KG，也不会把 ``paper_id``、candidate 容器归属或泛化描述（例如
``optimized electrolyte``）当成同一科学对象的证明。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


_CONNECTED_LABEL_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<label>"
    r"(?:[A-Za-z]*\d+(?:\.\d+)?[A-Za-z0-9]*|[A-Z]{2,}[A-Za-z0-9]*|Na[A-Z][A-Za-z0-9]*)"
    r"(?:\s*[+/_-]\s*[A-Za-z0-9.]+){1,4}"
    r")"
    r"(?![A-Za-z0-9])"
)
_DESIGNATED_LABEL_RE = re.compile(
    r"(?:designated\s+as|denoted\s+as|referred\s+to\s+as|called)\s+"
    r"(?P<label>[A-Za-z0-9][A-Za-z0-9.+/_-]{1,47})",
    re.IGNORECASE,
)
_NAMED_ELECTROLYTE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<label>[A-Za-z0-9][A-Za-z0-9.+/_-]{1,47})"
    r"\s+(?:electrolyte|system)\b",
    re.IGNORECASE,
)

# A stricter, typed extractor used by the LEDGER object compiler.  The legacy
# scientific-object extractor intentionally remains broader for compatibility,
# but it may surface electrode/cell identifiers such as Na||NVOPF.  Those are
# useful scientific-object labels in general, yet they must not be confused with
# the electrolyte formulation object that authorizes an electrolyte decision.
_ELECTROLYTE_OBJECT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<label>[A-Za-z0-9][A-Za-z0-9.+/_-]{1,47}(?:-(?:containing|based))?)"
    r"\s+(?:electrolytes?|electrolyte\s+systems?)\b",
    re.IGNORECASE,
)
_ELECTROLYTE_DESIGNATION_WINDOW_RE = re.compile(
    r"(?:electrolytes?|electrolyte\s+systems?).{0,120}?"
    r"(?:designated\s+as|denoted\s+as|referred\s+to\s+as|called)\s+"
    r"(?P<body>.{1,220}?)(?=(?:\.(?=\s|$)|;|$))",
    re.IGNORECASE,
)
_DESIGNATION_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<label>[A-Za-z0-9][A-Za-z0-9.+/_-]{1,47})(?![A-Za-z0-9])"
)
_GENERIC_ELECTROLYTE_OBJECT_LABELS = {
    "baseline",
    "blank",
    "control",
    "optimized",
    "modified",
    "functional",
    "corresponding",
    "reference",
    "electrolyte",
    "electrolytes",
    "system",
    "systems",
    "single-salt",
    "double-salt",
    "dual-salt",
}


def normalize_scientific_object_label(label: str) -> str:
    """把显式对象标签规范化为可跨 chunk 比较的稳定形式。"""

    value = str(label or "").strip().strip(".,;:()[]{}")
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"\s*([+/_-])\s*", r"\1", value)
    value = value.casefold().replace("/", "-").replace("_", "-")
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def _is_specific_label(
    raw_label: str,
    normalized_label: str,
    *,
    explicit_naming_context: bool = False,
) -> bool:
    """排除不足以唯一标识 formulation variant 的泛化短语/普通词组。

    ``explicit_naming_context`` 仅用于原文明示的 ``designated as X`` / ``X
    electrolyte`` 结构。此时允许 ``TDLSi``、``PEMP`` 这类不含数字的论文内
    variant 名称；普通游离 token 仍保持原有高精度规则，避免把常见化学词或
    泛化描述扩张成对象身份。
    """

    if not normalized_label or len(normalized_label) < 3 or len(normalized_label) > 64:
        return False
    # 配方标签通常保留化学缩写中的大写字母；纯小写的连字符词组（例如
    # ``2-sulfobenzoic``）或图号片段（例如 ``4g-i``）不应被当成对象 ID。
    if not any(char.isupper() for char in raw_label):
        return False
    if explicit_naming_context:
        uppercase_count = sum(char.isupper() for char in raw_label)
        # "Baseline electrolyte" / "Optimized electrolyte" 只有首字母大写，
        # 而论文内缩写型 variant（TDLSi/PEMP/FPM）通常含多个大写字母。
        if uppercase_count >= 2 and raw_label.casefold() not in {
            "baseline",
            "blank",
            "control",
            "optimized",
            "reference",
            "standard",
        }:
            return True
    if any(char.isdigit() for char in normalized_label):
        return True
    if "+" in normalized_label:
        return True
    if normalized_label.endswith("-e"):
        return True
    parts = [item for item in normalized_label.split("-") if item]
    return len(parts) >= 3


def extract_explicit_scientific_object_labels(text: str) -> list[str]:
    """提取高特异性的显式 formulation/variant 标签。

    返回值已经规范化。函数宁可漏掉只靠上下文代词表达的对象，也不推断其身份。
    """

    if not text:
        return []
    labels: set[str] = set()
    for pattern, explicit_naming_context in (
        (_CONNECTED_LABEL_RE, False),
        (_DESIGNATED_LABEL_RE, True),
        (_NAMED_ELECTROLYTE_RE, True),
    ):
        for match in pattern.finditer(text):
            raw_label = match.group("label")
            label = normalize_scientific_object_label(raw_label)
            if _is_specific_label(
                raw_label,
                label,
                explicit_naming_context=explicit_naming_context,
            ):
                labels.add(label)
    return sorted(labels, key=lambda item: (-len(item), item))


def _is_specific_electrolyte_object_label(raw_label: str, normalized_label: str) -> bool:
    """High-precision filter for electrolyte-formulation object labels only."""

    if not normalized_label or normalized_label in _GENERIC_ELECTROLYTE_OBJECT_LABELS:
        return False
    if len(normalized_label) < 2 or len(normalized_label) > 64:
        return False

    # Phrases such as ``SA-based`` and ``TDFP-containing`` are useful typed
    # formulation references when the stem itself is a specific chemical/variant
    # token. Purely generic lowercase phrases (``carbonate-based``) are not.
    for suffix in ("-based", "-containing"):
        if normalized_label.endswith(suffix):
            stem = normalized_label[: -len(suffix)]
            if stem in _GENERIC_ELECTROLYTE_OBJECT_LABELS or len(stem) < 2:
                return False
            return any(char.isupper() for char in raw_label) or any(
                char.isdigit() for char in raw_label
            )

    # Ordinary electrolyte labels should look like a code, composition shorthand,
    # or explicit chemistry token rather than an adjective/English noun.
    return (
        any(char.isupper() for char in raw_label)
        or any(char.isdigit() for char in raw_label)
        or any(token in raw_label for token in ("+", "/", "_"))
    )


def extract_explicit_electrolyte_object_labels(text: str) -> list[str]:
    """Extract only explicit *electrolyte formulation* object labels.

    This typed extractor intentionally excludes electrode/full-cell identifiers
    that the broader :func:`extract_explicit_scientific_object_labels` may return.
    It accepts labels syntactically attached to ``electrolyte`` and labels in an
    explicit electrolyte designation list. It does not resolve pronouns,
    same-paper references, or composition-derived aliases.
    """

    if not text:
        return []
    labels: set[str] = set()

    for match in _ELECTROLYTE_OBJECT_RE.finditer(text):
        raw_label = match.group("label")
        label = normalize_scientific_object_label(raw_label)
        if _is_specific_electrolyte_object_label(raw_label, label):
            labels.add(label)

    for window in _ELECTROLYTE_DESIGNATION_WINDOW_RE.finditer(text):
        body = window.group("body")
        for match in _DESIGNATION_TOKEN_RE.finditer(body):
            raw_label = match.group("label")
            label = normalize_scientific_object_label(raw_label)
            if _is_specific_electrolyte_object_label(raw_label, label):
                labels.add(label)

    return sorted(labels, key=lambda item: (-len(item), item))


def scientific_object_id(*, paper_id: str, label: str) -> str:
    """由论文身份和显式 variant label 生成稳定、可审计的对象 ID。"""

    normalized_label = normalize_scientific_object_label(label)
    paper = str(paper_id or "unknown").strip()
    digest = hashlib.sha256(f"{paper}|{normalized_label}".encode("utf-8")).hexdigest()[:12]
    return f"obj:{digest}:{normalized_label}"


def resolve_shared_explicit_object(
    *,
    paper_id: str,
    formulation_text: str,
    evidence_texts: list[str],
) -> tuple[str | None, str | None]:
    """只在配方证据与后续证据共享唯一显式标签时解析对象。

    若没有共享标签、共享标签多于一个，或证据只通过泛化称谓指向对象，则返回
    ``(None, None)``。这样同论文多 variant 的比较段落不会被错误压成单一对象。
    """

    formulation_labels = set(extract_explicit_scientific_object_labels(formulation_text))
    if not formulation_labels:
        return None, None
    downstream_labels: set[str] = set()
    for text in evidence_texts:
        downstream_labels.update(extract_explicit_scientific_object_labels(text))
    shared = formulation_labels.intersection(downstream_labels)
    if len(shared) != 1:
        return None, None
    label = next(iter(shared))
    return scientific_object_id(paper_id=paper_id, label=label), label


def text_explicitly_names_object(text: str, label: str) -> bool:
    """检查一段原文是否逐字/规范化后显式命名给定对象标签。"""

    normalized = normalize_scientific_object_label(label)
    return normalized in set(extract_explicit_scientific_object_labels(text))


def _record_text(record: dict[str, Any]) -> str:
    return str(
        record.get("text")
        or record.get("excerpt")
        or record.get("content")
        or ""
    )


def _record_paper_id(record: dict[str, Any]) -> str:
    chunk_id = str(record.get("chunk_id") or "").strip()
    if chunk_id and "::" in chunk_id:
        return chunk_id.split("::", 1)[0]
    return str(record.get("paper_id") or record.get("paper_title") or "").strip()


def resolve_candidate_payload_object_identity(
    raw_candidate: dict[str, Any],
) -> dict[str, Any]:
    """从候选已有 evidence 中重新获取唯一显式 scientific-object identity。

    该函数用于 enrichment/screening 边界：Agent 可能在初次候选生成后补充 property
    或 mechanism 证据，因此对象身份必须允许在后续阶段再次解析。只有同论文的
    formulation evidence 与 downstream evidence 共享唯一显式 variant label 时才赋值；
    未解析、别名粒度不一致或多 variant 比较均保持 unknown。
    """

    candidate = dict(raw_candidate)
    evidence = [
        dict(item) if isinstance(item, dict) else item
        for item in candidate.get("evidence") or []
    ]
    dict_evidence = [item for item in evidence if isinstance(item, dict)]
    formulations = [
        item
        for item in dict_evidence
        if str(item.get("evidence_kind") or "") == "formulation"
    ]
    if not formulations:
        candidate["evidence"] = evidence
        return candidate

    resolutions: dict[str, tuple[str, str]] = {}
    for formulation in formulations:
        paper_id = _record_paper_id(formulation)
        if not paper_id:
            continue
        downstream = [
            item
            for item in dict_evidence
            if item is not formulation
            and _record_paper_id(item) == paper_id
            and str(item.get("evidence_kind") or "")
            in {"property", "mechanism", "condition", "risk"}
        ]
        object_id, label = resolve_shared_explicit_object(
            paper_id=paper_id,
            formulation_text=_record_text(formulation),
            evidence_texts=[_record_text(item) for item in downstream],
        )
        if object_id and label:
            resolutions[object_id] = (paper_id, label)

    if len(resolutions) != 1:
        candidate["evidence"] = evidence
        return candidate

    object_id, (paper_id, label) = next(iter(resolutions.items()))
    if not str(candidate.get("scientific_object_id") or "").strip():
        candidate["scientific_object_id"] = object_id

    rebound: list[Any] = []
    for item in evidence:
        if not isinstance(item, dict):
            rebound.append(item)
            continue
        record = dict(item)
        if (
            not str(record.get("scientific_object_id") or "").strip()
            and _record_paper_id(record) == paper_id
            and text_explicitly_names_object(_record_text(record), label)
        ):
            # Resolver 只补全缺失身份；已有显式 metadata 必须保留，哪怕它与
            # 文本解析结果冲突。冲突属于 runtime authority gate 的输入，不能在
            # acquisition 阶段被静默“修正”掉。
            record["scientific_object_id"] = object_id
        rebound.append(record)
    candidate["evidence"] = rebound

    inferred_fields = list(candidate.get("inferred_fields") or [])
    marker = "scientific_object_id_from_candidate_evidence_v1"
    if marker not in inferred_fields:
        inferred_fields.append(marker)
    candidate["inferred_fields"] = inferred_fields
    return candidate
