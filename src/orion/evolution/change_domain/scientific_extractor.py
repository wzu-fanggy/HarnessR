"""从规范科学字段到 Harness 源码对象的确定性 coupling 提取。

该层不尝试理解任意自然语言语义，而只使用任务/证据合同中已经存在的规范字段
（例如 ``temperature_c``、``electrode``、``cell_configuration``）作为锚点，检查
代码与测试是否显式引用这些字段。这样可以发现源码 call graph 不一定连接、但共享同一
科学条件责任的对象，同时避免 LLM 自由生成 scientific edges。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from orion.task_evidence.evidence_applicability import CONDITION_FIELDS

from .models import CouplingEdge, CouplingLayer, HarnessArtifact, MultiplexHarnessGraph


@dataclass
class _FunctionAnchors:
    qualname: str
    lineno: int
    anchors: set[str] = field(default_factory=set)
    strong_anchors: set[str] = field(default_factory=set)


def _matches_anchor(text: str, anchor: str) -> bool:
    normalized = text.casefold().replace("-", "_").replace(" ", "_")
    target = re.escape(anchor.casefold())
    # 下划线视为结构分隔符而不是字母数字，因此既能命中
    # ``missing_cell_configuration``，又不会把 ``rate`` 误命中 ``generate``。
    return re.search(rf"(?<![a-z0-9]){target}(?![a-z0-9])", normalized) is not None


_STRONG_DECLARATION_KEYWORDS = {
    "pattern_code",
    "context_key",
    "context_field",
    "context_fields",
    "required_fields",
    "relevant_fields",
    "changed_fields",
    "supported_dimensions",
}


def _string_literals(node: ast.AST) -> list[str]:
    values: list[str] = []
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.append(item.value)
    return values


class _AnchorVisitor(ast.NodeVisitor):
    """扫描单个函数，不把嵌套函数的字段引用归到父函数。"""

    def __init__(self, anchors: set[str]) -> None:
        self.anchors = anchors
        self.matched: set[str] = set()
        self.strong_matched: set[str] = set()

    def _consume(self, text: str, *, strong: bool = False) -> None:
        for anchor in self.anchors:
            if _matches_anchor(text, anchor):
                self.matched.add(anchor)
                if strong:
                    self.strong_matched.add(anchor)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return None

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        self._consume(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        self._consume(node.attr)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, str):
            self._consume(node.value)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # 结构化 schema / gate / reflection 声明比普通字符串引用提供更强的责任证据。
        for keyword in node.keywords:
            if keyword.arg in _STRONG_DECLARATION_KEYWORDS:
                for text in _string_literals(keyword.value):
                    self._consume(text, strong=True)
        self.generic_visit(node)


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, anchors: set[str]) -> None:
        self.anchors = anchors
        self.scope: list[str] = []
        self.functions: list[_FunctionAnchors] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self.scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([*self.scope, node.name]) if self.scope else node.name
        visitor = _AnchorVisitor(self.anchors)
        # 函数名本身也是稳定结构信号，例如 ``_mine_missing_cell_configuration``。
        visitor._consume(node.name)
        for statement in node.body:
            visitor.visit(statement)
        if visitor.matched:
            self.functions.append(
                _FunctionAnchors(
                    qualname=qualname,
                    lineno=int(getattr(node, "lineno", 1)),
                    anchors=visitor.matched,
                    strong_anchors=visitor.strong_matched,
                )
            )
        self.scope.append(node.name)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(child)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)


def load_scientific_field_anchor_projection(
    graph: MultiplexHarnessGraph,
    *,
    source_files: Sequence[str | Path],
    project_root: str | Path | None = None,
    canonical_fields: Iterable[str] = CONDITION_FIELDS,
) -> None:
    """把规范 scientific context field 与显式引用它的源码对象连接起来。"""

    anchors = {str(item).strip() for item in canonical_fields if str(item).strip()}
    if not anchors:
        raise ValueError("canonical_fields must not be empty")
    root = Path(project_root).resolve() if project_root is not None else None

    for raw_path in source_files:
        path = Path(raw_path).resolve()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        collector = _FunctionCollector(anchors)
        collector.visit(tree)
        if root is not None:
            try:
                stable_path = path.relative_to(root).as_posix()
            except ValueError:
                stable_path = path.as_posix()
        else:
            stable_path = path.as_posix()

        for function in collector.functions:
            function_id = f"pyfunc:{stable_path}:{function.qualname}"
            graph.add_artifact(
                HarnessArtifact(
                    artifact_id=function_id,
                    kind="scientific_field_consumer",
                    label=function.qualname,
                    metadata={
                        "source_file": stable_path,
                        "source_location": f"L{function.lineno}",
                    },
                )
            )
            for field_name in sorted(function.anchors):
                field_id = f"scientific_context_field:{field_name}"
                graph.add_artifact(
                    HarnessArtifact(
                        artifact_id=field_id,
                        kind="scientific_context_field",
                        label=field_name,
                    )
                )
                is_strong = field_name in function.strong_anchors
                graph.add_edge(
                    CouplingEdge(
                        # impact direction: a scientific field/obligation change can invalidate
                        # code/tests that declare or reference that field. Keeping this direction
                        # prevents a multi-field consumer from becoming a bridge to unrelated fields.
                        source=field_id,
                        target=function_id,
                        layer=CouplingLayer.SCIENTIFIC,
                        weight=1.2 if is_strong else 0.25,
                        directed=True,
                        relation=(
                            "declares_scientific_responsibility"
                            if is_strong
                            else "references_scientific_field"
                        ),
                        provenance=f"ast-schema-anchor:{stable_path}:L{function.lineno}",
                    )
                )
