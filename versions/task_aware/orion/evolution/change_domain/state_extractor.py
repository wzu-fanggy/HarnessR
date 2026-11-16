"""从 Python 源码确定性提取 LangGraph-style state read/write coupling。

V0 只识别可审计的结构模式：

- ``state["field"]`` / ``state.get("field")`` -> read;
- ``state["field"] = ...`` -> write;
- 接收 state 参数的函数 ``return {"field": ...}`` -> state update。

不使用 LLM 猜测变量语义，不把任意字符串相似性当作状态关系。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .models import CouplingEdge, CouplingLayer, HarnessArtifact, MultiplexHarnessGraph


@dataclass
class _FunctionAccess:
    qualname: str
    lineno: int
    reads: set[str] = field(default_factory=set)
    writes: set[str] = field(default_factory=set)


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value:
        return node.value
    return None


def _subscript_key(node: ast.Subscript) -> str | None:
    return _literal_string(node.slice)


def _argument_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    return {arg.arg for arg in args}


class _AccessVisitor(ast.NodeVisitor):
    """扫描单个函数体，但不跨入嵌套函数定义。"""

    def __init__(self, *, state_names: set[str], infer_return_dict_writes: bool) -> None:
        self.state_names = state_names
        self.infer_return_dict_writes = infer_return_dict_writes
        self.reads: set[str] = set()
        self.writes: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return None

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        if isinstance(node.value, ast.Name) and node.value.id in self.state_names:
            key = _subscript_key(node)
            if key:
                if isinstance(node.ctx, ast.Store):
                    self.writes.add(key)
                else:
                    self.reads.add(key)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in self.state_names
            and func.attr == "get"
            and node.args
        ):
            key = _literal_string(node.args[0])
            if key:
                self.reads.add(key)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
        if self.infer_return_dict_writes and isinstance(node.value, ast.Dict):
            for key_node in node.value.keys:
                key = _literal_string(key_node)
                if key:
                    self.writes.add(key)
        self.generic_visit(node)


class _FunctionCollector(ast.NodeVisitor):
    def __init__(
        self,
        *,
        accepted_state_parameter_names: set[str],
        infer_return_dict_writes: bool,
    ) -> None:
        self.accepted_state_parameter_names = accepted_state_parameter_names
        self.infer_return_dict_writes = infer_return_dict_writes
        self.scope: list[str] = []
        self.functions: list[_FunctionAccess] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self.scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parameter_names = _argument_names(node)
        state_names = parameter_names & self.accepted_state_parameter_names
        qualname = ".".join([*self.scope, node.name]) if self.scope else node.name
        if state_names:
            visitor = _AccessVisitor(
                state_names=state_names,
                infer_return_dict_writes=self.infer_return_dict_writes,
            )
            for statement in node.body:
                visitor.visit(statement)
            if visitor.reads or visitor.writes:
                self.functions.append(
                    _FunctionAccess(
                        qualname=qualname,
                        lineno=int(getattr(node, "lineno", 1)),
                        reads=visitor.reads,
                        writes=visitor.writes,
                    )
                )
        # 嵌套函数仍单独收集，但不把其 access 算到父函数。
        self.scope.append(node.name)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(child)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)


def load_python_state_access_projection(
    graph: MultiplexHarnessGraph,
    *,
    source_files: Sequence[str | Path],
    project_root: str | Path | None = None,
    state_parameter_names: Iterable[str] = ("state",),
    infer_return_dict_writes: bool = True,
) -> None:
    """把显式 state read/write 投影为 ``STATE`` coupling edges。

    ``project_root`` 只用于产生稳定的相对 artifact id；不改变文件访问范围。
    无法解析的源码会抛出 ``SyntaxError``，不会静默降级成空图。
    """

    root = Path(project_root).resolve() if project_root is not None else None
    accepted_names = {name for name in state_parameter_names if name}
    if not accepted_names:
        raise ValueError("state_parameter_names must not be empty")

    for raw_path in source_files:
        path = Path(raw_path).resolve()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        collector = _FunctionCollector(
            accepted_state_parameter_names=accepted_names,
            infer_return_dict_writes=infer_return_dict_writes,
        )
        collector.visit(tree)
        if root is not None:
            try:
                stable_path = path.relative_to(root).as_posix()
            except ValueError:
                stable_path = path.as_posix()
        else:
            stable_path = path.as_posix()

        for access in collector.functions:
            function_id = f"pyfunc:{stable_path}:{access.qualname}"
            graph.add_artifact(
                HarnessArtifact(
                    artifact_id=function_id,
                    kind="stateful_function",
                    label=access.qualname,
                    metadata={
                        "source_file": stable_path,
                        "source_location": f"L{access.lineno}",
                    },
                )
            )
            for field_name in sorted(access.reads | access.writes):
                field_id = f"state_field:{field_name}"
                graph.add_artifact(
                    HarnessArtifact(
                        artifact_id=field_id,
                        kind="state_field",
                        label=field_name,
                    )
                )
            for field_name in sorted(access.reads):
                graph.add_edge(
                    CouplingEdge(
                        source=function_id,
                        target=f"state_field:{field_name}",
                        layer=CouplingLayer.STATE,
                        weight=1.0,
                        directed=True,
                        relation="reads_state",
                        provenance=f"ast:{stable_path}:L{access.lineno}",
                    )
                )
            for field_name in sorted(access.writes):
                graph.add_edge(
                    CouplingEdge(
                        source=function_id,
                        target=f"state_field:{field_name}",
                        layer=CouplingLayer.STATE,
                        weight=1.0,
                        directed=True,
                        relation="writes_state",
                        provenance=f"ast:{stable_path}:L{access.lineno}",
                    )
                )
