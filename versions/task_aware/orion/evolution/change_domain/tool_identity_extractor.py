"""Deterministic Agent-tool identity coupling for the graph-native Harness.

Ordinary Python call graphs miss common LangChain/LangGraph Harness patterns:

* a function is wrapped by ``@tool`` and becomes a Tool object;
* tests/runtime code call ``tool.func(...)`` / ``tool.invoke(...)``;
* ToolNode, ToolMessage and trace dictionaries refer to a tool by stable string name.

This extractor makes that Harness-specific identity explicit without asking an LLM to infer
semantics.  Direct object invocation is a strong edge; exact string-name references are weaker
behavior/regression evidence.  The wrapped source symbol is resolved against Graphify by exact
file + function label.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .models import CouplingEdge, CouplingLayer, HarnessArtifact, MultiplexHarnessGraph


@dataclass(frozen=True)
class ToolIdentityProjection:
    tool_ids: tuple[str, ...]
    definition_links: int
    direct_consumer_links: int
    name_reference_links: int


@dataclass
class _FunctionToolUse:
    qualname: str
    lineno: int
    direct: set[str] = field(default_factory=set)
    referenced: set[str] = field(default_factory=set)


def _decorator_is_tool(node: ast.expr) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id == "tool"
    if isinstance(target, ast.Attribute):
        return target.attr == "tool"
    return False


def _discover_tool_definitions(source_files: Sequence[Path]) -> dict[str, tuple[str, int]]:
    definitions: dict[str, tuple[str, int]] = {}
    for path in source_files:
        normalized_parts = {part.casefold() for part in path.parts}
        # Test fixtures often define local @tool doubles with the same stable name as production
        # tools. They are consumers/fixtures, not competing canonical ToolIdentity definitions.
        if "tests" in normalized_parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(_decorator_is_tool(item) for item in node.decorator_list):
                continue
            if node.name in definitions:
                # Stable tool identity must not silently alias two definitions.
                previous = definitions[node.name]
                if previous != (path.as_posix(), int(node.lineno)):
                    raise ValueError(f"duplicate @tool identity: {node.name}: {previous} vs {path}")
            definitions[node.name] = (path.as_posix(), int(node.lineno))
    return definitions


def _exact_graphify_symbol(
    graph: MultiplexHarnessGraph,
    *,
    source_file: str,
    tool_name: str,
) -> str | None:
    source_file = source_file.replace("\\", "/")
    matches = [
        artifact_id
        for artifact_id, artifact in graph.artifacts.items()
        if artifact_id.startswith("graphify:")
        and str(artifact.metadata.get("source_file") or "").replace("\\", "/") == source_file
        and artifact.label == f"{tool_name}()"
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"ambiguous Graphify tool symbol: {source_file}:{tool_name}:{matches}")
    return matches[0]


class _ToolUseVisitor(ast.NodeVisitor):
    def __init__(self, tool_names: set[str]) -> None:
        self.tool_names = tool_names
        self.direct: set[str] = set()
        self.referenced: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return None

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.tool_names:
            self.direct.add(func.id)
        elif isinstance(func, ast.Attribute):
            base = func.value
            if isinstance(base, ast.Name) and base.id in self.tool_names and func.attr in {
                "func",
                "invoke",
                "ainvoke",
                "run",
                "arun",
            }:
                self.direct.add(base.id)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, str) and node.value in self.tool_names:
            self.referenced.add(node.value)


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, tool_names: set[str]) -> None:
        self.tool_names = tool_names
        self.scope: list[str] = []
        self.uses: list[_FunctionToolUse] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self.scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([*self.scope, node.name]) if self.scope else node.name
        visitor = _ToolUseVisitor(self.tool_names)
        for statement in node.body:
            visitor.visit(statement)
        # A direct call is already stronger than an exact string occurrence in the same function.
        referenced = visitor.referenced - visitor.direct
        if visitor.direct or referenced:
            self.uses.append(
                _FunctionToolUse(
                    qualname=qualname,
                    lineno=int(node.lineno),
                    direct=set(visitor.direct),
                    referenced=set(referenced),
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


def load_tool_identity_projection(
    graph: MultiplexHarnessGraph,
    *,
    source_files: Sequence[str | Path],
    project_root: str | Path,
) -> ToolIdentityProjection:
    """Project ``@tool`` definitions and their exact runtime/test consumers."""

    root = Path(project_root).resolve()
    paths = tuple(Path(item).resolve() for item in source_files)
    definitions_abs = _discover_tool_definitions(paths)
    definitions: dict[str, tuple[str, int]] = {}
    for tool_name, (raw_path, line) in definitions_abs.items():
        path = Path(raw_path).resolve()
        try:
            stable_path = path.relative_to(root).as_posix()
        except ValueError:
            continue
        definitions[tool_name] = (stable_path, line)

    tool_names = set(definitions)
    definition_links = 0
    direct_links = 0
    reference_links = 0

    for tool_name, (source_file, line) in sorted(definitions.items()):
        tool_id = f"tool_identity:{tool_name}"
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=tool_id,
                kind="agent_tool_identity",
                label=tool_name,
                metadata={"definition_file": source_file, "definition_location": f"L{line}"},
            )
        )
        graphify_symbol = _exact_graphify_symbol(
            graph,
            source_file=source_file,
            tool_name=tool_name,
        )
        if graphify_symbol is not None:
            before = len(graph.edges)
            graph.add_edge(
                CouplingEdge(
                    source=tool_id,
                    target=graphify_symbol,
                    layer=CouplingLayer.SERVICE,
                    weight=1.5,
                    directed=True,
                    relation="wraps_tool_symbol",
                    provenance=f"ast:@tool:{source_file}:L{line}",
                )
            )
            definition_links += int(len(graph.edges) > before)

    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeError):
            continue
        try:
            stable_path = path.relative_to(root).as_posix()
        except ValueError:
            continue
        collector = _FunctionCollector(tool_names)
        collector.visit(tree)
        is_test = stable_path.startswith("tests/") or "/tests/" in f"/{stable_path}"
        layer = CouplingLayer.REGRESSION if is_test else CouplingLayer.BEHAVIOR
        for use in collector.uses:
            consumer_id = f"pyfunc:{stable_path}:{use.qualname}"
            graph.add_artifact(
                HarnessArtifact(
                    artifact_id=consumer_id,
                    kind="tool_identity_consumer",
                    label=use.qualname,
                    metadata={"source_file": stable_path, "source_location": f"L{use.lineno}"},
                )
            )
            for tool_name in sorted(use.direct):
                before = len(graph.edges)
                graph.add_edge(
                    CouplingEdge(
                        source=f"tool_identity:{tool_name}",
                        target=consumer_id,
                        layer=layer,
                        weight=1.25,
                        directed=True,
                        relation="direct_tool_object_use",
                        provenance=f"ast:tool-object-use:{stable_path}:L{use.lineno}",
                    )
                )
                direct_links += int(len(graph.edges) > before)
            for tool_name in sorted(use.referenced):
                before = len(graph.edges)
                graph.add_edge(
                    CouplingEdge(
                        source=f"tool_identity:{tool_name}",
                        target=consumer_id,
                        layer=layer,
                        weight=0.35,
                        directed=True,
                        relation="references_tool_name",
                        provenance=f"ast:exact-tool-name:{stable_path}:L{use.lineno}",
                    )
                )
                reference_links += int(len(graph.edges) > before)

    return ToolIdentityProjection(
        tool_ids=tuple(sorted(f"tool_identity:{name}" for name in tool_names)),
        definition_links=definition_links,
        direct_consumer_links=direct_links,
        name_reference_links=reference_links,
    )
