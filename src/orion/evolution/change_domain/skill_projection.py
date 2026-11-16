"""Machine-readable Skill workflow projection for the graph-native Scientific Harness.

ORION skills already carry YAML frontmatter with workflow steps and stable tool names.  This module
uses that existing contract directly instead of inventing a second task-role table for experiments.
A skill is a logical workflow object; its steps point to stable ToolIdentity objects.  Required and
optional steps remain explicit so runtime policy/scientific state can choose the smallest admissible
profile without changing the skill definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import CouplingEdge, CouplingLayer, HarnessArtifact, MultiplexHarnessGraph


@dataclass(frozen=True)
class SkillWorkflowStep:
    step_id: str
    tool_name: str
    required: bool
    description: str
    node_id: str
    tool_node_id: str


@dataclass(frozen=True)
class SkillWorkflowProjection:
    skill_name: str
    skill_node_id: str
    steps: tuple[SkillWorkflowStep, ...]
    frontmatter: Mapping[str, Any]

    @property
    def required_steps(self) -> tuple[SkillWorkflowStep, ...]:
        return tuple(step for step in self.steps if step.required)

    @property
    def optional_steps(self) -> tuple[SkillWorkflowStep, ...]:
        return tuple(step for step in self.steps if not step.required)


def read_skill_frontmatter(path: str | Path) -> dict[str, Any]:
    """Read one YAML frontmatter block from a Skill markdown file."""

    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"skill file has no YAML frontmatter: {path}")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"skill frontmatter is not closed: {path}")
    payload = yaml.safe_load(parts[1]) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"skill frontmatter must be a mapping: {path}")
    return payload


def add_skill_workflow_projection(
    graph: MultiplexHarnessGraph,
    *,
    skill_path: str | Path,
) -> SkillWorkflowProjection:
    """Project a skill's machine-readable workflow into the multiplex graph."""

    frontmatter = read_skill_frontmatter(skill_path)
    skill_name = str(frontmatter.get("name") or "").strip()
    if not skill_name:
        raise ValueError("skill frontmatter requires name")
    workflow = frontmatter.get("workflow") or {}
    if not isinstance(workflow, dict):
        raise ValueError(f"skill workflow must be a mapping: {skill_name}")
    raw_steps = workflow.get("steps") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"skill workflow must contain steps: {skill_name}")

    skill_node = f"skill:{skill_name}"
    graph.add_artifact(
        HarnessArtifact(
            artifact_id=skill_node,
            kind="skill_contract",
            label=skill_name,
            metadata={
                "description": str(frontmatter.get("description") or ""),
                "source_file": Path(skill_path).as_posix(),
            },
        )
    )

    steps: list[SkillWorkflowStep] = []
    previous_step_node: str | None = None
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise ValueError(f"workflow step must be a mapping: {skill_name}:{index}")
        step_id = str(raw.get("id") or "").strip()
        tool_name = str(raw.get("tool") or "").strip()
        if not step_id or not tool_name:
            raise ValueError(f"workflow step requires id/tool: {skill_name}:{index}")
        required = bool(raw.get("required", False))
        description = str(raw.get("description") or "").strip()
        step_node = f"skill_step:{skill_name}:{step_id}"
        tool_node = f"tool_identity:{tool_name}"
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=step_node,
                kind="skill_workflow_step",
                label=step_id,
                metadata={
                    "skill": skill_name,
                    "tool": tool_name,
                    "required": "true" if required else "false",
                    "description": description,
                    "order": str(index),
                },
            )
        )
        graph.add_artifact(
            HarnessArtifact(
                artifact_id=tool_node,
                kind="agent_tool_identity",
                label=tool_name,
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=skill_node,
                target=step_node,
                layer=CouplingLayer.BEHAVIOR,
                weight=1.0,
                directed=True,
                relation="contains_workflow_step",
                provenance="skill-frontmatter:workflow.steps",
            )
        )
        graph.add_edge(
            CouplingEdge(
                source=step_node,
                target=tool_node,
                layer=CouplingLayer.SERVICE,
                weight=1.25 if required else 0.7,
                directed=True,
                relation="uses_tool",
                provenance="skill-frontmatter:workflow.steps.tool",
            )
        )
        if previous_step_node is not None:
            graph.add_edge(
                CouplingEdge(
                    source=previous_step_node,
                    target=step_node,
                    layer=CouplingLayer.CONTROL,
                    weight=0.8,
                    directed=True,
                    relation="next_skill_step",
                    provenance="skill-frontmatter:workflow.order",
                )
            )
        previous_step_node = step_node
        steps.append(
            SkillWorkflowStep(
                step_id=step_id,
                tool_name=tool_name,
                required=required,
                description=description,
                node_id=step_node,
                tool_node_id=tool_node,
            )
        )

    return SkillWorkflowProjection(
        skill_name=skill_name,
        skill_node_id=skill_node,
        steps=tuple(steps),
        frontmatter=frontmatter,
    )


def active_skill_tool_nodes(
    projection: SkillWorkflowProjection,
    *,
    optional_step_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return required tools plus explicitly activated optional workflow steps."""

    optional = set(optional_step_ids)
    unknown = optional - {step.step_id for step in projection.optional_steps}
    if unknown:
        raise ValueError(f"unknown optional skill steps: {sorted(unknown)}")
    return tuple(
        step.tool_node_id
        for step in projection.steps
        if step.required or step.step_id in optional
    )
