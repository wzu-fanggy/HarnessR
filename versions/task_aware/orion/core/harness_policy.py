"""Stable Harness seams for domain-specific policy capabilities.

The main LangGraph topology owns the seam, not the domain policy itself. A
post-tool policy can inspect the state produced by a ToolNode and return a
LangGraph ``Command`` that decides the next generic Harness transition. This
keeps domain policy replaceable without making it a permanent graph node.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from langgraph.types import Command


class PostToolPolicy(Protocol):
    """Domain-policy contract attached behind the stable post-tool seam."""

    def __call__(self, state: Mapping[str, Any]) -> Command: ...


def _merge_state_with_tool_update(
    state: Mapping[str, Any],
    tool_update: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the state view seen by a post-tool policy in the same turn.

    LangGraph normally applies the ToolNode update before entering the next
    node. Once the domain decision is no longer a graph node, the policy still
    needs the equivalent merged view. ``messages`` is the only reducer-backed
    field that needs explicit concatenation here; other fields are ordinary
    last-write updates.
    """

    merged = dict(state)
    for key, value in tool_update.items():
        if key == "messages":
            merged[key] = [
                *list(state.get("messages") or []),
                *list(value or []),
            ]
        else:
            merged[key] = value
    return merged


def _combine_updates(
    tool_update: Mapping[str, Any],
    policy_update: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Combine ToolNode and policy updates into one graph-state update."""

    combined = dict(tool_update)
    if not policy_update:
        return combined
    for key, value in policy_update.items():
        if key == "messages":
            combined[key] = [
                *list(combined.get("messages") or []),
                *list(value or []),
            ]
        else:
            combined[key] = value
    return combined


def apply_post_tool_policy(
    state: Mapping[str, Any],
    tool_update: Mapping[str, Any],
    policy: PostToolPolicy | Callable[[Mapping[str, Any]], Command] | None,
    *,
    default_goto: str = "llm",
) -> Command:
    """Apply a replaceable domain policy behind the stable ToolNode seam.

    The Harness graph owns only ``tools``. The policy decides what happens
    after the tool results are available, but it is not materialized as a
    permanent LangGraph node. A different policy can therefore be selected
    without changing graph topology.
    """

    if policy is None:
        return Command(update=dict(tool_update), goto=default_goto)

    merged_state = _merge_state_with_tool_update(state, tool_update)
    decision = policy(merged_state)
    raw_policy_update = decision.update
    if raw_policy_update is None:
        policy_update: Mapping[str, Any] | None = None
    elif isinstance(raw_policy_update, Mapping):
        policy_update = raw_policy_update
    else:
        raise TypeError("post-tool policy update must be a mapping or None")

    return Command(
        graph=decision.graph,
        update=_combine_updates(tool_update, policy_update),
        resume=decision.resume,
        goto=decision.goto,
    )
