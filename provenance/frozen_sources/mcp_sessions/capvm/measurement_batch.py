"""Reuse a remote service snapshot within one synchronous cost-measurement pass.

Each pass is fresh; releasing or establishing resources never shares a previous
pass's snapshot. Nested passes reuse the outer snapshot to prevent redundant
requests from diagnostic recording after a budget check.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_snapshots: ContextVar[dict[int, dict[str, dict[str, Any]]] | None] = ContextVar(
    "capvm_remote_resource_snapshots",
    default=None,
)


@contextmanager
def measurement_batch() -> Iterator[None]:
    """Share one remote resource list per client for the duration of this pass."""

    if _snapshots.get() is not None:
        yield
        return
    token = _snapshots.set({})
    try:
        yield
    finally:
        _snapshots.reset(token)


def runtime_snapshots(client: Any) -> dict[str, dict[str, Any]]:
    """Return a fresh list outside a pass, or the current pass's list."""

    cache = _snapshots.get()
    if cache is None:
        return client.runtime_resources()
    key = id(client)
    if key not in cache:
        cache[key] = client.runtime_resources()
    return cache[key]
