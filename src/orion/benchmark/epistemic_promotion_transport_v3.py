"""Uniform pre-output transport retry policy for a future recovery preregistration.

This helper is development-only until a new exact snapshot is frozen.  It permits
at most one retry of the *same non-streaming HTTP model request* when the call
raises a connection/timeout/incomplete-transport exception before returning a
response object.  It never retries JSON/schema failures, semantic failures,
completed tool calls, or any call after raw output has been observed by the
caller.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

import httpx

try:
    from openai import APIConnectionError, APITimeoutError
except Exception:  # pragma: no cover - environment compatibility guard
    APIConnectionError = ()  # type: ignore[assignment,misc]
    APITimeoutError = ()  # type: ignore[assignment,misc]


T = TypeVar("T")

_RETRYABLE_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "ConnectionResetError",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "TimeoutError",
    "WriteError",
    "WriteTimeout",
}


def is_retryable_preoutput_transport_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if APIConnectionError and isinstance(current, APIConnectionError):
            return True
        if APITimeoutError and isinstance(current, APITimeoutError):
            return True
        if isinstance(current, (httpx.TransportError, asyncio.TimeoutError)):
            return True
        if type(current).__name__ in _RETRYABLE_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


def invoke_with_preoutput_transport_retry(
    invoke_fn: Callable[[], T],
    *,
    max_transport_retries: int = 1,
) -> tuple[T, int]:
    """Invoke a non-streaming request under a narrow transport-only retry policy.

    The caller must persist raw model output only after this function returns.
    Consequently, a retry can occur only after an exception and before any
    caller-observed response/raw text exists.
    """

    if max_transport_retries < 0 or max_transport_retries > 1:
        raise ValueError("formal recovery transport retries must be 0 or 1")

    for attempt in range(max_transport_retries + 1):
        try:
            return invoke_fn(), attempt
        except BaseException as exc:
            if attempt >= max_transport_retries:
                raise
            if not is_retryable_preoutput_transport_error(exc):
                raise
    raise AssertionError("unreachable")


__all__ = [
    "is_retryable_preoutput_transport_error",
    "invoke_with_preoutput_transport_retry",
]
