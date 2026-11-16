"""OpenAI-compatible endpoint configuration and zero-model preflight helpers."""

from __future__ import annotations

import ipaddress
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, TextIO
from urllib.parse import urlsplit, urlunsplit

import httpx


_TRUE_VALUES = {"1", "true", "yes", "on"}
_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


def _truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in _TRUE_VALUES)


def normalize_openai_base_url(value: str, *, label: str = "base_url") -> str:
    """Return the API base URL expected by OpenAI/LangChain clients.

    Provider documentation sometimes supplies a complete Chat Completions URL.
    LangChain expects the API base instead and appends ``/chat/completions``.
    Accept either form while rejecting ambiguous or credential-bearing URLs.
    """

    raw = (value or "").strip().rstrip("/")
    if not raw:
        raise ValueError(f"{label} is required")
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{label} must not contain a query string or fragment")

    path = parsed.path.rstrip("/")
    if path.lower().endswith(_CHAT_COMPLETIONS_SUFFIX):
        path = path[: -len(_CHAT_COMPLETIONS_SUFFIX)].rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", "")).rstrip("/")


def chat_completions_url(base_url: str) -> str:
    """Build the complete Chat Completions URL from either accepted input form."""

    return normalize_openai_base_url(base_url) + _CHAT_COMPLETIONS_SUFFIX


def _is_loopback(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _positive_float(value: str, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive number") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive number")
    return parsed


def _nonnegative_int(value: str, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed


@dataclass(frozen=True)
class GroundingEndpointConfig:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    request_timeout: float
    max_retries: int
    insecure_http: bool

    @property
    def chat_completions_url(self) -> str:
        return chat_completions_url(self.base_url)


def resolve_grounding_endpoint_config(
    environ: Mapping[str, str] | None = None,
) -> GroundingEndpointConfig | None:
    """Resolve and validate Orion's independent grounding-judge configuration.

    ``ORION_GROUNDING_*`` has priority over the legacy ``RAGAS_*`` variables.
    A remote plaintext HTTP route is refused unless the operator explicitly sets
    ``ORION_ALLOW_INSECURE_GROUNDING_HTTP=1``. Local loopback development URLs
    remain available without that opt-in.
    """

    env = os.environ if environ is None else environ
    base_value = (env.get("ORION_GROUNDING_BASE_URL") or env.get("RAGAS_BASE_URL") or "").strip()
    api_key = (env.get("ORION_GROUNDING_API_KEY") or env.get("RAGAS_API_KEY") or "").strip()
    if not base_value and not api_key:
        return None
    if not base_value or not api_key:
        raise RuntimeError(
            "Independent grounding judge is partially configured; both base URL and API key are required"
        )

    base_url = normalize_openai_base_url(base_value, label="grounding base URL")
    parsed = urlsplit(base_url)
    insecure_http = parsed.scheme == "http" and not _is_loopback(parsed.hostname)
    if insecure_http and not _truthy(env.get("ORION_ALLOW_INSECURE_GROUNDING_HTTP")):
        raise RuntimeError(
            "Remote grounding judge uses plaintext HTTP; set "
            "ORION_ALLOW_INSECURE_GROUNDING_HTTP=1 only after accepting credential exposure risk"
        )

    model = (env.get("ORION_GROUNDING_MODEL") or env.get("RAGAS_MODEL") or "gpt-5.6-terra").strip()
    if not model:
        raise ValueError("grounding model is required")
    return GroundingEndpointConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        request_timeout=_positive_float(
            env.get("ORION_GROUNDING_REQUEST_TIMEOUT", "120"),
            label="ORION_GROUNDING_REQUEST_TIMEOUT",
        ),
        max_retries=_nonnegative_int(
            env.get("ORION_GROUNDING_MAX_RETRIES", "0"),
            label="ORION_GROUNDING_MAX_RETRIES",
        ),
        insecure_http=insecure_http,
    )


def preflight_chat_completions_endpoint(
    config: GroundingEndpointConfig,
    *,
    timeout: float = 20.0,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Check endpoint routing and authentication without requesting a model run.

    The deliberately invalid empty JSON object contains no model or messages.
    A normal OpenAI-compatible server should reject it with JSON 400/422 before
    dispatching inference. Authentication, path, and transport failures remain
    distinct so callers cannot mistake reachability for a successful model test.
    """

    started = time.perf_counter()
    request_payload: dict[str, Any] = {}
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            http2=False,
            transport=transport,
        ) as client:
            response = client.post(
                config.chat_completions_url,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
            )
        try:
            body = response.json()
        except ValueError:
            body = None
        response_keys = sorted(body) if isinstance(body, dict) else []
        if response.status_code in {400, 422} and isinstance(body, dict):
            status = "PASS"
            failure_class = None
        elif response.status_code in {401, 403}:
            status = "FAIL"
            failure_class = "authentication"
        elif response.status_code == 404:
            status = "FAIL"
            failure_class = "endpoint_path"
        else:
            status = "FAIL"
            failure_class = "unexpected_http_response"
        return {
            "schema_version": "orion-grounding-endpoint-preflight-v1",
            "status": status,
            "zero_model_request": True,
            "request_payload": request_payload,
            "base_url": config.base_url,
            "chat_completions_url": config.chat_completions_url,
            "insecure_http": config.insecure_http,
            "http_status": response.status_code,
            "response_keys": response_keys,
            "failure_class": failure_class,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except Exception as exc:
        return {
            "schema_version": "orion-grounding-endpoint-preflight-v1",
            "status": "ERROR",
            "zero_model_request": True,
            "request_payload": request_payload,
            "base_url": config.base_url,
            "chat_completions_url": config.chat_completions_url,
            "insecure_http": config.insecure_http,
            "http_status": None,
            "response_keys": [],
            "failure_class": "transport",
            "error_type": type(exc).__name__,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def emit_json_stdout(
    value: Any,
    *,
    stream: TextIO | None = None,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> None:
    """Emit parseable ASCII JSON even when Windows stdout is limited to GBK."""

    target = sys.stdout if stream is None else stream
    target.write(json.dumps(value, ensure_ascii=True, indent=indent, sort_keys=sort_keys) + "\n")
    target.flush()
