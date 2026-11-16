"""Fresh v2 provider-health firewall and future-probe failure classification.

目标：provider/runtime 不健康时必须在 formal lock 之前停住；已经进入 future probe 后，
零 answer/零 tool-call 的 provider failure 也不得被当成科学上的 ``do_not_pre_activate``。
本模块本身不调用网络，真实 health invocation 由后续 v2 runner 注入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field


HEALTH_TOKEN = "ORION_RUNTIME_HEALTH_OK"


class FreshV2RuntimeHealthReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["orion_self_evolution_fresh_v2_runtime_health_v1"] = (
        "orion_self_evolution_fresh_v2_runtime_health_v1"
    )
    passed: bool
    response_observed: bool
    expected_token: str = HEALTH_TOKEN
    observed_text_sha256: str | None = None
    sanitized_error_class: str | None = None
    sanitized_http_status: int | None = None
    sanitized_error_code: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


class FreshV2FutureProbeClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["orion_self_evolution_fresh_v2_future_probe_classification_v1"] = (
        "orion_self_evolution_fresh_v2_future_probe_classification_v1"
    )
    terminal_class: Literal[
        "behavior_observable",
        "preoutput_transport_exhausted",
        "orchestration_blocked",
    ]
    behavior_observable: bool
    provider_failure: bool
    answer_observed: bool
    tool_call_observed: bool
    sanitized_error_class: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class FreshV2HealthGuardResult:
    health: FreshV2RuntimeHealthReceipt
    lock_acquired: bool
    lock_value: Any = None


class FreshV2RuntimeUnhealthy(RuntimeError):
    """pre-lock runtime health gate 失败。"""


def runtime_health_messages() -> tuple[str, str]:
    """返回完全 task-independent 的 health prompt 文本。"""

    system = (
        "This is a provider/runtime availability probe only. It carries no scientific task, "
        "candidate, family, source, boundary, or desired experimental outcome. "
        f"Return exactly {HEALTH_TOKEN}."
    )
    user = f"Return exactly {HEALTH_TOKEN}."
    return system, user


def _safe_scalar(value: Any) -> str | None:
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return text[:120] if text else None
    return None


def sanitize_provider_exception(exc: BaseException) -> dict[str, Any]:
    """只保留错误类别、HTTP status 与短 error code；不保存 body/header/credential。"""

    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        status = response_status if isinstance(response_status, int) else None

    code = _safe_scalar(getattr(exc, "code", None))
    if code is None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            code = _safe_scalar(body.get("code"))
            if code is None and isinstance(body.get("error"), dict):
                code = _safe_scalar(body["error"].get("code"))

    return {
        "error_class": type(exc).__name__,
        "http_status": status,
        "error_code": code,
    }


def run_runtime_health_probe(
    invoke: Callable[[], Any],
    *,
    extract_text: Callable[[Any], str] | None = None,
) -> FreshV2RuntimeHealthReceipt:
    """执行一次 health invocation；无重试、无 scientific payload。"""

    import hashlib

    if extract_text is None:
        extract_text = lambda response: str(getattr(response, "content", response) or "")
    try:
        response = invoke()
    except Exception as exc:
        safe = sanitize_provider_exception(exc)
        return FreshV2RuntimeHealthReceipt(
            passed=False,
            response_observed=False,
            sanitized_error_class=safe["error_class"],
            sanitized_http_status=safe["http_status"],
            sanitized_error_code=safe["error_code"],
            reason_codes=["v2_runtime_health_invoke_failed"],
        )

    text = extract_text(response).strip()
    observed_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    passed = text == HEALTH_TOKEN
    return FreshV2RuntimeHealthReceipt(
        passed=passed,
        response_observed=True,
        observed_text_sha256=observed_sha,
        reason_codes=(
            ["v2_runtime_health_passed"]
            if passed
            else ["v2_runtime_health_token_mismatch"]
        ),
    )


def acquire_only_after_health(
    health: FreshV2RuntimeHealthReceipt,
    acquire_lock: Callable[[], Any],
) -> FreshV2HealthGuardResult:
    """health PASS 前绝不调用 formal lock acquire function。"""

    if not health.passed:
        raise FreshV2RuntimeUnhealthy("runtime health failed before formal lock acquisition")
    value = acquire_lock()
    return FreshV2HealthGuardResult(health=health, lock_acquired=True, lock_value=value)


_PROVIDER_FAILURE_TYPES = {
    "APIStatusError",
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "BadGatewayError",
}


def _probe_error_type(probe: dict[str, Any]) -> str | None:
    direct = str(probe.get("error_type") or "").strip()
    if direct:
        return direct
    for reason in probe.get("failure_reasons") or []:
        text = str(reason)
        if text.startswith("agent_error:"):
            return text.split(":", 1)[1].strip() or None
    return None


def classify_v2_future_probe(probe: dict[str, Any]) -> FreshV2FutureProbeClassification:
    """在行为 scorer 之前把 provider failure 与科学行为分开。"""

    answer_observed = bool(str(probe.get("answer") or "").strip())
    tool_calls = list(probe.get("tool_calls") or [])
    tool_call_observed = bool(tool_calls)
    error_type = _probe_error_type(probe)
    ok = bool(probe.get("ok", True))

    if not ok and error_type in _PROVIDER_FAILURE_TYPES and not answer_observed and not tool_call_observed:
        return FreshV2FutureProbeClassification(
            terminal_class="preoutput_transport_exhausted",
            behavior_observable=False,
            provider_failure=True,
            answer_observed=False,
            tool_call_observed=False,
            sanitized_error_class=error_type,
            reason_codes=["v2_provider_failure_before_observable_behavior"],
        )

    if not ok and not answer_observed and not tool_call_observed:
        return FreshV2FutureProbeClassification(
            terminal_class="orchestration_blocked",
            behavior_observable=False,
            provider_failure=False,
            answer_observed=False,
            tool_call_observed=False,
            sanitized_error_class=error_type,
            reason_codes=["v2_no_observable_behavior_nonprovider_failure"],
        )

    return FreshV2FutureProbeClassification(
        terminal_class="behavior_observable",
        behavior_observable=True,
        provider_failure=bool(not ok and error_type in _PROVIDER_FAILURE_TYPES),
        answer_observed=answer_observed,
        tool_call_observed=tool_call_observed,
        sanitized_error_class=error_type,
        reason_codes=["v2_behavior_observable_for_scoring"],
    )


__all__ = [
    "HEALTH_TOKEN",
    "FreshV2FutureProbeClassification",
    "FreshV2HealthGuardResult",
    "FreshV2RuntimeHealthReceipt",
    "FreshV2RuntimeUnhealthy",
    "acquire_only_after_health",
    "classify_v2_future_probe",
    "run_runtime_health_probe",
    "runtime_health_messages",
    "sanitize_provider_exception",
]
