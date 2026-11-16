"""Active unload/reload canary for the production conductivity tool resource.

This is a physical lifecycle validation, not a policy comparison. The exact same
production tool request is executed twice through ToolNode. Between calls the
managed RF process is explicitly demoted to CATALOG. The canary verifies that the
second call starts a new worker process and returns the same scientific prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from orion.capvm import ResidencyLevel
from orion.core.default_resource_runtime import build_default_tool_resource_runtime
from tools.electrolyte_design.conductivity_tool import (
    CONDUCTIVITY_PROCESS_RESOURCE_ID,
    predict_formulation_conductivity,
)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


REQUEST = {
    "candidates": [
        {
            "candidate_id": "canary-candidate",
            "salt": "NaPF6",
            "solvents": ["EC", "EMC", "PC"],
            "additives": [],
        }
    ],
    "temperature_c": 0.0,
    "composition_inputs": {
        "canary-candidate": {
            "x_NaSalt": 0.12438287472883666,
            "x_EC": 0.6545108436328688,
            "x_PC": 0.5161749545925916,
        }
    },
}


def _state(call_id: str) -> dict[str, Any]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "predict_formulation_conductivity",
                        "args": {"request": REQUEST},
                        "id": call_id,
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }


def _tool_payload(result: dict[str, Any]) -> Any:
    messages = list(result.get("messages") or [])
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    if not tool_messages:
        raise RuntimeError("conductivity ToolNode produced no ToolMessage")
    raw = tool_messages[-1].content
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def run(*, model_path: Path, event_path: Path | None = None) -> dict[str, Any]:
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    previous_model = os.environ.get("ORION_NA_CONDUCTIVITY_MODEL_PATH")
    previous_events = os.environ.get("ORION_RESOURCE_EVENTS_PATH")
    previous_enabled = os.environ.get("ORION_RESOURCE_POOL_ENABLED")
    os.environ["ORION_NA_CONDUCTIVITY_MODEL_PATH"] = str(model_path)
    os.environ["ORION_RESOURCE_POOL_ENABLED"] = "1"
    if event_path is not None:
        event_path = event_path.resolve()
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.unlink(missing_ok=True)
        os.environ["ORION_RESOURCE_EVENTS_PATH"] = str(event_path)
    else:
        os.environ.pop("ORION_RESOURCE_EVENTS_PATH", None)

    runtime = None
    try:
        runtime = build_default_tool_resource_runtime()
        if runtime is None:
            raise RuntimeError("default resource runtime unexpectedly disabled")

        node = ToolNode([predict_formulation_conductivity])
        graph = StateGraph(MessagesState)
        graph.add_node(
            "tools",
            lambda state: runtime.bridge.invoke(
                node,
                state,
                request_id="conductivity-unload-reload-canary",
            ),
        )
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        app = graph.compile()

        first = app.invoke(_state("call-before-unload"))
        first_payload = _tool_payload(first)
        first_snapshot = runtime.snapshot()[0]
        first_pid = first_snapshot["instance_pid"]
        if first_pid is None:
            raise RuntimeError("first conductivity call did not start a worker")

        runtime.governor.demote(
            CONDUCTIVITY_PROCESS_RESOURCE_ID,
            ResidencyLevel.CATALOG,
        )
        after_demote = runtime.governor.resource_observation(
            CONDUCTIVITY_PROCESS_RESOURCE_ID
        )
        if bool(after_demote["resident"]):
            raise RuntimeError("conductivity resource remained resident after demote")
        if after_demote["instance_pid"] is not None:
            raise RuntimeError("conductivity worker PID remained after demote")

        second = app.invoke(_state("call-after-unload"))
        second_payload = _tool_payload(second)
        second_snapshot = runtime.snapshot()[0]
        second_pid = second_snapshot["instance_pid"]
        if second_pid is None:
            raise RuntimeError("second conductivity call did not restart a worker")

        same_output = first_payload == second_payload
        pid_changed = int(first_pid) != int(second_pid)
        state = runtime.governor.state(CONDUCTIVITY_PROCESS_RESOURCE_ID)
        result = {
            "schema_version": "cwsr-conductivity-resource-unload-reload-canary-v1",
            "model_path": str(model_path),
            "model_sha256": _sha256_file(model_path),
            "first_pid": int(first_pid),
            "second_pid": int(second_pid),
            "pid_changed_after_unload": pid_changed,
            "scientific_output_identical": same_output,
            "first_output": first_payload,
            "second_output": second_payload,
            "promotions": int(state.promotions),
            "demotions": int(state.demotions),
            "refaults": int(state.refaults),
            "first_prepare_seconds": first_snapshot.get("last_prepare_seconds"),
            "second_prepare_seconds": second_snapshot.get("last_prepare_seconds"),
            "second_measured_cost": second_snapshot.get("measured_cost"),
            "active_count_after_second_call": int(second_snapshot["active_count"]),
            "resident_after_second_call": bool(second_snapshot["resident"]),
            "event_path": str(event_path) if event_path is not None else None,
            "pass": bool(
                pid_changed
                and same_output
                and int(state.promotions) >= 2
                and int(state.demotions) >= 1
                and int(second_snapshot["active_count"]) == 0
                and bool(second_snapshot["resident"])
            ),
            "interpretation_guard": (
                "This canary validates physical unload/reload feasibility and output "
                "stability for one RF resource. It is not evidence that any TTL or "
                "adaptive retention policy improves a workload."
            ),
        }
        result["canary_sha256"] = _sha256_json(result)
        return result
    finally:
        if runtime is not None:
            runtime.close()
        if previous_model is None:
            os.environ.pop("ORION_NA_CONDUCTIVITY_MODEL_PATH", None)
        else:
            os.environ["ORION_NA_CONDUCTIVITY_MODEL_PATH"] = previous_model
        if previous_events is None:
            os.environ.pop("ORION_RESOURCE_EVENTS_PATH", None)
        else:
            os.environ["ORION_RESOURCE_EVENTS_PATH"] = previous_events
        if previous_enabled is None:
            os.environ.pop("ORION_RESOURCE_POOL_ENABLED", None)
        else:
            os.environ["ORION_RESOURCE_POOL_ENABLED"] = previous_enabled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--event-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(model_path=args.model_path, event_path=args.event_path)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output.resolve())
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
