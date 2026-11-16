"""Resident JSONL worker for electrolyte temperature-law fitting."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np
import scipy


FIT_MODULE_PATH = Path(__file__).with_name("temperature_fit.py")
FIT_MODULE_SPEC = importlib.util.spec_from_file_location(
    "orion_temperature_fit_worker_impl",
    FIT_MODULE_PATH,
)
if FIT_MODULE_SPEC is None or FIT_MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot load temperature-fit implementation: {FIT_MODULE_PATH}")
FIT_MODULE = importlib.util.module_from_spec(FIT_MODULE_SPEC)
FIT_MODULE_SPEC.loader.exec_module(FIT_MODULE)
fit_temperature_laws = FIT_MODULE.fit_temperature_laws


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> int:
    _emit({"status": "BOOTED", "pid": os.getpid(), "versions": {
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
    }})
    _emit({"status": "READY", "pid": os.getpid()})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            action = payload.get("action")
            if action == "stop":
                _emit({"status": "STOPPED", "pid": os.getpid()})
                return 0
            if action == "fit_temperature_laws":
                request = payload.get("request")
                if not isinstance(request, dict):
                    raise ValueError("fit_temperature_laws requires request object")
                _emit({"status": "RESULT", "pid": os.getpid(),
                       "result": fit_temperature_laws(request)})
                continue
            raise ValueError(f"unknown action: {action!r}")
        except Exception as exc:
            _emit({"status": "ERROR", "pid": os.getpid(),
                   "error": type(exc).__name__, "message": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
