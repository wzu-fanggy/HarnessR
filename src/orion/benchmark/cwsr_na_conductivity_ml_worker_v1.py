"""Resident CPU worker for the bounded Na-electrolyte conductivity model.

The parent process sends a compact, already-cleaned public training table once per
materialization.  The worker fits the deterministic RandomForest model and then serves
prediction requests until CWSR demotes the Process capability.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import joblib
import pandas as pd
import sklearn

from tools.electrolyte_design.conductivity_ml import (
    ConductivityFeatures,
    OutOfDomainError,
    SodiumConductivityRandomForest,
)


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _read_request(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    return payload


def main() -> int:
    model: SodiumConductivityRandomForest | None = None
    _emit(
        {
            "status": "BOOTED",
            "pid": os.getpid(),
            "versions": {
                "python": sys.version.split()[0],
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
    )

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = _read_request(raw)
            action = request.get("action")
            if action == "stop":
                _emit({"status": "STOPPED", "pid": os.getpid()})
                return 0
            if action == "fit":
                records = request.get("records")
                if not isinstance(records, list) or not records:
                    raise ValueError("fit requires non-empty records")
                table = pd.DataFrame.from_records(records)
                model = SodiumConductivityRandomForest(
                    n_estimators=int(request.get("n_estimators", 192)),
                    random_state=int(request.get("random_state", 20260906)),
                ).fit(table)
                _emit(
                    {
                        "status": "READY",
                        "pid": os.getpid(),
                        "training_rows": model.training_rows,
                        "training_formulations": model.training_formulations,
                        "domain": model.domain(),
                    }
                )
                continue
            if action == "load":
                model_path = Path(str(request.get("model_path") or "")).expanduser().resolve()
                if not model_path.exists():
                    raise FileNotFoundError(model_path)
                load_started = perf_counter()
                loaded = joblib.load(model_path)
                model_load_seconds = perf_counter() - load_started
                if not isinstance(loaded, SodiumConductivityRandomForest) or not loaded.fitted:
                    raise TypeError("serialized model is not a fitted SodiumConductivityRandomForest")
                model = loaded
                _emit(
                    {
                        "status": "READY",
                        "pid": os.getpid(),
                        "training_rows": model.training_rows,
                        "training_formulations": model.training_formulations,
                        "domain": model.domain(),
                        "model_path": str(model_path),
                        "model_load_seconds": model_load_seconds,
                    }
                )
                continue
            if action == "predict_features":
                if model is None:
                    raise RuntimeError("model is not fitted")
                values = request.get("features")
                if not isinstance(values, dict):
                    raise ValueError("predict_features requires feature mapping")
                features = ConductivityFeatures(
                    inverse_temperature=float(values["inverse_temperature"]),
                    x_NaSalt=float(values["x_NaSalt"]),
                    x_EC=float(values["x_EC"]),
                    x_PC=float(values["x_PC"]),
                )
                try:
                    prediction = model.predict_features(features)
                except OutOfDomainError as exc:
                    _emit(
                        {
                            "status": "OUT_OF_DOMAIN",
                            "pid": os.getpid(),
                            "reason": str(exc),
                        }
                    )
                    continue
                _emit(
                    {
                        "status": "RESULT",
                        "pid": os.getpid(),
                        "prediction": prediction.to_dict(),
                    }
                )
                continue
            _emit({"status": "ERROR", "pid": os.getpid(), "error": "unknown_action"})
        except Exception as exc:
            _emit(
                {
                    "status": "ERROR",
                    "pid": os.getpid(),
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
