"""Resident JSONL worker for robust multi-temperature formulation ranking.

The worker fits its group-bootstrap random-forest ensemble once at start-up from
the frozen public bundle (reserved holdout split bucket removed) and then answers
``rank_formulations`` requests.  Ensemble fitting is the establishment cost that
makes this a genuine managed runtime rather than a thin function call; the process
keeps the fitted members warm between requests and clears per-request arrays when
the lease is released.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np
import pandas as pd
import sklearn

from tools.electrolyte_design.robust_ranking import (
    DEFAULT_MEMBERS,
    DEFAULT_TREES_PER_MEMBER,
    development_frames,
    development_training_table,
    fit_robust_ensemble,
    rank_formulations,
    resolve_bundle_dir,
)


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default), flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _member_count() -> int:
    raw = os.environ.get("ORION_ROBUST_RANKING_MEMBERS", "").strip()
    if not raw:
        return DEFAULT_MEMBERS
    return max(2, int(raw))


def main() -> int:
    _emit(
        {
            "status": "BOOTED",
            "pid": os.getpid(),
            "versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
    )
    bundle = resolve_bundle_dir()
    frames = development_frames(bundle)
    full_table = development_training_table(frames, drop_reserved_holdout=False)
    table = development_training_table(frames, drop_reserved_holdout=True)
    ensemble = fit_robust_ensemble(
        table,
        members=_member_count(),
        trees_per_member=DEFAULT_TREES_PER_MEMBER,
    )
    ensemble.excluded_holdout_formulations = int(
        full_table["formulation_group"].nunique() - table["formulation_group"].nunique()
    )
    _emit(
        {
            "status": "READY",
            "pid": os.getpid(),
            "bundle_dir": str(bundle),
            "training_rows": ensemble.training_rows,
            "training_formulations": ensemble.training_formulations,
            "excluded_holdout_formulations": ensemble.excluded_holdout_formulations,
            "ensemble_members": ensemble.member_count,
        }
    )

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
            if action == "rank_formulations":
                request = payload.get("request")
                if not isinstance(request, dict):
                    raise ValueError("rank_formulations requires request object")
                _emit(
                    {
                        "status": "RESULT",
                        "pid": os.getpid(),
                        "result": rank_formulations(request, ensemble=ensemble),
                    }
                )
                continue
            raise ValueError(f"unknown action: {action!r}")
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
