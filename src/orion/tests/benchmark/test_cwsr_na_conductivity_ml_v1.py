from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    return json.loads((ROOT / "benchmark" / name).read_text(encoding="utf-8"))


def test_public_na_conductivity_model_has_nontrivial_heldout_formulation_accuracy() -> None:
    result = _load("cwsr_na_conductivity_rf_validation_v1_result.json")
    assert result["status"] == "PASS"
    assert result["source"]["raw_rows_by_generation"] == [720, 720, 720]
    assert result["model"]["training_rows_after_replicate_aggregation"] == 647
    assert result["model"]["training_unique_formulations"] == 72

    cv = result["held_out_formulation_cv"]
    assert cv["folds"] == 5
    assert cv["unique_formulations"] == 72
    assert cv["rmse_log10"] < 0.25
    assert cv["mae_log10"] < 0.15
    assert cv["r2_log10"] > 0.70


def test_real_ml_process_is_evicted_and_refaulted_with_same_predictions() -> None:
    result = _load("cwsr_na_conductivity_ml_process_v1_result.json")
    assert result["status"] == "PASS"
    assert result["source"]["raw_rows"] == 2160
    assert result["source"]["aggregated_training_rows"] == 647
    assert result["pressure"]["ml_running_after_pressure"] is False
    assert result["new_pid_after_refault"] is True
    assert result["predictions_identical_after_refault"] is True
    assert result["ml_refault_count"] == 1
    assert result["out_of_domain_guard"]["status"] == "OUT_OF_DOMAIN"
    assert result["first_materialization"]["rss_mb"] > 100.0
    assert result["refault_materialization"]["rss_mb"] > 100.0
