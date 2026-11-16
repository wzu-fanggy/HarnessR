from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from orion.tools.electrolyte_design.conductivity_ml import (
    OutOfDomainError,
    SodiumConductivityRandomForest,
    features_from_mmol,
    grouped_cross_validation,
    prepare_training_table,
    temperature_c_to_inverse_temperature,
)


def _synthetic_frame() -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    temperatures = (-10.0, 0.0, 10.0, 20.0, 30.0)
    formulations = (
        (2.0, 5.0, 20.0, 3.0),
        (2.5, 8.0, 17.0, 5.0),
        (3.0, 10.0, 15.0, 7.0),
        (3.5, 12.0, 13.0, 9.0),
        (4.0, 15.0, 10.0, 11.0),
        (4.5, 17.0, 8.0, 13.0),
        (5.0, 19.0, 6.0, 15.0),
        (5.5, 21.0, 4.0, 17.0),
    )
    for formulation_index, (salt, ec, emc, pc) in enumerate(formulations):
        total = salt + ec + emc + pc
        x_salt = salt / total
        x_pc = pc / (ec + emc + pc)
        x_ec = ec / (ec + emc)
        for temperature_c in temperatures:
            inv_t = temperature_c_to_inverse_temperature(temperature_c)
            # Smooth deterministic response used only to test plumbing and grouped CV.
            log_cond = (
                -3.7
                + 0.020 * (temperature_c + 10.0)
                + 1.8 * x_salt
                + 0.25 * x_ec
                - 0.35 * x_pc
            )
            for replicate in range(3):
                rows.append(
                    {
                        "NaPF6_mmol": salt,
                        "EC_mmol": ec,
                        "EMC_mmol": emc,
                        "PC_mmol": pc,
                        "inverse temperature": inv_t,
                        "conductivity": 10.0 ** (log_cond + (replicate - 1) * 0.002),
                        "source": float(formulation_index),
                    }
                )
    return pd.DataFrame(rows)


def test_temperature_conversion_matches_1000_over_kelvin() -> None:
    assert temperature_c_to_inverse_temperature(25.0) == pytest.approx(1000.0 / 298.15)


def test_feature_construction_matches_public_na_notebook_definitions() -> None:
    features = features_from_mmol(
        temperature_c=20.0,
        napf6_mmol=4.0,
        ec_mmol=8.0,
        emc_mmol=12.0,
        pc_mmol=6.0,
    )
    assert features.x_NaSalt == pytest.approx(4.0 / 30.0)
    assert features.x_EC == pytest.approx(8.0 / 20.0)
    assert features.x_PC == pytest.approx(6.0 / 26.0)


def test_prepare_training_table_aggregates_replicates() -> None:
    table = prepare_training_table([_synthetic_frame()])
    assert len(table) == 8 * 5
    assert table["formulation_group"].nunique() == 8
    assert set(table["replicate_count"].unique()) == {3}
    assert np.isfinite(table["log10_conductivity"]).all()


def test_random_forest_predicts_in_domain_and_rejects_extrapolation() -> None:
    table = prepare_training_table([_synthetic_frame()])
    model = SodiumConductivityRandomForest(n_estimators=64, random_state=7).fit(table)

    prediction = model.predict_mmol(
        temperature_c=20.0,
        napf6_mmol=3.5,
        ec_mmol=12.0,
        emc_mmol=13.0,
        pc_mmol=9.0,
    )
    assert prediction.in_domain is True
    assert prediction.conductivity_s_cm > 0
    assert prediction.conductivity_ms_cm == pytest.approx(
        prediction.conductivity_s_cm * 1000.0
    )
    assert prediction.estimated_factor_1sigma >= 1.0

    with pytest.raises(OutOfDomainError):
        model.predict_mmol(
            temperature_c=80.0,
            napf6_mmol=3.5,
            ec_mmol=12.0,
            emc_mmol=13.0,
            pc_mmol=9.0,
        )


def test_grouped_cv_keeps_formulations_separate_and_returns_finite_metrics() -> None:
    table = prepare_training_table([_synthetic_frame()])
    metrics = grouped_cross_validation(table, folds=4, random_state=11)
    assert metrics.folds == 4
    assert metrics.unique_formulations == 8
    assert metrics.rows == 40
    assert math.isfinite(metrics.rmse_log10)
    assert math.isfinite(metrics.mae_log10)
    assert math.isfinite(metrics.r2_log10)
    assert metrics.multiplicative_rmse_factor >= 1.0
