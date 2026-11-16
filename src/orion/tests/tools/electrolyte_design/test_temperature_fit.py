from __future__ import annotations

import math

import pytest

from tools.electrolyte_design.temperature_fit import fit_temperature_laws


def _request_for_law(law, temperatures_c=(-20, -5, 10, 25, 40, 55, 70)):
    return {
        "formulation_id": "measured-formulation-01",
        "measurements": [
            {
                "temperature_c": float(temperature_c),
                "conductivity_ms_cm": 10.0 ** law(temperature_c + 273.15),
            }
            for temperature_c in temperatures_c
        ],
        "provenance": {"dataset": "controlled-test-series"},
    }


def test_arrhenius_series_recovers_activation_energy_and_is_preferred() -> None:
    activation_energy_j_mol = 18_000.0
    slope = -activation_energy_j_mol / (1000.0 * 8.31446261815324 * math.log(10.0))
    result = fit_temperature_laws(
        _request_for_law(lambda temperature_k: 4.0 + slope * 1000.0 / temperature_k)
    )

    models = {row["model"]: row for row in result["models"]}
    assert result["status"] == "fitted"
    assert result["preferred_by_aicc"] == "arrhenius"
    assert result["measurement_count"] == 7
    assert result["temperature_range_c"] == [-20.0, 70.0]
    assert models["arrhenius"]["parameters"]["activation_energy_j_mol"] == pytest.approx(
        activation_energy_j_mol, rel=1e-8
    )
    assert models["arrhenius"]["loo_rmse_log10"] < 1e-10
    assert result["provenance"] == {"dataset": "controlled-test-series"}
    assert "does not establish a transport mechanism" in result["interpretation_boundary"]


def test_vtf_series_is_preferred_and_recovers_parameters() -> None:
    intercept, b_kelvin, t0_kelvin = 2.1, 410.0, 165.0
    result = fit_temperature_laws(
        _request_for_law(
            lambda temperature_k: intercept - b_kelvin / (temperature_k - t0_kelvin),
            temperatures_c=(-30, -15, 0, 15, 30, 45, 60, 75),
        )
    )

    models = {row["model"]: row for row in result["models"]}
    assert result["preferred_by_aicc"] == "vtf"
    assert models["vtf"]["parameters"]["b_kelvin"] == pytest.approx(b_kelvin, rel=1e-6)
    assert models["vtf"]["parameters"]["t0_kelvin"] == pytest.approx(t0_kelvin, rel=1e-6)
    assert models["vtf"]["loo_rmse_log10"] < 1e-8


@pytest.mark.parametrize(
    "measurements, message",
    [
        (
            [{"temperature_c": value, "conductivity_ms_cm": 1.0} for value in range(4)],
            "at least 5",
        ),
        ([{"temperature_c": 25.0, "conductivity_ms_cm": 1.0}] * 5, "distinct"),
        (
            [
                {
                    "temperature_c": float(value),
                    "conductivity_ms_cm": -1.0 if value == 4 else 1.0,
                }
                for value in range(5)
            ],
            "positive",
        ),
    ],
)
def test_invalid_measurement_series_is_rejected(measurements, message) -> None:
    with pytest.raises(ValueError, match=message):
        fit_temperature_laws({"measurements": measurements})
