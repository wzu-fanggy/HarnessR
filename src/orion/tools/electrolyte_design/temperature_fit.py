"""Deterministic Arrhenius/VTF fitting for electrolyte conductivity series."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


MIN_TEMPERATURE_POINTS = 5
GAS_CONSTANT_J_MOL_K = 8.31446261815324


class TemperatureFitInputError(ValueError):
    """请求数据不满足输入合同，允许Agent修正后重试。"""


def _vtf(temperature_k: np.ndarray, intercept: float, b_kelvin: float,
         t0_kelvin: float) -> np.ndarray:
    return intercept - b_kelvin / (temperature_k - t0_kelvin)


def _aicc(observed: np.ndarray, predicted: np.ndarray, parameter_count: int) -> float:
    n = int(observed.size)
    rss = max(float(np.sum(np.square(observed - predicted))), 1e-15)
    if n <= parameter_count + 1:
        return math.inf
    aic = n * math.log(rss / n) + 2.0 * parameter_count
    return aic + (2.0 * parameter_count * (parameter_count + 1)) / (
        n - parameter_count - 1
    )


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(observed - predicted))))


def _fit_arrhenius(temperature_k: np.ndarray, log10_ms_cm: np.ndarray) -> dict[str, Any]:
    inverse_temperature = 1000.0 / temperature_k
    slope, intercept = np.polyfit(inverse_temperature, log10_ms_cm, 1)
    fitted = intercept + slope * inverse_temperature
    activation_energy_j_mol = -float(slope) * 1000.0 * GAS_CONSTANT_J_MOL_K * math.log(10.0)
    return {
        "model": "arrhenius",
        "parameters": {
            "intercept_log10_ms_cm": float(intercept),
            "slope_per_1000_over_k": float(slope),
            "activation_energy_j_mol": activation_energy_j_mol,
            "activation_energy_kj_mol": activation_energy_j_mol / 1000.0,
        },
        "fitted": fitted,
        "aicc": _aicc(log10_ms_cm, fitted, 2),
        "rmse_log10": _rmse(log10_ms_cm, fitted),
        "residual_summary_log10": _residual_summary(log10_ms_cm, fitted),
    }


def _fit_vtf(temperature_k: np.ndarray, log10_ms_cm: np.ndarray) -> dict[str, Any]:
    minimum_temperature = float(np.min(temperature_k))
    upper_t0 = minimum_temperature - 1.0
    lower_t0 = max(1.0, minimum_temperature - 250.0)
    initial_t0 = min(max(minimum_temperature - 80.0, lower_t0), upper_t0)
    initial = (float(np.max(log10_ms_cm)) + 1.0, 500.0, initial_t0)
    parameters, _covariance = curve_fit(
        _vtf,
        temperature_k,
        log10_ms_cm,
        p0=initial,
        bounds=((-20.0, 0.0, lower_t0), (20.0, 20000.0, upper_t0)),
        maxfev=50000,
    )
    fitted = _vtf(temperature_k, *parameters)
    return {
        "model": "vtf",
        "parameters": {
            "intercept_log10_ms_cm": float(parameters[0]),
            "b_kelvin": float(parameters[1]),
            "t0_kelvin": float(parameters[2]),
        },
        "fitted": fitted,
        "aicc": _aicc(log10_ms_cm, fitted, 3),
        "rmse_log10": _rmse(log10_ms_cm, fitted),
        "residual_summary_log10": _residual_summary(log10_ms_cm, fitted),
    }


def _residual_summary(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residuals = observed - predicted
    return {
        "mean": float(np.mean(residuals)),
        "mean_absolute": float(np.mean(np.abs(residuals))),
        "max_absolute": float(np.max(np.abs(residuals))),
    }


def _loo_rmse(
    temperature_k: np.ndarray,
    log10_ms_cm: np.ndarray,
    fitter,
) -> float | None:
    predictions: list[float] = []
    observed: list[float] = []
    for index in range(int(temperature_k.size)):
        mask = np.ones(temperature_k.size, dtype=bool)
        mask[index] = False
        try:
            fitted = fitter(temperature_k[mask], log10_ms_cm[mask])
            parameters = fitted["parameters"]
            if fitted["model"] == "arrhenius":
                prediction = (
                    parameters["intercept_log10_ms_cm"]
                    + parameters["slope_per_1000_over_k"]
                    * 1000.0 / float(temperature_k[index])
                )
            else:
                prediction = float(
                    _vtf(
                        np.asarray([temperature_k[index]], dtype=float),
                        parameters["intercept_log10_ms_cm"],
                        parameters["b_kelvin"],
                        parameters["t0_kelvin"],
                    )[0]
                )
        except (RuntimeError, ValueError, FloatingPointError):
            return None
        predictions.append(float(prediction))
        observed.append(float(log10_ms_cm[index]))
    return _rmse(np.asarray(observed), np.asarray(predictions))


def _normalize_measurements(
    measurements: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    if len(measurements) < MIN_TEMPERATURE_POINTS:
        raise TemperatureFitInputError(
            f"at least {MIN_TEMPERATURE_POINTS} distinct temperature points are required"
        )
    rows: list[dict[str, float]] = []
    seen: set[float] = set()
    for raw in measurements:
        try:
            temperature_c = float(raw["temperature_c"])
            conductivity_ms_cm = float(raw["conductivity_ms_cm"])
        except (KeyError, TypeError, ValueError) as error:
            raise TemperatureFitInputError('每个测量点须有可转换为数值的temperature_c与conductivity_ms_cm') from error
        if not math.isfinite(temperature_c) or temperature_c <= -273.15:
            raise TemperatureFitInputError("temperature_c must be finite and above absolute zero")
        if not math.isfinite(conductivity_ms_cm) or conductivity_ms_cm <= 0:
            raise TemperatureFitInputError("conductivity_ms_cm must be finite and positive")
        if temperature_c in seen:
            raise TemperatureFitInputError("temperature points must be distinct")
        seen.add(temperature_c)
        rows.append({
            "temperature_c": temperature_c,
            "conductivity_ms_cm": conductivity_ms_cm,
        })
    rows.sort(key=lambda row: row["temperature_c"])
    temperature_k = np.asarray([row["temperature_c"] + 273.15 for row in rows])
    log10_ms_cm = np.log10(
        np.asarray([row["conductivity_ms_cm"] for row in rows], dtype=float)
    )
    return temperature_k, log10_ms_cm, rows


def fit_temperature_laws(request: Mapping[str, object]) -> dict[str, Any]:
    """Fit Arrhenius and VTF models without turning predictions into observations."""

    measurements = request.get("measurements")
    if not isinstance(measurements, list) or not all(
        isinstance(item, Mapping) for item in measurements
    ):
        raise TemperatureFitInputError("measurements must be a list of objects")
    temperature_k, log10_ms_cm, normalized = _normalize_measurements(measurements)

    arrhenius = _fit_arrhenius(temperature_k, log10_ms_cm)
    vtf = _fit_vtf(temperature_k, log10_ms_cm)
    arrhenius["loo_rmse_log10"] = _loo_rmse(
        temperature_k, log10_ms_cm, _fit_arrhenius
    )
    vtf["loo_rmse_log10"] = _loo_rmse(temperature_k, log10_ms_cm, _fit_vtf)
    for result in (arrhenius, vtf):
        result.pop("fitted", None)

    preferred = min((arrhenius, vtf), key=lambda row: (row["aicc"], row["model"]))
    return {
        "status": "fitted",
        "formulation_id": str(request.get("formulation_id") or "").strip() or None,
        "measurement_count": len(normalized),
        "temperature_range_c": [
            normalized[0]["temperature_c"], normalized[-1]["temperature_c"]
        ],
        "conductivity_unit": "mS/cm",
        "response_scale": "log10(mS/cm)",
        "models": [arrhenius, vtf],
        "preferred_by_aicc": preferred["model"],
        "provenance": request.get("provenance"),
        "interpretation_boundary": (
            "Statistical fit to supplied measurements; it does not establish a transport mechanism "
            "or authorize extrapolation beyond the observed temperature range."
        ),
    }
