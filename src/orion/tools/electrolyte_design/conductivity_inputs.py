"""核对同一电解液组成的不同输入表示，不允许模型特征覆盖冲突的组成。"""
from __future__ import annotations

from math import isclose, isfinite
from typing import Any

from .conductivity_ml import ConductivityFeatures, features_from_mmol, temperature_c_to_inverse_temperature

FEATURES = ("x_NaSalt", "x_EC", "x_PC")


def resolve_conductivity_inputs(candidate: dict[str, Any], *, temperature_c: float,
                                override: dict[str, Any] | None) -> ConductivityFeatures | None:
    """先从组分物质的量/摩尔比例换算，再核对调用者另给的模型特征。

    x_NaSalt=盐/(盐+全部溶剂)，x_EC=EC/(EC+EMC)，x_PC=PC/全部溶剂。
    不同表示必须一致。只有明确模型特征而没有完整组成时，保持原接口支持。
    """
    sources = [candidate]
    for key in ("conductivity_features", "composition_inputs"):
        if isinstance(candidate.get(key), dict):
            sources.append(candidate[key])
    if override is not None:
        sources.append(override)
    physical = []
    explicit = []
    for payload in sources:
        for kind in ("component_mmol", "molar_ratios"):
            values = payload.get(kind)
            if isinstance(values, dict):
                physical.append(features_from_mmol(
                    temperature_c=temperature_c, napf6_mmol=float(values.get("NaPF6", 0)),
                    ec_mmol=float(values.get("EC", 0)), emc_mmol=float(values.get("EMC", 0)),
                    pc_mmol=float(values.get("PC", 0))))
        features = payload.get("features")
        if not isinstance(features, dict):
            features = payload if any(key in payload for key in FEATURES) else None
        if features is not None:
            if not all(key in features for key in FEATURES):
                raise ValueError("explicit model features require x_NaSalt, x_EC and x_PC")
            values = {key: float(features[key]) for key in FEATURES}
            if not all(isfinite(value) for value in values.values()):
                raise ValueError("explicit model features must be finite")
            explicit.append(ConductivityFeatures(
                inverse_temperature=temperature_c_to_inverse_temperature(temperature_c), **values))
    representations = physical + explicit
    if not representations:
        return None
    reference = representations[0]
    for item in representations[1:]:
        mismatches = [key for key in FEATURES if not isclose(getattr(item, key), getattr(reference, key),
                                                            rel_tol=1e-6, abs_tol=1e-8)]
        if mismatches:
            raise ValueError("conflicting composition representations for " + ", ".join(mismatches)
                             + "; supply the molar composition without inconsistent model features. "
                             + "x_EC=EC/(EC+EMC); x_PC=PC/(EC+EMC+PC); "
                             + "x_NaSalt=NaPF6/(NaPF6+EC+EMC+PC)")
    return reference
