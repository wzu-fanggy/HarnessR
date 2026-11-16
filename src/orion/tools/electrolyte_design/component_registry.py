"""组分注册表：电解液组分的基本物性与模型可评估标记。

用途：
1. ``generate_candidate_space`` 校验组分的合法性，并判断候选是否能被现有电导率模型评估；
2. ``check_formulation_feasibility`` 用纯组分物性做保守筛选（凝固、闪点下限、黏度粗估）。

物性来源
--------
溶剂物性取自两篇综述的公开表格，数值在来源间存在分歧时保留区间并取代表值：

* RSC 2025 review, Table 2, DOI 10.1039/d5ya00065c
* Energy Environ. Mater. 2022, 5, e12043, Table 1, DOI 10.1002/eom2.12043
* 光学期刊网《高温锂离子电池电解液研究进展》表 2（与上述两篇一致）

盐的摩尔质量按元素原子量计算，不引用外部来源。

重要限制
--------
本表只提供**纯组分**物性。混合体系的凝固点、氧化还原电位、电导率不能由纯组分
物性直接相加得到，因此不在本模块内给出数值，由上层工具显式标注
``requires_measurement``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

ComponentKind = Literal["salt", "solvent", "additive"]

# 电导率模型当前能评估的组分集合，与 conductivity_ml 的 SUPPORTED_* 保持一致。
MODEL_SUPPORTED_SALT = "NaPF6"
MODEL_SUPPORTED_SOLVENTS = frozenset({"EC", "EMC", "PC"})

# 训练数据覆盖范围（由 prepare_training_table 在 647 行上实算，2026-10-19）。
TRAINING_X_NASALT_RANGE = (0.0501, 0.1997)
TRAINING_TEMPERATURE_RANGE_C = (-20.0, 60.0)


@dataclass(frozen=True)
class Component:
    name: str
    kind: ComponentKind
    molar_mass_g_mol: float
    # 纯组分物性；None 表示本表未收录，调用方必须按"未知"处理而不是按 0 处理。
    melting_point_c: float | None = None
    boiling_point_c: float | None = None
    flash_point_c: float | None = None
    viscosity_cp_25c: float | None = None
    density_g_cm3_25c: float | None = None
    dielectric_constant: float | None = None
    # 物性区间（来源间分歧时记录），格式 (min, max)
    melting_point_range_c: tuple[float, float] | None = None
    source: str = ""
    notes: str = ""


_SOLVENT_SOURCE = "10.1039/d5ya00065c Table 2; 10.1002/eom2.12043 Table 1"

_SOLVENTS: dict[str, Component] = {
    "EC": Component(
        name="EC", kind="solvent", molar_mass_g_mol=88.06,
        melting_point_c=36.4, boiling_point_c=248.0, flash_point_c=160.0,
        viscosity_cp_25c=1.90, density_g_cm3_25c=1.32, dielectric_constant=89.8,
        source=_SOLVENT_SOURCE,
        notes="环状碳酸酯；熔点 36.4 ℃，纯态室温为固体，必须与共溶剂混用。"
              "黏度常引用 40 ℃ 值 1.90 cP。",
    ),
    "PC": Component(
        name="PC", kind="solvent", molar_mass_g_mol=102.09,
        melting_point_c=-48.8, boiling_point_c=242.0, flash_point_c=132.0,
        viscosity_cp_25c=2.53, density_g_cm3_25c=1.20, dielectric_constant=64.9,
        source=_SOLVENT_SOURCE,
    ),
    "EMC": Component(
        name="EMC", kind="solvent", molar_mass_g_mol=104.10,
        melting_point_c=-53.0, boiling_point_c=110.0, flash_point_c=23.0,
        viscosity_cp_25c=0.65, density_g_cm3_25c=1.01, dielectric_constant=2.96,
        source=_SOLVENT_SOURCE,
    ),
    "DEC": Component(
        name="DEC", kind="solvent", molar_mass_g_mol=118.13,
        melting_point_c=-74.3, boiling_point_c=126.0, flash_point_c=31.0,
        viscosity_cp_25c=0.75, density_g_cm3_25c=0.97, dielectric_constant=2.81,
        source=_SOLVENT_SOURCE,
    ),
    "DMC": Component(
        name="DMC", kind="solvent", molar_mass_g_mol=90.08,
        melting_point_c=4.6, boiling_point_c=91.0, flash_point_c=18.0,
        viscosity_cp_25c=0.59, density_g_cm3_25c=1.06, dielectric_constant=3.11,
        source=_SOLVENT_SOURCE,
    ),
    "FEC": Component(
        name="FEC", kind="solvent", molar_mass_g_mol=106.05,
        melting_point_c=19.0, boiling_point_c=249.0, flash_point_c=120.0,
        viscosity_cp_25c=2.35, density_g_cm3_25c=1.45, dielectric_constant=78.4,
        melting_point_range_c=(18.0, 22.0),
        source=_SOLVENT_SOURCE,
        notes="熔点来源间有 18/20/22 ℃ 三种取值，代表值取 19 ℃，区间记录在 "
              "melting_point_range_c。",
    ),
    "VC": Component(
        name="VC", kind="solvent", molar_mass_g_mol=86.05,
        melting_point_c=22.0, boiling_point_c=162.0, flash_point_c=73.0,
        viscosity_cp_25c=2.23, density_g_cm3_25c=1.35, dielectric_constant=126.0,
        source=_SOLVENT_SOURCE,
        notes="通常作为添加剂使用而非主溶剂。",
    ),
}

_SALTS: dict[str, Component] = {
    "NaPF6": Component(
        name="NaPF6", kind="salt", molar_mass_g_mol=167.95,
        source="按元素原子量计算：Na 22.99 + P 30.97 + 6F 19.00",
        notes="当前电导率模型唯一支持的钠盐。",
    ),
    "NaFSI": Component(
        name="NaFSI", kind="salt", molar_mass_g_mol=203.06,
        source="按元素原子量计算：Na 22.99 + N(SO2F)2 180.07",
        notes="模型未覆盖，需要重训后使用。",
    ),
    "NaTFSI": Component(
        name="NaTFSI", kind="salt", molar_mass_g_mol=303.09,
        source="按元素原子量计算：Na 22.99 + N(SO2CF3)2 280.10",
        notes="模型未覆盖，需要重训后使用。",
    ),
    "NaClO4": Component(
        name="NaClO4", kind="salt", molar_mass_g_mol=122.44,
        source="按元素原子量计算：Na 22.99 + Cl 35.45 + 4O 16.00",
        notes="模型未覆盖；强氧化性，配制与干燥条件有额外限制。",
    ),
}

_ADDITIVES: dict[str, Component] = {
    "FEC": Component(
        name="FEC", kind="additive", molar_mass_g_mol=106.05,
        melting_point_c=19.0, boiling_point_c=249.0, flash_point_c=120.0,
        viscosity_cp_25c=2.35, density_g_cm3_25c=1.45,
        melting_point_range_c=(18.0, 22.0),
        source=_SOLVENT_SOURCE,
        notes="成膜添加剂；按质量分数添加，不参与主溶剂摩尔份额。",
    ),
    "VC": Component(
        name="VC", kind="additive", molar_mass_g_mol=86.05,
        melting_point_c=22.0, boiling_point_c=162.0, flash_point_c=73.0,
        density_g_cm3_25c=1.35,
        source=_SOLVENT_SOURCE,
        notes="成膜添加剂。",
    ),
}

_ALL: dict[str, Component] = {**_SALTS, **_SOLVENTS, **_ADDITIVES}


def get_component(name: str) -> Component | None:
    """按名称查组分；大小写与首尾空白不敏感。"""

    if not isinstance(name, str):
        return None
    key = name.strip()
    if key in _ALL:
        return _ALL[key]
    upper = key.upper()
    for candidate_name, component in _ALL.items():
        if candidate_name.upper() == upper:
            return component
    return None


def require_component(name: str, *, kind: ComponentKind | None = None) -> Component:
    component = get_component(name)
    if component is None:
        known = ", ".join(sorted(_ALL))
        raise ValueError(f"未知组分 {name!r}；已收录：{known}")
    if kind is not None and component.kind != kind:
        raise ValueError(
            f"组分 {name!r} 的类别是 {component.kind!r}，此处要求 {kind!r}"
        )
    return component


def is_model_supported_salt(name: str) -> bool:
    return name.strip().upper() == MODEL_SUPPORTED_SALT.upper()


def is_model_supported_solvent(name: str) -> bool:
    return name.strip().upper() in MODEL_SUPPORTED_SOLVENTS


def model_domain_report(salt: str, solvents: list[str]) -> dict[str, object]:
    """判断一组组分能否被当前电导率模型评估。

    返回 ``supported`` 与逐项原因。``EC`` 与 ``EMC`` 必须至少有一个非零，
    否则 x_EC 的分母为 0，特征无定义。
    """

    reasons: list[str] = []
    if not is_model_supported_salt(salt):
        reasons.append(f"盐 {salt} 不在模型域内（仅支持 {MODEL_SUPPORTED_SALT}）")
    unsupported = [name for name in solvents if not is_model_supported_solvent(name)]
    if unsupported:
        reasons.append(
            "溶剂不在模型域内：" + ", ".join(unsupported)
            + f"（仅支持 {', '.join(sorted(MODEL_SUPPORTED_SOLVENTS))}）"
        )
    return {"supported": not reasons, "reasons": reasons}


def arrhenius_blend_viscosity(
    shares: dict[str, float], *, temperature_c: float | None = None
) -> dict[str, object]:
    """按 Arrhenius 混合规则估算混合溶剂黏度。

    ``ln(eta_mix) = sum(x_i * ln(eta_i))``

    这是黏度量级的粗估，不包含过量体积与相互作用修正，误差通常在数十个百分点。
    缺少任一纯组分黏度时返回 ``status="insufficient_data"``，不做插补。
    """

    usable: list[tuple[str, float, float]] = []
    missing: list[str] = []
    for name, share in shares.items():
        if share <= 0:
            continue
        component = get_component(name)
        if component is None or component.viscosity_cp_25c is None:
            missing.append(name)
            continue
        usable.append((name, share, component.viscosity_cp_25c))

    if missing:
        return {
            "status": "insufficient_data",
            "reason": "缺少纯组分黏度：" + ", ".join(missing),
            "value_cp": None,
        }
    total_share = sum(item[1] for item in usable)
    if total_share <= 0:
        return {"status": "insufficient_data", "reason": "没有正的溶剂份额", "value_cp": None}

    log_eta = sum((share / total_share) * math.log(eta) for _, share, eta in usable)
    return {
        "status": "estimated",
        "value_cp": float(math.exp(log_eta)),
        "method": "Arrhenius 混合规则 ln(eta)=sum(x_i ln(eta_i))，纯组分黏度取 25 ℃ 值",
        "temperature_c": temperature_c,
        "note": "量级估计，未含过量体积与相互作用修正",
    }


def flash_point_lower_bound(shares: dict[str, float]) -> dict[str, object]:
    """闪点的保守下限：取参与组分的纯组分闪点最小值。

    混合体系闪点不能由纯组分线性相加得到；取最小值是**安全侧**的保守估计，
    不是预测值。返回中显式标注这一点。
    """

    values: list[tuple[str, float]] = []
    missing: list[str] = []
    for name, share in shares.items():
        if share <= 0:
            continue
        component = get_component(name)
        if component is None or component.flash_point_c is None:
            missing.append(name)
            continue
        values.append((name, component.flash_point_c))

    if not values:
        return {
            "status": "insufficient_data",
            "reason": "缺少纯组分闪点：" + (", ".join(missing) or "无可用组分"),
            "value_c": None,
        }
    name, value = min(values, key=lambda item: item[1])
    return {
        "status": "lower_bound",
        "value_c": float(value),
        "controlling_component": name,
        "method": "取纯组分闪点最小值",
        "note": "保守下限，不是混合物闪点预测值；缺少数据的组分："
                + (", ".join(missing) if missing else "无"),
    }


def list_known_components() -> dict[str, list[dict[str, object]]]:
    """返回注册表摘要，供工具在报错时提示候选组分。"""

    def brief(component: Component) -> dict[str, object]:
        return {
            "name": component.name,
            "kind": component.kind,
            "molar_mass_g_mol": component.molar_mass_g_mol,
            "melting_point_c": component.melting_point_c,
            "model_supported": (
                is_model_supported_salt(component.name)
                if component.kind == "salt"
                else is_model_supported_solvent(component.name)
            ),
        }

    return {
        "salts": [brief(item) for item in _SALTS.values()],
        "solvents": [brief(item) for item in _SOLVENTS.values()],
        "additives": [brief(item) for item in _ADDITIVES.values()],
    }
