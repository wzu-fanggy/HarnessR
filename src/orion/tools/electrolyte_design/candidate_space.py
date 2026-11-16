"""候选配方空间生成：把"目标条件"变成一组可评估的候选配方。



这是逆向设计流程的起点，与"从文献里检索已有配方"是两件事：



* ``retrieve_reported_formulations``（原 ``query_formulation_space``）回答

  "别人做过什么"；

* 本模块回答"在给定的组分与配比范围内，有哪些组合值得算一遍"。



设计约束

--------

1. **底层表示是绝对摩尔量（mmol）**，不是三个模型特征。

   特征的分母各不相同（x_NaSalt 除盐+全部溶剂，x_EC 除 EC+EMC，x_PC 除全部溶剂），

   直接采样特征再反解会产生不一致的组成。先定组成、再算特征，才能保证自洽。

2. **默认边界来自训练数据**，不是拍的。x_NaSalt 取 0.0501–0.1997，

   温度取 −20–60 ℃，均由 647 行训练表实算得到。超出边界的候选会被标出，不静默接受。

3. **可评估率必须如实报告**。当前电导率模型只支持 NaPF6 与 EC/EMC/PC；

   生成 NaFSI 或 DEC 的组合是允许的（用于规划扩充后的空间），但会被标为

   ``model_supported: false``，不计入可评估率。

"""



from __future__ import annotations



import math

import random

from typing import Any, Iterator



import numpy as np

from langchain_core.tools import tool



from .component_registry import (

    MODEL_SUPPORTED_SALT,

    MODEL_SUPPORTED_SOLVENTS,

    TRAINING_TEMPERATURE_RANGE_C,

    TRAINING_X_NASALT_RANGE,

    arrhenius_blend_viscosity,

    get_component,

    list_known_components,

    require_component,

)





_DEFAULT_TOTAL_SOLVENT_MMOL = 10_000.0

_DEFAULT_N_SAMPLES = 100

_MAX_N_SAMPLES = 5_000





class CandidateSpaceError(ValueError):

    """候选空间定义不合法。"""





def _as_float(value: Any, field_name: str) -> float:

    try:

        return float(value)

    except (TypeError, ValueError) as exc:

        raise CandidateSpaceError(f"{field_name} 必须是数值，收到 {value!r}") from exc





_KIND_BY_PLURAL = {"salts": "salt", "solvents": "solvent", "additives": "additive"}





def _parse_component_specs(

    raw: Any, *, kind: str, allow_empty: bool

) -> list[dict[str, Any]]:

    singular = _KIND_BY_PLURAL[kind]

    if raw is None:

        raw = []

    if not isinstance(raw, list):

        raise CandidateSpaceError(f"{kind} 必须是列表")

    if not raw and not allow_empty:

        raise CandidateSpaceError(f"{kind} 不能为空")



    specs: list[dict[str, Any]] = []

    seen: set[str] = set()

    for item in raw:

        if isinstance(item, str):

            item = {"name": item}

        if not isinstance(item, dict):

            raise CandidateSpaceError(

                f"{kind} 的每一项必须是字符串或对象，收到 {item!r}"

            )

        name = str(item.get("name") or "").strip()

        if not name:

            raise CandidateSpaceError(f"{kind} 中存在缺少 name 的项")

        component = require_component(name)

        if component.kind != singular:

            raise CandidateSpaceError(

                f"{name} 的类别是 {component.kind}，不能放在 {kind} 中"

            )

        if component.name in seen:

            continue

        seen.add(component.name)

        specs.append({"name": component.name, "raw": item})

    return specs





def _parse_range(

    spec: dict[str, Any], key: str, default: tuple[float, float], label: str

) -> tuple[float, float]:

    raw = spec["raw"].get(key)

    if raw is None:

        return default

    if not isinstance(raw, (list, tuple)) or len(raw) != 2:

        raise CandidateSpaceError(f"{label}.{key} 必须是 [下限, 上限]")

    low, high = _as_float(raw[0], key), _as_float(raw[1], key)

    if low > high:

        raise CandidateSpaceError(f"{label}.{key} 的下限大于上限：{low} > {high}")

    if low < 0:

        raise CandidateSpaceError(f"{label}.{key} 不能为负：{low}")

    return low, high





def _simplex_grid(n_components: int, n_parts: int) -> list[tuple[float, ...]]:

    """把总量等分成 n_parts 份，枚举所有份额组合（stars and bars）。"""



    if n_components == 1:

        return [(1.0,)]



    def compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:

        if parts == 1:

            yield (total,)

            return

        for first in range(total, -1, -1):

            for rest in compositions(total - first, parts - 1):

                yield (first, *rest)



    return [

        tuple(value / n_parts for value in combo)

        for combo in compositions(n_parts, n_components)

    ]





def _latin_hypercube_on_simplex(

    n: int, d: int, rng: np.random.Generator

) -> np.ndarray:

    """拉丁超立方采样后按指数化映射到单纯形。



    先得到每维分层的均匀点，再用 ``w_i = -ln(u_i) / sum(-ln(u_j))`` 映射。

    该变换把独立均匀分布映成单纯形上的均匀分布（Dirichlet(1,...,1)），

    且保持单调性，因此分层结构大致保留。归一化后不是严格 LHS，

    在返回中标注为近似。

    """



    strata = (np.arange(n)[:, None] + rng.random((n, d))) / n

    for column in range(d):

        rng.shuffle(strata[:, column])

    weights = -np.log(np.clip(strata, 1e-12, 1.0))

    return weights / weights.sum(axis=1, keepdims=True)





def _mass_fraction(components_mmol: dict[str, float]) -> dict[str, float]:

    masses: dict[str, float] = {}

    for name, amount in components_mmol.items():

        if amount <= 0:

            continue

        component = require_component(name)

        masses[name] = amount * component.molar_mass_g_mol / 1000.0

    total = sum(masses.values())

    if total <= 0:

        return {}

    return {name: mass / total for name, mass in masses.items()}





def _volume_fraction(

    components_mmol: dict[str, float]

) -> tuple[dict[str, float], list[str]]:

    """按纯组分密度估算体积分数。忽略超额体积；缺密度时返回空字典与缺失清单。"""



    volumes: dict[str, float] = {}

    missing: list[str] = []

    for name, amount in components_mmol.items():

        if amount <= 0:

            continue

        component = get_component(name)

        if component is None or component.density_g_cm3_25c is None:

            missing.append(name)

            continue

        mass_g = amount * component.molar_mass_g_mol / 1000.0

        volumes[name] = mass_g / component.density_g_cm3_25c

    if missing or not volumes:

        return {}, missing

    total = sum(volumes.values())

    return {name: volume / total for name, volume in volumes.items()}, []





@tool

def generate_candidate_space(request: dict[str, Any]) -> dict[str, Any]:

    """按组分与配比范围生成候选配方空间。这是逆向设计的第一步。



    参数 ``request``：



    - ``salts``：列表，每项 ``{"name": "NaPF6", "mol_fraction_range": [0.05, 0.20]}``。

      摩尔分数定义为 盐/(盐+全部溶剂)。

    - ``solvents``：列表，每项 ``{"name": "EC", "molar_share_range": [0.0, 1.0]}``。

      ``molar_share_range`` 指该溶剂在溶剂总量中的摩尔份额，全部溶剂份额之和为 1。

    - ``additives``：可选，每项 ``{"name": "FEC", "mass_fraction_range": [0.0, 0.05]}``。

    - ``n_samples``：整数，默认 100，上限 5000。

    - ``method``：``grid`` / ``lhs`` / ``random``，默认 ``lhs``。

    - ``seed``：整数，默认 42，用于复现。

    - ``total_solvent_mmol``：溶剂总摩尔量基准，默认 10000 mmol。

    - ``temperatures_c``：可选，目标温度列表，须落在 −20–60 ℃。

    - ``result_mode``：inline（默认，兼容旧调用）或artifact。大批候选建议artifact，完整结果保存文件，只返回batch_id、统计和两项预览；生成不等于评价或排名。



    返回 ``candidates``、``space``（空间规模、采样数、覆盖率、可评估率）与 ``warnings``。



    默认边界来自训练数据实测（x_NaSalt 0.0501–0.1997）。超出该范围的候选会被标为

    ``outside_training_range``，仍会返回，但不计入可评估数。当前电导率模型只支持

    NaPF6 与 EC/EMC/PC，其他组分生成的候选标 ``model_supported: false``。

    ``grid`` 方法给出可枚举的总点数与覆盖率；``lhs`` / ``random`` 的总点数未知，

    ``coverage`` 返回 null，不假装知道。

    """



    if not isinstance(request, dict):

        raise CandidateSpaceError("request 必须是对象")



    result_mode = request.get("result_mode", "inline")

    if result_mode not in {"inline", "artifact"}:

        raise CandidateSpaceError("result_mode只支持inline或artifact")



    salts = _parse_component_specs(request.get("salts"), kind="salts", allow_empty=False)

    solvents = _parse_component_specs(

        request.get("solvents"), kind="solvents", allow_empty=False

    )

    additives = _parse_component_specs(

        request.get("additives"), kind="additives", allow_empty=True

    )



    n_samples = int(request.get("n_samples") or _DEFAULT_N_SAMPLES)

    if not 1 <= n_samples <= _MAX_N_SAMPLES:

        raise CandidateSpaceError(f"n_samples 必须在 1–{_MAX_N_SAMPLES} 之间")



    method = str(request.get("method") or "lhs").strip().lower()

    if method not in {"grid", "lhs", "random"}:

        raise CandidateSpaceError(f"method 只支持 grid / lhs / random，收到 {method!r}")



    raw_seed = request.get("seed")

    seed = int(42 if raw_seed is None else raw_seed)

    rng = np.random.default_rng(seed)

    random.seed(seed)



    raw_total = request.get("total_solvent_mmol")

    total_solvent_mmol = _as_float(

        _DEFAULT_TOTAL_SOLVENT_MMOL if raw_total is None else raw_total,

        "total_solvent_mmol",

    )

    if total_solvent_mmol <= 0:

        raise CandidateSpaceError("total_solvent_mmol 必须为正")



    temperatures = request.get("temperatures_c") or []

    if not isinstance(temperatures, list):

        raise CandidateSpaceError("temperatures_c 必须是列表")

    low_t, high_t = TRAINING_TEMPERATURE_RANGE_C

    warnings: list[str] = []

    parsed_temperatures: list[float] = []

    for value in temperatures:

        temperature = _as_float(value, "temperatures_c")

        if not low_t <= temperature <= high_t:

            warnings.append(f"温度 {temperature} ℃ 超出训练覆盖 {low_t}–{high_t} ℃")

        parsed_temperatures.append(temperature)



    solvent_names = [spec["name"] for spec in solvents]

    solvent_bounds = {

        spec["name"]: _parse_range(

            spec, "molar_share_range", (0.0, 1.0), f"solvent {spec['name']}"

        )

        for spec in solvents

    }



    if not any(name in {"EC", "EMC"} for name in solvent_names):

        raise CandidateSpaceError(

            "溶剂中必须包含 EC 或 EMC：模型特征 x_EC 的分母是 EC+EMC，"

            "两者都缺席时候选无法评估"

        )

    if "EC" in solvent_names and "EMC" not in solvent_names:

        warnings.append(

            "溶剂集合只有 EC、没有 EMC：x_EC 恒等于 1，模型无法区分 EC 与 EMC 的比例。"

            "建议补上 EMC，或确认这是纯 EC 体系的刻意设定"

        )

    if "EMC" in solvent_names and "EC" not in solvent_names:

        warnings.append(

            "溶剂集合只有 EMC、没有 EC：x_EC 恒等于 0。若目标包含低温性能，"

            "通常需要 EC 或 FEC 参与成膜"

        )



    salt_name = salts[0]["name"]

    if len(salts) > 1:

        other = ", ".join(spec["name"] for spec in salts[1:])

        warnings.append(

            f"一次只按一种钠盐生成：已取列表第一项 {salt_name}，忽略 {other}。"

            "多盐体系请分多次调用后合并"

        )

    salt_low, salt_high = _parse_range(

        salts[0],

        "mol_fraction_range",

        TRAINING_X_NASALT_RANGE,

        f"salt {salt_name}",

    )



    # ---- 构造样本流 ----

    enumerated_total: int | None = None

    grid_combos: list[tuple[tuple[float, ...], float]] = []

    if method == "grid":

        n_solvents = len(solvent_names)

        salt_levels = [salt_low, (salt_low + salt_high) / 2.0, salt_high]



        def _acceptable(shares: tuple[float, ...]) -> bool:

            # EC 与 EMC 份额同时为 0 时 x_EC 无定义，该组合结构上不可评估，

            # 不计入可枚举总数，否则覆盖率会被这个必然被拒的角落拉低。

            return any(

                share > 0

                for name, share in zip(solvent_names, shares)

                if name in {"EC", "EMC"}

            )



        # 逐步加密分格，直到可枚举总数不少于 n_samples，避免"要 100 个只给 21 个"

        for parts in range(1, 31):

            share_pool = _simplex_grid(n_solvents, parts)

            grid_combos = [

                (shares, salt_fraction)

                for shares in share_pool

                if _acceptable(shares)

                for salt_fraction in salt_levels

            ]

            if len(grid_combos) >= n_samples:

                break



        enumerated_total = len(grid_combos)

        if len(grid_combos) > n_samples:

            indices = np.linspace(0, len(grid_combos) - 1, n_samples).round().astype(int)

            grid_combos = [grid_combos[int(index)] for index in indices]



    def sample_stream() -> Iterator[tuple[tuple[float, ...], float]]:

        if method == "grid":

            for shares, salt_fraction in grid_combos:

                yield shares, salt_fraction

            return

        if method == "lhs":

            batch_size = max(n_samples * 4, 64)

            while True:

                batch = _latin_hypercube_on_simplex(

                    batch_size, len(solvent_names), rng

                )

                for row in batch:

                    yield (

                        tuple(float(value) for value in row),

                        float(rng.uniform(salt_low, salt_high)),

                    )

        else:

            while True:

                row = _random_dirichlet_one(len(solvent_names), rng)

                yield row, float(rng.uniform(salt_low, salt_high))



    candidates: list[dict[str, Any]] = []

    rejected = 0

    attempts = 0

    max_attempts = max(n_samples * 50, 500)



    for shares, salt_fraction in sample_stream():

        if len(candidates) >= n_samples or attempts >= max_attempts:

            break

        attempts += 1



        share_map = dict(zip(solvent_names, shares))

        if any(

            not (

                solvent_bounds[name][0] - 1e-9

                <= share_map[name]

                <= solvent_bounds[name][1] + 1e-9

            )

            for name in solvent_names

        ):

            rejected += 1

            continue



        share_sum = sum(share_map.values())

        if share_sum <= 0:

            rejected += 1

            continue

        share_map = {name: value / share_sum for name, value in share_map.items()}



        if not salt_low - 1e-12 <= salt_fraction <= salt_high + 1e-12:

            rejected += 1

            continue

        if not 0.0 < salt_fraction < 1.0:

            rejected += 1

            continue



        salt_mmol = total_solvent_mmol * salt_fraction / (1.0 - salt_fraction)

        components_mmol: dict[str, float] = {salt_name: salt_mmol}

        for name in solvent_names:

            components_mmol[name] = total_solvent_mmol * share_map[name]



        for spec in additives:

            add_name = spec["name"]

            add_low, add_high = _parse_range(

                spec, "mass_fraction_range", (0.0, 0.05), f"additive {add_name}"

            )

            fraction = float(rng.uniform(add_low, add_high))

            if fraction <= 0:

                continue

            component = require_component(add_name)

            base_mass = sum(

                amount * require_component(name).molar_mass_g_mol / 1000.0

                for name, amount in components_mmol.items()

            )

            add_mass = base_mass * fraction / max(1e-9, 1.0 - fraction)

            components_mmol[add_name] = add_mass / component.molar_mass_g_mol * 1000.0



        ec = components_mmol.get("EC", 0.0)

        emc = components_mmol.get("EMC", 0.0)

        pc = components_mmol.get("PC", 0.0)

        total_solvent = sum(components_mmol.get(name, 0.0) for name in solvent_names)

        if (ec + emc) <= 0 or total_solvent <= 0:

            rejected += 1

            continue

        x_ec = ec / (ec + emc)

        x_pc = pc / total_solvent



        unsupported: list[str] = []

        if salt_name.upper() != MODEL_SUPPORTED_SALT.upper():

            unsupported.append(salt_name)

        unsupported.extend(

            name for name in solvent_names if name not in MODEL_SUPPORTED_SOLVENTS

        )

        unsupported.extend(

            spec["name"]

            for spec in additives

            if spec["name"] not in MODEL_SUPPORTED_SOLVENTS

        )

        model_supported = not unsupported



        outside_training = not (

            TRAINING_X_NASALT_RANGE[0] <= salt_fraction <= TRAINING_X_NASALT_RANGE[1]

        )



        mass_fraction = _mass_fraction(components_mmol)

        # 体积分数只对溶剂计算：盐的纯组分密度不适用于溶液体积份额，

        # 把盐算进去会给出无物理意义的数。

        solvent_amounts = {

            name: components_mmol[name] for name in solvent_names if name in components_mmol

        }

        volume_fraction, missing_density = _volume_fraction(solvent_amounts)



        candidates.append(

            {

                "candidate_id": f"gen-{len(candidates) + 1:04d}",

                "salt": salt_name,

                "salts": [salt_name],

                "solvents": solvent_names,

                # 字段名与 conductivity_inputs.resolve_conductivity_inputs 的契约一致

                # （component_mmol，单数），保证生成结果可直接送入预测工具。

                "component_mmol": {

                    name: round(amount, 6) for name, amount in components_mmol.items()

                },

                "solvent_molar_shares": {

                    name: round(value, 6) for name, value in share_map.items()

                },

                # 与 FormulationCandidate / CandidateState 的既有字段对齐，

                # 保证生成结果能进入设计状态而不丢失配比。

                "solvent_ratios": {

                    name: round(value, 6) for name, value in share_map.items()

                },

                "solvent_ratio_basis": "molar",

                "x_NaSalt": round(salt_fraction, 8),

                "x_EC": round(x_ec, 8),

                "x_PC": round(x_pc, 8),

                "mass_fraction": {

                    key: round(value, 6) for key, value in mass_fraction.items()

                },

                "solvent_volume_fraction": (

                    {key: round(value, 6) for key, value in volume_fraction.items()}

                    if volume_fraction

                    else None

                ),

                "solvent_volume_fraction_missing_density": missing_density,

                "model_supported": bool(model_supported),

                "unsupported_components": unsupported,

                "outside_training_range": outside_training,

                "sampling_method": method,

                "evidence": [],

                "evidence_level": "weak",

                "target_rationale": [

                    f"由组分空间采样生成（{method}, seed={seed}），非文献已报道配方；"

                    "使用前须经可行性检查与先例检索"

                ],

            }

        )



    if not candidates:

        raise CandidateSpaceError(

            "在给定范围与采样次数内没有生成任何候选。可能原因：溶剂份额上下界互斥，"

            f"或盐摩尔分数范围过窄。已拒绝 {rejected} 个样本。"

        )



    supported_count = sum(1 for item in candidates if item["model_supported"])

    outside_count = sum(1 for item in candidates if item["outside_training_range"])



    space: dict[str, Any] = {

        "method": method,

        "seed": seed,

        "sampled": len(candidates),

        "requested": n_samples,

        "rejected_samples": rejected,

        "enumerated_total": enumerated_total,

        "coverage": (

            round(len(candidates) / enumerated_total, 4) if enumerated_total else None

        ),

        "coverage_note": (

            "grid 方法下为 采样数/可枚举总数；lhs 与 random 的总点数未知，返回 null"

        ),

        "declared_bounds": {

            "salt": {salt_name: {"mol_fraction_range": [salt_low, salt_high]}},

            "solvent": {

                name: {"molar_share_range": list(solvent_bounds[name])}

                for name in solvent_names

            },

            "temperatures_c": parsed_temperatures,

            "total_solvent_mmol": total_solvent_mmol,

        },

        "training_reference": {

            "x_NaSalt_range": list(TRAINING_X_NASALT_RANGE),

            "temperature_range_c": list(TRAINING_TEMPERATURE_RANGE_C),

            "source": "647 行训练表实算（prepare_training_table，2026-10-19）",

        },

        "model_supported_count": supported_count,

        "model_supported_ratio": round(supported_count / len(candidates), 4),

        "outside_training_range_count": outside_count,

    }



    if method == "lhs":

        space["sampling_note"] = (

            "拉丁超立方先在超立方体分层，再指数化映射到单纯形；归一化后为近似 LHS"

        )

    space["viscosity_probe_first_candidate"] = arrhenius_blend_viscosity(

        candidates[0]["solvent_molar_shares"]

    )



    if supported_count == 0:

        warnings.append(

            "没有任何候选能被当前电导率模型评估：组分含模型未覆盖的盐或溶剂。"

            "生成的候选可用于规划扩充后的空间，但若要求数值必须先重训模型"

        )

    if rejected:

        warnings.append(f"有 {rejected} 个样本因超出上下界被拒绝")



    result = {"status": "ok", "candidates": candidates, "space": space, "warnings": warnings}

    if result_mode == "artifact":

        from .candidate_batches import save_candidate_batch

        return save_candidate_batch(result, request)

    return result





def _random_dirichlet_one(d: int, rng: np.random.Generator) -> tuple[float, ...]:

    row = rng.dirichlet(np.ones(d))[0] if d > 1 else np.asarray([1.0])

    return tuple(float(value) for value in np.atleast_1d(row))

