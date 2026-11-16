"""Robust multi-temperature ranking of electrolyte candidates under model uncertainty.



This module answers a different question from the single-point conductivity

predictor in :mod:`tools.electrolyte_design.conductivity_ml`.  There, one frozen

random forest returns one value plus a per-tree spread at one temperature.  Here,

a *group-bootstrap ensemble* of random forests returns a distribution over the

whole temperature set, so a designer can ask how often a candidate actually meets

a minimum-conductivity constraint at every required temperature.



Method boundary (mandatory reading for anyone quoting these numbers):



* The output is a model-robustness analysis.  It never upgrades a prediction into

  an experimental fact and it never overrides literature evidence.

* The ensemble is fitted on the frozen public bundle with the reserved holdout

  split bucket removed.  Candidates supplied by a caller are normally drawn from

  the same public measurement set, so this is an *in-sample* robustness analysis,

  not a held-out accuracy claim.  Each ranked candidate therefore reports

  ``bootstrap_inclusion_frequency``: the fraction of ensemble members whose

  bootstrap resample actually contained that formulation group.  Members that did

  not see the group still vote, which is what produces the spread.

* Resampling is done over *formulation groups*, never over individual rows, so a

  single formulation's nine temperature points cannot leak into its own bootstrap

  replicate more than once.

"""



from __future__ import annotations



import hashlib

import math

import os

from dataclasses import dataclass, field

from pathlib import Path

from threading import Lock

from typing import Any, Iterable, Mapping, Sequence



import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor



from .conductivity_ml import (

    FEATURE_COLUMNS,

    SUPPORTED_SALT,

    SUPPORTED_SOLVENTS,

    UPSTREAM_DATA_DOI,

    ConductivityModelError,

    ConductivityFeatures,

    features_from_mmol,

    prepare_training_table,

    temperature_c_to_inverse_temperature,

)





MODEL_ID = "na-conductivity-robust-ensemble-v1"

METHOD = "group_bootstrap_random_forest_ensemble"

SPLIT_MODULUS = 5

RESERVED_HOLDOUT_BUCKET = 4

DEFAULT_MEMBERS = 32

DEFAULT_TREES_PER_MEMBER = 96

DEFAULT_MIN_SAMPLES_LEAF = 2

DEFAULT_RANDOM_STATE = 20260924

DEFAULT_REJECT_PROBABILITY = 0.5

MIN_TEMPERATURES = 1

MAX_TEMPERATURES = 6

MAX_CANDIDATES = 12





class RobustRankingError(ConductivityModelError):

    """Raised when a ranking request cannot be answered at all."""





ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parents[4] / "data" / "na_training_bundle_v1"

_ENSEMBLE_CACHE: dict[tuple[str, int], RobustRankingEnsemble] = {}

_ENSEMBLE_LOCK = Lock()





def resolve_bundle_dir(explicit: str | Path | None = None) -> Path:

    """Resolve the frozen public bundle, honouring the process environment."""



    configured = str(explicit or "").strip() or os.environ.get(

        "ORION_ROBUST_RANKING_BUNDLE_DIR", ""

    ).strip()

    if configured:

        return Path(configured).expanduser().resolve()

    return DEFAULT_BUNDLE_DIR





def in_process_ensemble(

    bundle_dir: str | Path | None = None,

    *,

    members: int = DEFAULT_MEMBERS,

    trees_per_member: int = DEFAULT_TREES_PER_MEMBER,

) -> RobustRankingEnsemble:

    """Return a cached in-process ensemble for direct calls and tests.



    The Agent path must use the managed ``process:robust-ranking-worker`` instance

    instead; this helper exists so that direct function calls do not require a

    governor and so repeated calls in one process do not refit the ensemble.

    """



    resolved = resolve_bundle_dir(bundle_dir)

    key = (str(resolved), int(members))

    cached = _ENSEMBLE_CACHE.get(key)

    if cached is not None:

        return cached

    with _ENSEMBLE_LOCK:

        cached = _ENSEMBLE_CACHE.get(key)

        if cached is not None:

            return cached

        frames = development_frames(resolved)

        full_table = development_training_table(frames, drop_reserved_holdout=False)

        table = development_training_table(frames, drop_reserved_holdout=True)

        ensemble = fit_robust_ensemble(

            table,

            members=int(members),

            trees_per_member=int(trees_per_member),

        )

        ensemble.excluded_holdout_formulations = int(

            full_table["formulation_group"].nunique()

            - table["formulation_group"].nunique()

        )

        _ENSEMBLE_CACHE[key] = ensemble

        return ensemble





def split_bucket(formulation_group: str) -> int:

    """Return the frozen split bucket for a formulation identity."""



    digest = hashlib.sha256(str(formulation_group).encode("utf-8")).hexdigest()

    return int(digest[:16], 16) % SPLIT_MODULUS





def development_frames(bundle_dir: Path) -> list[pd.DataFrame]:

    """Load the three frozen public CSVs used by the conductivity pipeline."""



    paths = sorted(Path(bundle_dir).resolve().glob("hts_data_gen*.csv"))

    if len(paths) != 3:

        raise RobustRankingError(

            f"bundle_dir must contain exactly three hts_data_gen*.csv files: {bundle_dir}"

        )

    return [pd.read_csv(path) for path in paths]





def development_training_table(

    frames: Iterable[pd.DataFrame],

    *,

    drop_reserved_holdout: bool = True,

) -> pd.DataFrame:

    """Build the training table, optionally removing the reserved split bucket."""



    table = prepare_training_table(frames)

    if not drop_reserved_holdout:

        return table

    keep = table["formulation_group"].map(split_bucket) != RESERVED_HOLDOUT_BUCKET

    filtered = table.loc[keep].reset_index(drop=True)

    if filtered["formulation_group"].nunique() < 5:

        raise RobustRankingError("too few development formulations after split filtering")

    return filtered





@dataclass

class _Member:

    model: RandomForestRegressor

    groups: tuple[str, ...]





@dataclass

class RobustRankingEnsemble:

    """Group-bootstrap random-forest ensemble over the public Na conductivity table."""



    members: list[_Member]

    domain: dict[str, dict[str, float]]

    training_rows: int

    training_formulations: int

    excluded_holdout_formulations: int

    random_state: int = DEFAULT_RANDOM_STATE

    feature_columns: tuple[str, ...] = field(default=FEATURE_COLUMNS)



    @property

    def member_count(self) -> int:

        return len(self.members)



    def bootstrap_inclusion_frequency(self, formulation_group: str) -> float:

        """Fraction of members whose resample contained this formulation group."""



        if not formulation_group:

            return 0.0

        hits = sum(1 for member in self.members if formulation_group in member.groups)

        return round(hits / self.member_count, 6)



    def _matrix(self, features: Sequence[ConductivityFeatures]) -> np.ndarray:

        return np.asarray([item.as_array()[0] for item in features], dtype=float)



    def check_domain(self, features: ConductivityFeatures, *, tolerance: float = 1e-10) -> None:

        values = features.as_array()[0]

        outside: list[str] = []

        for name, value in zip(self.feature_columns, values):

            bounds = self.domain[name]

            lo, hi = float(bounds["min"]), float(bounds["max"])

            if value < lo - tolerance or value > hi + tolerance:

                outside.append(f"{name}={value:.6g} outside [{lo:.6g}, {hi:.6g}]")

        if outside:

            raise RobustRankingError("; ".join(outside))



    def member_log10_predictions(

        self,

        features: Sequence[ConductivityFeatures],

    ) -> np.ndarray:

        """Return a ``(member, point)`` matrix of predicted log10 conductivity."""



        if not features:

            raise RobustRankingError("no feature vectors supplied")

        matrix = self._matrix(features)

        rows = [member.model.predict(matrix) for member in self.members]

        return np.asarray(rows, dtype=float)





def _group_bootstrap_sample(

    table: pd.DataFrame,

    groups: Sequence[str],

    rng: np.random.Generator,

) -> tuple[pd.DataFrame, tuple[str, ...]]:

    drawn = rng.choice(len(groups), size=len(groups), replace=True)

    chosen = tuple(str(groups[index]) for index in drawn)

    frames = [

        table.loc[table["formulation_group"] == group]

        for group in chosen

    ]

    if not frames:

        raise RobustRankingError("bootstrap resample produced no rows")

    return pd.concat(frames, ignore_index=True), chosen





def fit_robust_ensemble(

    table: pd.DataFrame,

    *,

    members: int = DEFAULT_MEMBERS,

    trees_per_member: int = DEFAULT_TREES_PER_MEMBER,

    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,

    random_state: int = DEFAULT_RANDOM_STATE,

) -> RobustRankingEnsemble:

    """Fit the group-bootstrap ensemble; deterministic for a fixed seed."""



    if members < 2:

        raise RobustRankingError("a robustness ensemble needs at least two members")

    if table["formulation_group"].nunique() < 3:

        raise RobustRankingError("ensemble fitting needs at least three formulations")

    groups = tuple(sorted(str(item) for item in table["formulation_group"].unique()))

    rng = np.random.default_rng(int(random_state))

    x = table.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)

    fitted_members: list[_Member] = []

    for index in range(int(members)):

        sample, chosen = _group_bootstrap_sample(table, groups, rng)

        model = RandomForestRegressor(

            n_estimators=int(trees_per_member),

            min_samples_leaf=int(min_samples_leaf),

            max_depth=None,

            random_state=int(random_state) + index * 7919,

            n_jobs=1,

            bootstrap=True,

        )

        model.fit(

            sample.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float),

            sample["log10_conductivity"].to_numpy(dtype=float),

        )

        fitted_members.append(_Member(model=model, groups=chosen))

    domain = {

        name: {"min": float(x[:, position].min()), "max": float(x[:, position].max())}

        for position, name in enumerate(FEATURE_COLUMNS)

    }

    return RobustRankingEnsemble(

        members=fitted_members,

        domain=domain,

        training_rows=int(len(table)),

        training_formulations=len(groups),

        excluded_holdout_formulations=0,

        random_state=int(random_state),

    )





def _candidate_feature(

    candidate: Mapping[str, Any],

    *,

    temperatures_c: Sequence[float],

    override: Mapping[str, Any] | None,

) -> list[ConductivityFeatures]:

    """与点预测共用组成解析，核对完整物质的量与显式特征的一致性。"""

    from .conductivity_inputs import resolve_conductivity_inputs



    features: list[ConductivityFeatures] = []

    for temperature_c in temperatures_c:

        try:

            feature = resolve_conductivity_inputs(

                dict(candidate), temperature_c=temperature_c,

                override=dict(override) if override is not None else None,

            )

        except (TypeError, ValueError) as error:

            raise RobustRankingError(str(error)) from error

        if feature is None:

            raise RobustRankingError(

                "exact molar composition is unavailable; supply x_NaSalt/x_EC/x_PC, "

                "component_mmol or molar_ratios"

            )

        features.append(feature)

    return features





def _domain_verdict(candidate: Mapping[str, Any]) -> str:

    salt = str(candidate.get("salt") or "").replace(" ", "")

    if salt.casefold() != SUPPORTED_SALT.casefold():

        return f"model supports {SUPPORTED_SALT}; candidate salt is {salt or 'unknown'}"

    solvents = {

        str(item).strip().upper()

        for item in candidate.get("solvents") or []

        if str(item).strip()

    }

    if not solvents or not solvents <= set(SUPPORTED_SOLVENTS):

        return (

            "model supports only EC/EMC/PC solvent compositions; candidate solvents="

            + ",".join(sorted(solvents))

        )

    if [str(item).strip() for item in candidate.get("additives") or [] if str(item).strip()]:

        return "additive-containing formulations are outside the frozen model domain"

    return ""





def _median_log_gap(values: np.ndarray, mask: np.ndarray) -> float | None:

    """log10 median ratio between members that saw the group and members that did not."""



    seen = values[mask]

    unseen = values[~mask]

    if seen.size < 3 or unseen.size < 3:

        return None

    if not (bool((seen > 0).all()) and bool((unseen > 0).all())):

        return None

    return float(math.log10(float(np.median(seen)) / float(np.median(unseen))))





def _pareto_layers(rows: list[dict[str, Any]]) -> list[list[str]]:

    """Layer ranked candidates on (joint probability, worst-temperature median)."""



    remaining = list(rows)

    layers: list[list[str]] = []

    while remaining:

        layer: list[dict[str, Any]] = []

        for row in remaining:

            dominated = False

            for other in remaining:

                if other is row:

                    continue

                at_least = (

                    other["joint_meets_constraint_probability"]

                    >= row["joint_meets_constraint_probability"]

                    and other["worst_temperature_median_ms_cm"]

                    >= row["worst_temperature_median_ms_cm"]

                    and (

                        other["joint_meets_constraint_probability"]

                        > row["joint_meets_constraint_probability"]

                        or other["worst_temperature_median_ms_cm"]

                        > row["worst_temperature_median_ms_cm"]

                    )

                )

                if at_least:

                    dominated = True

                    break

            if not dominated:

                layer.append(row)

        layer_ids = sorted(

            str(row["candidate_id"])

            for row in layer

        )

        layers.append(layer_ids)

        layer_set = set(layer_ids)

        remaining = [row for row in remaining if str(row["candidate_id"]) not in layer_set]

    return layers





def _spearman(left: Sequence[float], right: Sequence[float]) -> float:

    if len(left) != len(right) or len(left) < 2:

        return float("nan")

    a = np.asarray(left, dtype=float)

    b = np.asarray(right, dtype=float)

    a = np.argsort(np.argsort(a))

    b = np.argsort(np.argsort(b))

    a = a - a.mean()

    b = b - b.mean()

    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))

    if denominator == 0.0:

        return float("nan")

    return float((a * b).sum() / denominator)





def _attach_rank_stability(rows: list[dict[str, Any]]) -> None:

    """Rank the *candidates* inside each ensemble member and summarise the spread.



    The earlier formulation ranked ensemble members for a single candidate, which

    measures nothing about candidate choice.  Here each member ranks the whole

    candidate set by its worst-temperature conductivity, and a candidate's rank

    stability is the distribution of that candidate's rank across members.

    """



    if not rows:

        return

    if len(rows) == 1:

        rows[0]["rank_stability"] = {

            "mean_rank": 1.0,

            "rank_std": 0.0,

            "top1_frequency": 1.0,

            "mean_spearman_vs_ensemble_median": None,

            "member_count": int(rows[0]["_per_member_min"].size),

        }

        return

    scores = np.vstack([row["_per_member_min"] for row in rows]).T

    members, candidate_count = scores.shape

    order = np.argsort(-scores, axis=1, kind="stable")

    rank_matrix = np.empty_like(order, dtype=float)

    for member in range(members):

        rank_matrix[member, order[member]] = np.arange(1, candidate_count + 1)

    reference = np.asarray(

        [float(row["worst_temperature_median_ms_cm"]) for row in rows], dtype=float

    )

    agreements = [

        _spearman(scores[member, :], reference) for member in range(members)

    ]

    agreements = [value for value in agreements if not math.isnan(value)]

    for position, row in enumerate(rows):

        ranks = rank_matrix[:, position]

        row["rank_stability"] = {

            "mean_rank": round(float(ranks.mean()), 6),

            "rank_std": round(float(ranks.std(ddof=0)), 6),

            "top1_frequency": round(float((ranks == 1.0).mean()), 6),

            "mean_spearman_vs_ensemble_median": (

                round(float(sum(agreements) / len(agreements)), 6) if agreements else None

            ),

            "member_count": members,

        }





def validate_ranking_request(request: Mapping[str, Any]) -> tuple[list[dict], list[float], float]:

    """不加载模型，先核对工具请求；不为任务补造最低阈值。"""



    if not isinstance(request, Mapping):

        raise RobustRankingError("request must be an object")

    raw_candidates = request.get("candidates")

    if not isinstance(raw_candidates, (list, tuple)) or not raw_candidates:

        raise RobustRankingError("request.candidates must be a non-empty list")

    candidates = [dict(item) for item in raw_candidates if isinstance(item, Mapping)]

    if len(candidates) != len(raw_candidates):

        raise RobustRankingError("every candidate must be an object")

    if len(candidates) > MAX_CANDIDATES:

        raise RobustRankingError(f"at most {MAX_CANDIDATES} candidates per request")



    temperatures = request.get("temperature_c")

    if isinstance(temperatures, (int, float)):

        temperatures = [float(temperatures)]

    if not isinstance(temperatures, (list, tuple)) or not temperatures:

        raise RobustRankingError("request.temperature_c must be a number or a non-empty list")

    try:

        temperatures_c = [float(item) for item in temperatures]

    except (TypeError, ValueError) as exc:

        raise RobustRankingError("temperature_c entries must be numeric") from exc

    if not MIN_TEMPERATURES <= len(temperatures_c) <= MAX_TEMPERATURES:

        raise RobustRankingError(

            f"request between {MIN_TEMPERATURES} and {MAX_TEMPERATURES} temperatures"

        )

    if len(set(temperatures_c)) != len(temperatures_c):

        raise RobustRankingError("temperature_c entries must be distinct")

    if any(not math.isfinite(item) for item in temperatures_c):

        raise RobustRankingError("temperature_c entries must be finite")



    threshold = request.get("min_conductivity_ms_cm")

    if threshold is None:

        raise RobustRankingError("request.min_conductivity_ms_cm is required")

    try:

        threshold = float(threshold)

    except (TypeError, ValueError) as exc:

        raise RobustRankingError("min_conductivity_ms_cm must be numeric") from exc

    if not math.isfinite(threshold) or threshold <= 0.0:

        raise RobustRankingError("min_conductivity_ms_cm must be a positive finite number")



    return candidates, temperatures_c, threshold





def rank_formulations(

    request: Mapping[str, Any],

    *,

    ensemble: RobustRankingEnsemble,

    member_budget: int | None = None,

    reject_probability: float = DEFAULT_REJECT_PROBABILITY,

) -> dict[str, Any]:

    """Rank candidates by joint multi-temperature constraint satisfaction."""



    candidates, temperatures_c, threshold = validate_ranking_request(request)



    requested_budget = request.get("ensemble_budget", member_budget)

    try:

        budget = ensemble.member_count if requested_budget is None else int(requested_budget)

    except (TypeError, ValueError) as exc:

        raise RobustRankingError("ensemble_budget must be an integer") from exc

    if budget < 2 or budget > ensemble.member_count:

        raise RobustRankingError(

            f"ensemble_budget must be between 2 and the fitted {ensemble.member_count} members"

        )

    active = ensemble.members[:budget]

    used_ensemble = RobustRankingEnsemble(

        members=active,

        domain=ensemble.domain,

        training_rows=ensemble.training_rows,

        training_formulations=ensemble.training_formulations,

        excluded_holdout_formulations=ensemble.excluded_holdout_formulations,

        random_state=ensemble.random_state,

    )



    overrides = request.get("composition_inputs") or {}

    overrides = overrides if isinstance(overrides, Mapping) else {}



    scored: list[dict[str, Any]] = []

    unresolved: list[dict[str, Any]] = []

    prepared = []

    for candidate in candidates:

        candidate_id = str(candidate.get("candidate_id") or "").strip() or f"candidate-{len(prepared)+1}"

        reason = _domain_verdict(candidate)

        if reason:

            unresolved.append(

                {

                    "candidate_id": candidate_id,

                    "status": "out_of_domain",

                    "rank": None,

                    "rejection_reasons": [reason],

                }

            )

            continue

        try:

            features = _candidate_feature(

                candidate,

                temperatures_c=temperatures_c,

                override=(

                    overrides.get(candidate_id)

                    if isinstance(overrides.get(candidate_id), Mapping)

                    else None

                ),

            )

        except (RobustRankingError, TypeError, ValueError) as exc:

            unresolved.append(

                {

                    "candidate_id": candidate_id,

                    "status": "insufficient_composition",

                    "rank": None,

                    "rejection_reasons": [str(exc)],

                }

            )

            continue

        for feature in features:

            try:

                used_ensemble.check_domain(feature)

            except RobustRankingError as exc:

                unresolved.append(

                    {

                        "candidate_id": candidate_id,

                        "status": "out_of_domain",

                        "rank": None,

                        "rejection_reasons": [str(exc)],

                    }

                )

                break

        else:

            prepared.append((candidate, candidate_id, features))



    # 同一请求一次计算全部有效候选；整体与逐成员执行使用相同路径。

    # 仅改变样本的预测批量，不改变成员顺序、汇总统计或域外处理。

    all_features = [feature for _, _, features in prepared for feature in features]

    predictions = (used_ensemble.member_log10_predictions(all_features)

                   if all_features else None)

    offset = 0

    for candidate, candidate_id, features in prepared:

        matrix = predictions[:, offset:offset + len(features)]

        offset += len(features)



        conductivity = 10.0 ** matrix * 1000.0

        meets = conductivity >= threshold

        per_temperature = [

            {

                "temperature_c": round(temperatures_c[position], 6),

                "median_ms_cm": float(np.median(conductivity[:, position])),

                "p05_ms_cm": float(np.percentile(conductivity[:, position], 5)),

                "p95_ms_cm": float(np.percentile(conductivity[:, position], 95)),

                "ensemble_log10_std": float(np.std(matrix[:, position], ddof=0)),

                "meets_constraint_probability": float(meets[:, position].mean()),

            }

            for position in range(len(temperatures_c))

        ]

        joint = float(meets.all(axis=1).mean())

        worst_position = int(np.argmin([row["median_ms_cm"] for row in per_temperature]))

        group = str(candidate.get("formulation_group") or "")

        group_members = [member for member in used_ensemble.members]

        if group:

            seen_mask = np.asarray(

                [group in member.groups for member in group_members], dtype=bool

            )

        else:

            seen_mask = np.zeros(len(group_members), dtype=bool)

        worst_conductivity = conductivity[:, worst_position]

        unseen_joint: float | None = None

        if group and int((~seen_mask).sum()) >= 3:

            unseen_joint = float(meets[~seen_mask].all(axis=1).mean())

        memorization_gap = (

            _median_log_gap(worst_conductivity, seen_mask) if group else None

        )

        per_member_min = conductivity.min(axis=1)

        scored.append(

            {

                "candidate_id": candidate_id,

                "status": "ranked",

                "per_temperature": per_temperature,

                "joint_meets_constraint_probability": joint,

                "joint_meets_constraint_probability_unseen_members": unseen_joint,

                "worst_temperature_c": round(temperatures_c[worst_position], 6),

                "worst_temperature_median_ms_cm": per_temperature[worst_position][

                    "median_ms_cm"

                ],

                "pareto_layer": None,

                "dominated_by": [],

                "rank_stability": None,

                "bootstrap_inclusion_frequency": (

                    round(float(seen_mask.mean()), 6) if group else None

                ),

                "in_sample_group_memorization_gap_log10": memorization_gap,

                "_per_member_min": per_member_min,

            }

        )



    _attach_rank_stability(scored)

    scored.sort(

        key=lambda row: (

            -row["joint_meets_constraint_probability"],

            -row["worst_temperature_median_ms_cm"],

            row["candidate_id"],

        )

    )

    layers = _pareto_layers(scored)

    layer_of: dict[str, int] = {}

    for index, layer in enumerate(layers, start=1):

        for candidate_id in layer:

            layer_of[candidate_id] = index

    for position, row in enumerate(scored, start=1):

        row["rank"] = position

        row["pareto_layer"] = layer_of[str(row["candidate_id"])]

        row["dominated_by"] = sorted(

            str(other["candidate_id"])

            for other in scored

            if layer_of[str(other["candidate_id"])] < layer_of[str(row["candidate_id"])]

        )

        stability = row["rank_stability"]

        rejection_reasons: list[str] = []

        if row["joint_meets_constraint_probability"] < reject_probability:

            rejection_reasons.append(

                "joint constraint probability "

                f"{row['joint_meets_constraint_probability']:.3f} < {reject_probability:.2f}"

            )

        for item in row["per_temperature"]:

            if item["meets_constraint_probability"] < reject_probability:

                rejection_reasons.append(

                    f"constraint at {item['temperature_c']:g} °C met in only "

                    f"{item['meets_constraint_probability']:.3f} of ensemble members"

                )

        if row["dominated_by"]:

            rejection_reasons.append(

                "Pareto-dominated by " + ", ".join(row["dominated_by"])

            )

        caveats: list[str] = []

        if stability["rank_std"] > 1.0:

            caveats.append(

                f"rank unstable across members (rank std {stability['rank_std']:.3f})"

            )

        unseen = row["joint_meets_constraint_probability_unseen_members"]

        if (

            unseen is not None

            and row["joint_meets_constraint_probability"] >= reject_probability

            and unseen < reject_probability

        ):

            caveats.append(

                "apparent pass driven by members fitted on this formulation's own "

                f"measurements; members without it reach the constraint only {unseen:.3f} "

                "of the time"

            )

        gap = row["in_sample_group_memorization_gap_log10"]

        if gap is not None and gap > 0.3:

            caveats.append(

                f"members that saw this formulation predict {10.0 ** gap:.2f}x higher at the "

                "worst temperature than members that did not"

            )

        row["rejection_reasons"] = rejection_reasons

        row["caveats"] = caveats

        row["decision"] = "rejected" if rejection_reasons else "recommended"

        row.pop("_per_member_min", None)



    ranked = list(scored)

    ordered = [str(row["candidate_id"]) for row in ranked]

    return {

        "tool": "rank_formulations_under_uncertainty",

        "model_id": MODEL_ID,

        "method": METHOD,

        "evidence_status": "model_based_robustness_analysis_not_experimental_fact",

        "source_data_doi": UPSTREAM_DATA_DOI,

        "temperature_c": [round(item, 6) for item in temperatures_c],

        "min_conductivity_ms_cm": threshold,

        "reject_probability": float(reject_probability),

        "ensemble_members_used": used_ensemble.member_count,

        "ensemble_members_fitted": ensemble.member_count,

        "training_rows": ensemble.training_rows,

        "training_formulations": ensemble.training_formulations,

        "candidates": ranked,

        "unranked_candidates": unresolved,

        "ranking": ordered,

        "pareto_layers": layers,

        "notes": [

            "Probabilities are ensemble frequencies over group-bootstrap members, not "

            "calibrated frequentist coverage.",

            "Candidates drawn from the same public measurement set the ensemble was "

            "fitted on make this an in-sample robustness analysis; read "

            "bootstrap_inclusion_frequency per candidate.",

            "Model robustness does not replace literature evidence or experimental fact.",

        ],

    }





__all__ = [

    "DEFAULT_BUNDLE_DIR",

    "DEFAULT_MEMBERS",

    "DEFAULT_REJECT_PROBABILITY",

    "METHOD",

    "MODEL_ID",

    "RESERVED_HOLDOUT_BUCKET",

    "RobustRankingEnsemble",

    "RobustRankingError",

    "development_frames",

    "development_training_table",

    "fit_robust_ensemble",

    "in_process_ensemble",

    "rank_formulations",

    "resolve_bundle_dir",

    "split_bucket",

]

