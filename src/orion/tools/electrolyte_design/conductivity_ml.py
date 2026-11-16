"""Lightweight ionic-conductivity model for a bounded NaPF6/EC-EMC-PC domain.

The feature construction follows the public sodium-electrolyte preprocessing used by
Fischer et al. (Digital Discovery, 2026; data DOI 10.5281/zenodo.18429013):

    inverse_temperature = 1000 / T[K]
    x_NaSalt = n(NaPF6) / sum(n(NaPF6), n(EC), n(EMC), n(PC))
    x_EC = n(EC) / (n(EC) + n(EMC))
    x_PC = n(PC) / (n(EC) + n(EMC) + n(PC))

This module deliberately does not claim transfer to other salts, ether electrolytes, or
additive-containing formulations.  Out-of-domain requests are rejected rather than
silently extrapolated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import math
from pathlib import Path
from typing import Iterable, Sequence
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


UPSTREAM_DATA_DOI = "10.5281/zenodo.18429013"
UPSTREAM_DATA_URL = "https://zenodo.org/records/18429013/files/data.zip?download=1"
UPSTREAM_DATA_LICENSE = "MIT"
UPSTREAM_PAPER_DOI = "10.1039/D6DD00149A"
SUPPORTED_SALT = "NaPF6"
SUPPORTED_SOLVENTS = ("EC", "EMC", "PC")
FEATURE_COLUMNS = ("inverse temperature", "x_NaSalt", "x_EC", "x_PC")
RAW_REQUIRED_COLUMNS = (
    "NaPF6_mmol",
    "EC_mmol",
    "EMC_mmol",
    "PC_mmol",
    "inverse temperature",
    "conductivity",
)


class ConductivityModelError(RuntimeError):
    """Base error for the bounded conductivity model."""


class OutOfDomainError(ConductivityModelError):
    """Raised when a prediction request lies outside the fitted training domain.

    ``violations`` carries one record per offending feature so callers can report
    which physical variable is out and by how much instead of parsing the message.
    """

    def __init__(self, message: str, violations: list[dict[str, object]] | None = None) -> None:
        super().__init__(message)
        self.violations: list[dict[str, object]] = violations or []


# 硬容差（1e-10）会把「训练集里已有 x_PC=0.0015、请求 x_PC=0」这类组分单纯形边界
# 判成外推，而 0.0015 mol 分数实际等价于 PC-free。这里改为按特征量程给相对容差：
# 允许最多越过训练边界 1% 的量程，含义是「低到观测下限附近、组分可视为缺失」。
DOMAIN_TOLERANCE_FRACTION = 0.01


@dataclass(frozen=True)
class ConductivityFeatures:
    inverse_temperature: float
    x_NaSalt: float
    x_EC: float
    x_PC: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [[self.inverse_temperature, self.x_NaSalt, self.x_EC, self.x_PC]],
            dtype=float,
        )


@dataclass(frozen=True)
class ConductivityPrediction:
    conductivity_s_cm: float
    conductivity_ms_cm: float
    log10_conductivity_s_cm: float
    ensemble_std_log10: float
    estimated_factor_1sigma: float
    in_domain: bool
    features: ConductivityFeatures

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["features"] = asdict(self.features)
        return payload


@dataclass(frozen=True)
class GroupedCvMetrics:
    folds: int
    rows: int
    unique_formulations: int
    rmse_log10: float
    mae_log10: float
    r2_log10: float
    multiplicative_rmse_factor: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def temperature_c_to_inverse_temperature(temperature_c: float) -> float:
    kelvin = float(temperature_c) + 273.15
    if not math.isfinite(kelvin) or kelvin <= 0:
        raise ValueError("temperature must be above absolute zero")
    return 1000.0 / kelvin


def features_from_mmol(
    *,
    temperature_c: float,
    napf6_mmol: float,
    ec_mmol: float,
    emc_mmol: float,
    pc_mmol: float,
) -> ConductivityFeatures:
    amounts = np.asarray([napf6_mmol, ec_mmol, emc_mmol, pc_mmol], dtype=float)
    if not np.all(np.isfinite(amounts)) or np.any(amounts < 0):
        raise ValueError("component amounts must be finite and non-negative")
    if napf6_mmol <= 0:
        raise ValueError("NaPF6 amount must be positive")
    solvent_total = float(ec_mmol + emc_mmol + pc_mmol)
    ec_emc_total = float(ec_mmol + emc_mmol)
    total = float(amounts.sum())
    if solvent_total <= 0 or ec_emc_total <= 0 or total <= 0:
        raise ValueError("EC/EMC solvent content must be positive for this model domain")
    return ConductivityFeatures(
        inverse_temperature=temperature_c_to_inverse_temperature(temperature_c),
        x_NaSalt=float(napf6_mmol / total),
        x_EC=float(ec_mmol / ec_emc_total),
        x_PC=float(pc_mmol / solvent_total),
    )


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(raw))


def raw_na_frames_from_zip_bytes(payload: bytes) -> list[pd.DataFrame]:
    """Extract the three public sodium raw generations from the upstream ZIP bytes."""

    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(BytesIO(payload)) as zf:
        candidates = [
            name
            for name in zf.namelist()
            if "na/raw/hts_data_gen" in name.replace("\\", "/").lower()
            and not name.endswith("/")
        ]
        if not candidates:
            raise ConductivityModelError(
                "upstream ZIP does not contain Na/raw/hts_data_gen* files"
            )
        for name in sorted(candidates):
            raw = zf.read(name)
            try:
                frame = _read_csv_bytes(raw)
            except Exception as exc:  # pragma: no cover - upstream format guard
                raise ConductivityModelError(f"cannot parse upstream CSV {name!r}") from exc
            missing = sorted(set(RAW_REQUIRED_COLUMNS) - set(frame.columns))
            if missing:
                raise ConductivityModelError(
                    f"upstream CSV {name!r} is missing columns: {', '.join(missing)}"
                )
            frame = frame.copy()
            frame["_upstream_member"] = name
            frames.append(frame)
    return frames


def _download_bytes_resumable(
    url: str,
    *,
    timeout_s: float,
    max_attempts: int = 6,
    chunk_size: int = 1024 * 1024,
) -> bytes:
    """Download a large public artifact with in-memory HTTP Range resume.

    Zenodo occasionally closes long transfers early on unstable networks.  Retrying from
    byte zero would make the 138 MB source unnecessarily fragile, so this helper resumes
    from the number of bytes already received whenever the server honors Range requests.
    No persistent cache is created.
    """

    payload = bytearray()
    expected_total: int | None = None
    session = requests.Session()
    try:
        for attempt in range(1, max_attempts + 1):
            headers = {
                "User-Agent": "ORION-CWSR-research/1.0",
                "Accept-Encoding": "identity",
            }
            if payload:
                headers["Range"] = f"bytes={len(payload)}-"
            try:
                with session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(20.0, float(timeout_s)),
                ) as response:
                    if payload and response.status_code == 200:
                        # Server ignored Range; restart from the complete response body.
                        payload.clear()
                        expected_total = None
                    elif response.status_code not in {200, 206}:
                        response.raise_for_status()

                    if response.status_code == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if "/" in content_range:
                            total_text = content_range.rsplit("/", 1)[-1]
                            if total_text.isdigit():
                                expected_total = int(total_text)
                    elif expected_total is None:
                        content_length = response.headers.get("Content-Length", "")
                        if content_length.isdigit():
                            expected_total = int(content_length)

                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            payload.extend(chunk)
            except requests.RequestException:
                if attempt >= max_attempts:
                    raise
                continue

            if expected_total is None:
                if payload:
                    return bytes(payload)
            elif len(payload) >= expected_total:
                if len(payload) != expected_total:
                    raise ConductivityModelError(
                        f"download size mismatch: received={len(payload)}, expected={expected_total}"
                    )
                return bytes(payload)

        raise ConductivityModelError(
            f"incomplete download after {max_attempts} attempts: "
            f"received={len(payload)}, expected={expected_total}"
        )
    finally:
        session.close()


def download_upstream_na_frames(
    *,
    url: str = UPSTREAM_DATA_URL,
    timeout_s: float = 90.0,
) -> list[pd.DataFrame]:
    """Read only the three Na CSV members from the public ZIP when Range is available.

    The Zenodo archive is about 138 MB while the required Na raw CSVs total only about
    0.3 MB uncompressed.  ``fsspec`` exposes a seekable HTTP file backed by Range
    requests, allowing ``zipfile`` to fetch the central directory and those members
    without downloading the entire archive.  The existing resumable full-download path
    remains a compatibility fallback for servers/environments without usable Range I/O.
    """

    try:
        import fsspec

        with fsspec.open(
            url,
            mode="rb",
            block_size=1024 * 1024,
            cache_type="readahead",
            timeout=float(timeout_s),
        ).open() as remote:
            with zipfile.ZipFile(remote) as zf:
                names = [
                    name
                    for name in zf.namelist()
                    if "na/raw/hts_data_gen" in name.replace("\\", "/").lower()
                    and not name.endswith("/")
                ]
                frames: list[pd.DataFrame] = []
                for name in sorted(names):
                    frame = _read_csv_bytes(zf.read(name))
                    missing = sorted(set(RAW_REQUIRED_COLUMNS) - set(frame.columns))
                    if missing:
                        raise ConductivityModelError(
                            f"upstream CSV {name!r} is missing columns: {', '.join(missing)}"
                        )
                    frame = frame.copy()
                    frame["_upstream_member"] = name
                    frames.append(frame)
                if frames:
                    return frames
    except Exception:
        # Fallback deliberately keeps the previously validated resumable download.
        pass

    payload = _download_bytes_resumable(url, timeout_s=timeout_s)
    return raw_na_frames_from_zip_bytes(payload)


def load_upstream_na_frames(path: str | Path) -> list[pd.DataFrame]:
    payload = Path(path).read_bytes()
    return raw_na_frames_from_zip_bytes(payload)


def prepare_training_table(
    frames: Iterable[pd.DataFrame],
    *,
    max_inverse_temperature: float = 4.0,
    aggregate_replicates: bool = True,
) -> pd.DataFrame:
    """Build the bounded four-feature regression table from raw public measurements.

    The upstream sodium notebook removes the -30 C region using inverse temperature > 4.
    We apply the same coarse temperature boundary here.  Repeated measurements at the same
    formulation/temperature are collapsed by median log-conductivity to reduce replicate
    leakage and measurement noise before grouped cross-validation.
    """

    source = pd.concat([frame.copy() for frame in frames], ignore_index=True)
    missing = sorted(set(RAW_REQUIRED_COLUMNS) - set(source.columns))
    if missing:
        raise ConductivityModelError(f"training data missing columns: {', '.join(missing)}")

    numeric = source.loc[:, RAW_REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    numeric = numeric[
        (numeric["conductivity"] > 0)
        & (numeric["inverse temperature"] <= float(max_inverse_temperature))
        & (numeric[["NaPF6_mmol", "EC_mmol", "EMC_mmol", "PC_mmol"]] >= 0).all(axis=1)
    ].copy()

    total = numeric[["NaPF6_mmol", "EC_mmol", "EMC_mmol", "PC_mmol"]].sum(axis=1)
    solvent_total = numeric[["EC_mmol", "EMC_mmol", "PC_mmol"]].sum(axis=1)
    ec_emc_total = numeric[["EC_mmol", "EMC_mmol"]].sum(axis=1)
    valid = (total > 0) & (solvent_total > 0) & (ec_emc_total > 0) & (numeric["NaPF6_mmol"] > 0)
    numeric = numeric.loc[valid].copy()
    total = total.loc[valid]
    solvent_total = solvent_total.loc[valid]
    ec_emc_total = ec_emc_total.loc[valid]

    numeric["x_NaSalt"] = numeric["NaPF6_mmol"] / total
    numeric["x_EC"] = numeric["EC_mmol"] / ec_emc_total
    numeric["x_PC"] = numeric["PC_mmol"] / solvent_total
    numeric["log10_conductivity"] = np.log10(numeric["conductivity"])

    # Composition identity is independent of temperature. Rounding only absorbs tiny
    # floating-point differences introduced by repeated liquid-handling records.
    numeric["formulation_group"] = (
        numeric[["x_NaSalt", "x_EC", "x_PC"]]
        .round(8)
        .astype(str)
        .agg("|".join, axis=1)
    )

    if aggregate_replicates:
        group_columns = ["formulation_group", *FEATURE_COLUMNS]
        numeric = (
            numeric.groupby(group_columns, as_index=False, sort=True)
            .agg(
                log10_conductivity=("log10_conductivity", "median"),
                replicate_count=("conductivity", "size"),
            )
        )
    else:
        numeric["replicate_count"] = 1

    if len(numeric) < 20 or numeric["formulation_group"].nunique() < 5:
        raise ConductivityModelError("too little in-domain training data after cleaning")
    return numeric.reset_index(drop=True)


class SodiumConductivityRandomForest:
    """Small CPU-only random forest for the public NaPF6/EC-EMC-PC domain."""

    def __init__(
        self,
        *,
        n_estimators: int = 192,
        min_samples_leaf: int = 2,
        max_depth: int | None = None,
        random_state: int = 20260906,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            min_samples_leaf=int(min_samples_leaf),
            max_depth=max_depth,
            random_state=int(random_state),
            n_jobs=1,
            bootstrap=True,
        )
        self.random_state = int(random_state)
        self._domain_min: np.ndarray | None = None
        self._domain_max: np.ndarray | None = None
        self.training_rows = 0
        self.training_formulations = 0

    @property
    def fitted(self) -> bool:
        return self._domain_min is not None and self._domain_max is not None

    def fit(self, table: pd.DataFrame) -> "SodiumConductivityRandomForest":
        x = table.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        y = table["log10_conductivity"].to_numpy(dtype=float)
        self.model.fit(x, y)
        self._domain_min = np.min(x, axis=0)
        self._domain_max = np.max(x, axis=0)
        self.training_rows = int(len(table))
        self.training_formulations = int(table["formulation_group"].nunique())
        return self

    def domain(self) -> dict[str, dict[str, float]]:
        if not self.fitted:
            raise ConductivityModelError("model is not fitted")
        assert self._domain_min is not None and self._domain_max is not None
        return {
            name: {"min": float(lo), "max": float(hi)}
            for name, lo, hi in zip(FEATURE_COLUMNS, self._domain_min, self._domain_max)
        }

    def describe_domain(self) -> dict[str, object]:
        """Translate the fitted feature box into physical units for capability reporting."""

        if not self.fitted:
            raise ConductivityModelError("model is not fitted")
        box = self.domain()
        inv_lo = box["inverse temperature"]["min"]
        inv_hi = box["inverse temperature"]["max"]
        return {
            "salt": SUPPORTED_SALT,
            "solvents": list(SUPPORTED_SOLVENTS),
            "temperature_c": {
                "min": round(1000.0 / inv_hi - 273.15, 2),
                "max": round(1000.0 / inv_lo - 273.15, 2),
            },
            "salt_mole_fraction": box["x_NaSalt"],
            "ec_share_of_ec_emc": box["x_EC"],
            "pc_share_of_solvent": box["x_PC"],
            "additives_supported": False,
            "training_rows": int(self.training_rows),
            "training_formulations": int(self.training_formulations),
            "tolerance_fraction_of_span": DOMAIN_TOLERANCE_FRACTION,
        }

    def domain_violations(self, features: ConductivityFeatures) -> list[dict[str, object]]:
        """Return the violations for ``features`` without raising (empty list if in domain)."""

        try:
            self._check_domain(features)
        except OutOfDomainError as exc:
            return list(exc.violations)
        return []

    def _check_domain(self, features: ConductivityFeatures, *, tolerance: float | None = None) -> None:
        if not self.fitted:
            raise ConductivityModelError("model is not fitted")
        assert self._domain_min is not None and self._domain_max is not None
        values = features.as_array()[0]
        span = self._domain_max - self._domain_min
        violations: list[dict[str, object]] = []
        for name, value, lo, hi, width in zip(
            FEATURE_COLUMNS, values, self._domain_min, self._domain_max, span
        ):
            band = tolerance if tolerance is not None else DOMAIN_TOLERANCE_FRACTION * float(width)
            band = max(float(band), 1e-10)
            if value < lo - band or value > hi + band:
                side = "below" if value < lo else "above"
                edge = float(lo) if value < lo else float(hi)
                violations.append(
                    {
                        "feature": name,
                        "value": float(value),
                        "training_min": float(lo),
                        "training_max": float(hi),
                        "side": side,
                        "outside_by": float(abs(value - edge)),
                        "tolerance_band": band,
                        "fold_outside_span": float(abs(value - edge) / width) if width > 0 else float("inf"),
                    }
                )
        if violations:
            parts = [
                f"{v['feature']}={v['value']:.6g} {v['side']} [{v['training_min']:.6g}, "
                f"{v['training_max']:.6g}] by {v['outside_by']:.3g}"
                for v in violations
            ]
            raise OutOfDomainError("; ".join(parts), violations)

    def predict_features(self, features: ConductivityFeatures) -> ConductivityPrediction:
        self._check_domain(features)
        x = features.as_array()
        log_pred = float(self.model.predict(x)[0])
        per_tree = np.asarray([tree.predict(x)[0] for tree in self.model.estimators_], dtype=float)
        std = float(np.std(per_tree, ddof=0))
        conductivity = float(10.0 ** log_pred)
        return ConductivityPrediction(
            conductivity_s_cm=conductivity,
            conductivity_ms_cm=conductivity * 1000.0,
            log10_conductivity_s_cm=log_pred,
            ensemble_std_log10=std,
            estimated_factor_1sigma=float(10.0**std),
            in_domain=True,
            features=features,
        )

    def predict_mmol(
        self,
        *,
        temperature_c: float,
        napf6_mmol: float,
        ec_mmol: float,
        emc_mmol: float,
        pc_mmol: float,
    ) -> ConductivityPrediction:
        return self.predict_features(
            features_from_mmol(
                temperature_c=temperature_c,
                napf6_mmol=napf6_mmol,
                ec_mmol=ec_mmol,
                emc_mmol=emc_mmol,
                pc_mmol=pc_mmol,
            )
        )


def grouped_cross_validation(
    table: pd.DataFrame,
    *,
    folds: int = 5,
    random_state: int = 20260906,
) -> GroupedCvMetrics:
    """Evaluate held-out formulations, keeping all temperatures of one formulation together."""

    groups = table["formulation_group"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(int(folds), len(unique_groups))
    if n_splits < 2:
        raise ConductivityModelError("grouped CV requires at least two formulations")
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    x = table.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    y = table["log10_conductivity"].to_numpy(dtype=float)
    truth: list[float] = []
    pred: list[float] = []
    for train_idx, test_idx in splitter.split(x, y, groups=groups):
        model = SodiumConductivityRandomForest(random_state=random_state)
        model.model.fit(x[train_idx], y[train_idx])
        fold_pred = model.model.predict(x[test_idx])
        truth.extend(y[test_idx].tolist())
        pred.extend(fold_pred.tolist())

    truth_array = np.asarray(truth, dtype=float)
    pred_array = np.asarray(pred, dtype=float)
    rmse = float(math.sqrt(mean_squared_error(truth_array, pred_array)))
    mae = float(mean_absolute_error(truth_array, pred_array))
    r2 = float(r2_score(truth_array, pred_array))
    return GroupedCvMetrics(
        folds=n_splits,
        rows=int(len(table)),
        unique_formulations=int(len(unique_groups)),
        rmse_log10=rmse,
        mae_log10=mae,
        r2_log10=r2,
        multiplicative_rmse_factor=float(10.0**rmse),
    )


def fit_public_na_model(
    frames: Sequence[pd.DataFrame],
    *,
    random_state: int = 20260906,
) -> tuple[SodiumConductivityRandomForest, pd.DataFrame]:
    table = prepare_training_table(frames)
    model = SodiumConductivityRandomForest(random_state=random_state).fit(table)
    return model, table
