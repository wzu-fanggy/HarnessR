"""独立原型：兼容科研计算共用进程，预测模型与排序模型分别持有。



不接入生产 Agent，不自动选工具，不改变科学计算。串行处理请求，释放操作

只会在一次计算结束后执行。UNLOADED 表示对象引用已移除，不表示内存已归还。

"""



from __future__ import annotations



import gc

import hashlib

import json

import os

from pathlib import Path

import sys

from threading import RLock

from time import perf_counter

from typing import Any



ROOT = Path(__file__).resolve().parents[2]

for import_root in (ROOT, ROOT.parent):

    if str(import_root) not in sys.path:

        sys.path.insert(0, str(import_root))





def safe_path(value: str | Path) -> Path:

    """只读资源须位于本项目根目录，解析符号链接后再校验。"""

    path = Path(value).expanduser().resolve()

    if not path.is_relative_to(ROOT.parent):

        raise ValueError(f"resource path escapes project: {path}")

    return path





def _sha256(path: Path) -> str:

    with path.open("rb") as stream:

        return hashlib.file_digest(stream, "sha256").hexdigest()





class SharedComputeState:

    """分别持有两个计算对象，准备失败保留此前可用对象。"""



    def __init__(self) -> None:

        self._objects: dict[str, Any] = {}

        self._keys: dict[str, tuple] = {}

        self._metadata: dict[str, dict] = {}

        self._prepare_counts: dict[str, int] = {}

        self._lock = RLock()



    def handle(self, payload: dict) -> dict:

        with self._lock:

            return self._handle(payload)



    def _handle(self, payload: dict) -> dict:

        if not isinstance(payload, dict):

            raise ValueError("request must be a JSON object")

        action = payload.get("action")

        if action == "inspect":

            return {"status": "STATE", "loaded_objects": sorted(self._objects),

                    "prepare_counts": dict(self._prepare_counts),

                    "objects": {key: dict(value) for key, value in self._metadata.items()},

                    "unrelated_imports": [name for name in ("torch", "transformers")

                                          if name in sys.modules]}

        if action == "unload":

            name = payload.get("object")

            if name not in {"conductivity", "robust_ranking"}:

                raise ValueError(f"unknown object: {name!r}")

            existed = name in self._objects

            self._objects.pop(name, None)

            self._keys.pop(name, None)

            self._metadata.pop(name, None)

            gc.collect()

            return {"status": "UNLOADED", "object": name, "was_loaded": existed,

                    "loaded_objects": sorted(self._objects)}

        if action == "load_conductivity":

            return self._load_conductivity(payload)

        if action == "prepare_ranking":

            return self._prepare_ranking(payload)

        if action == "predict_features":

            from tools.electrolyte_design.conductivity_ml import ConductivityFeatures, OutOfDomainError

            model = self._required("conductivity")

            values = payload.get("features")

            if not isinstance(values, dict):

                raise ValueError("predict_features requires feature mapping")

            features = ConductivityFeatures(**{name: float(values[name]) for name in

                ("inverse_temperature", "x_NaSalt", "x_EC", "x_PC")})

            try:

                prediction = model.predict_features(features)

            except OutOfDomainError as exc:

                return {"status": "OUT_OF_DOMAIN", "reason": str(exc)}

            return {"status": "RESULT", "prediction": prediction.to_dict()}

        if action == "rank_formulations":

            from tools.electrolyte_design.robust_ranking import rank_formulations

            return {"status": "RESULT", "result": rank_formulations(

                self._request(payload), ensemble=self._required("robust_ranking"))}

        if action == "fit_temperature_laws":

            from tools.electrolyte_design.temperature_fit import fit_temperature_laws

            return {"status": "RESULT", "result": fit_temperature_laws(self._request(payload))}

        if action == "stop":

            self._objects.clear()

            self._keys.clear()

            self._metadata.clear()

            gc.collect()

            return {"status": "STOPPED"}

        raise ValueError(f"unknown action: {action!r}")



    @staticmethod

    def _request(payload: dict) -> dict:

        request = payload.get("request")

        if not isinstance(request, dict):

            raise ValueError("action requires request object")

        return request



    def _required(self, name: str) -> Any:

        if name not in self._objects:

            raise RuntimeError(f"object is not loaded: {name}")

        return self._objects[name]



    def _ready(self, name: str, reused: bool) -> dict:

        return {"status": "READY", "object": name, "reused": reused,

                **self._metadata[name]}



    def _load_conductivity(self, payload: dict) -> dict:

        import joblib

        from tools.electrolyte_design.conductivity_ml import SodiumConductivityRandomForest

        path = safe_path(str(payload.get("model_path") or ""))

        key = (str(path), _sha256(path))

        if self._keys.get("conductivity") == key:

            return self._ready("conductivity", True)

        started = perf_counter()

        model = joblib.load(path)

        if not isinstance(model, SodiumConductivityRandomForest) or not model.fitted:

            raise TypeError("serialized model is not a fitted SodiumConductivityRandomForest")

        if _sha256(path) != key[1]:

            raise RuntimeError("model source changed during preparation")

        metadata = {"model_path": str(path), "model_sha256": key[1],

                    "training_rows": model.training_rows,

                    "training_formulations": model.training_formulations,

                    "domain": model.domain(), "prepare_seconds": perf_counter() - started}

        self._publish("conductivity", model, key, metadata)

        return self._ready("conductivity", False)



    def _prepare_ranking(self, payload: dict) -> dict:

        from tools.electrolyte_design.robust_ranking import (

            DEFAULT_MEMBERS, DEFAULT_TREES_PER_MEMBER, development_frames,

            development_training_table, fit_robust_ensemble, resolve_bundle_dir,

        )

        bundle = safe_path(resolve_bundle_dir(payload.get("bundle_dir")))

        paths = sorted(bundle.glob("hts_data_gen*.csv"))

        if len(paths) != 3:

            raise ValueError("ranking bundle requires exactly three public CSVs")

        members = int(payload.get("members", DEFAULT_MEMBERS))

        trees = int(payload.get("trees_per_member", DEFAULT_TREES_PER_MEMBER))

        if members < 2 or trees < 1:

            raise ValueError("ranking needs at least two members and one tree per member")

        # 路径、内容及计算参数均参与身份判断，不把旧对象用于不同模型。

        key = (str(bundle), tuple((str(path), _sha256(safe_path(path))) for path in paths),

               members, trees)

        if self._keys.get("robust_ranking") == key:

            return self._ready("robust_ranking", True)

        started = perf_counter()

        frames = development_frames(bundle)

        full_table = development_training_table(frames, drop_reserved_holdout=False)

        table = development_training_table(frames, drop_reserved_holdout=True)

        ensemble = fit_robust_ensemble(table, members=members, trees_per_member=trees)

        ensemble.excluded_holdout_formulations = int(

            full_table["formulation_group"].nunique() - table["formulation_group"].nunique())

        if tuple((str(path), _sha256(safe_path(path))) for path in paths) != key[1]:

            raise RuntimeError("ranking source changed during preparation")

        metadata = {"bundle_dir": str(bundle), "source_sha256": dict(key[1]),

                    "ensemble_members": ensemble.member_count, "trees_per_member": trees,

                    "training_rows": ensemble.training_rows,

                    "training_formulations": ensemble.training_formulations,

                    "excluded_holdout_formulations": ensemble.excluded_holdout_formulations,

                    "prepare_seconds": perf_counter() - started}

        self._publish("robust_ranking", ensemble, key, metadata)

        return self._ready("robust_ranking", False)



    def _publish(self, name: str, obj: Any, key: tuple, metadata: dict) -> None:

        # 计算完成且对象校验成功后才替换，异常不破坏此前对象。

        self._objects[name] = obj

        self._keys[name] = key

        self._metadata[name] = metadata

        self._prepare_counts[name] = self._prepare_counts.get(name, 0) + 1





def _emit(payload: dict) -> None:

    print(json.dumps({"pid": os.getpid(), **payload}, ensure_ascii=False, sort_keys=True), flush=True)





def main() -> int:

    # 先回报实际解释器 PID，准备超时时父进程能终止真正的 worker。

    _emit({"status": "BOOTED", "python": sys.version.split()[0]})

    import joblib  # noqa: F401

    import numpy  # noqa: F401

    import pandas  # noqa: F401

    import scipy  # noqa: F401

    import sklearn  # noqa: F401

    from tools.electrolyte_design import conductivity_ml, robust_ranking, temperature_fit  # noqa: F401

    state = SharedComputeState()

    _emit({"status": "READY", "object": "environment"})

    for raw in sys.stdin:

        if not raw.strip():

            continue

        try:

            result = state.handle(json.loads(raw))

            _emit(result)

            if result["status"] == "STOPPED":

                return 0

        except Exception as exc:

            _emit({"status": "ERROR", "error": type(exc).__name__, "message": str(exc)})

    return 0





if __name__ == "__main__":

    raise SystemExit(main())

