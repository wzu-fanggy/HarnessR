"""Sentence-transformers embedder for bge-m3."""
from __future__ import annotations

import gc
import math
import os
import threading
from time import monotonic

import torch
from sentence_transformers import SentenceTransformer

from kg_rag.config import EMBEDDING_MODEL
from kg_rag.resource_telemetry import process_rss_mib, recorder

try:
    from neo4j_graphrag.embeddings import Embedder
except Exception:  # pragma: no cover - neo4j-graphrag may be absent in minimal env
    Embedder = object


def _to_list(value):
    """Normalize encode output (numpy array or list) to plain list."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _gpu_allocated_mib(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        return round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3)
    except (RuntimeError, ValueError):
        return None


def _gpu_vram_pressure(device: str) -> float | None:
    """返回设备总显存的当前使用比例；CPU 或不可查询时返回 None。"""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if total_bytes <= 0:
            return None
        pressure = 1.0 - (free_bytes / total_bytes)
        return round(pressure, 6) if 0.0 <= pressure <= 1.0 else None
    except (RuntimeError, ValueError):
        return None


class Bgem3Embedder(Embedder):
    """bge-m3 embedder with process-wide model singleton.

    Loading the same model twice (e.g. once for kg_rag hybrid search and once
    for Orion long-term memory) quickly exhausts RAM/VRAM on a laptop. We keep
    a class-level cache keyed by (model_name, device) so all callers share one
    instance. A lock serializes the actual load to avoid the Windows
    torch.storage segfault that can happen when multiple threads load the
    model concurrently.
    """

    _model_cache: dict[tuple[str, str], SentenceTransformer] = {}
    _load_lock: threading.Lock = threading.Lock()

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None):
        if device is None:
            device = os.environ.get("KG_RAG_EMBEDDING_DEVICE", "")
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model_name = model_name
        self._cache_key = (model_name, device)
        self._resource_key = f"embedding:{os.path.basename(os.path.normpath(model_name))}:{device}"
        self.load_count = 0
        self.release_count = 0
        self.last_load_seconds = 0.0
        self.last_host_ram_mib: float | None = None
        self.last_rss_before_mib: float | None = None
        self.last_rss_after_mib: float | None = None
        self.last_rss_delta_mib: float | None = None
        self.host_ram_measurement = "not_observed"
        self.last_release_rss_before_mib: float | None = None
        self.last_release_rss_after_mib: float | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the process-wide model instance is currently materialized."""

        return self._cache_key in Bgem3Embedder._model_cache

    @property
    def model(self) -> SentenceTransformer:
        """Return the shared model, loading it on demand after a runtime release."""

        return self.ensure_loaded()

    def ensure_loaded(self) -> SentenceTransformer:
        """Materialize the process-wide model if it is currently released."""

        if self._cache_key not in Bgem3Embedder._model_cache:
            with Bgem3Embedder._load_lock:
                if self._cache_key not in Bgem3Embedder._model_cache:
                    rss_before_mib = process_rss_mib()
                    gpu_before_mib = _gpu_allocated_mib(self._device)
                    started = monotonic()
                    Bgem3Embedder._model_cache[self._cache_key] = SentenceTransformer(
                        self._model_name,
                        device=self._device,
                        model_kwargs={"low_cpu_mem_usage": False},
                    )
                    self.load_count += 1
                    self.last_load_seconds = monotonic() - started
                    rss_after_mib = process_rss_mib()
                    self.last_rss_before_mib = rss_before_mib
                    self.last_rss_after_mib = rss_after_mib
                    if (rss_before_mib is not None and rss_after_mib is not None
                            and math.isfinite(rss_before_mib) and math.isfinite(rss_after_mib)):
                        self.last_rss_delta_mib = float(rss_after_mib) - float(rss_before_mib)
                        self.last_host_ram_mib = (self.last_rss_delta_mib
                                                 if self.last_rss_delta_mib > 0 else None)
                        self.host_ram_measurement = ("load_rss_increment"
                            if self.last_rss_delta_mib > 0 else "nonpositive_load_delta")
                    else:
                        self.last_rss_delta_mib = None
                        self.last_host_ram_mib = None
                        self.host_ram_measurement = "unavailable"
                    try:
                        recorder().loaded(
                            self._resource_key,
                            load_seconds=self.last_load_seconds,
                            rss_before_mib=rss_before_mib,
                            gpu_before_mib=gpu_before_mib,
                            gpu_allocated_mib=_gpu_allocated_mib(self._device),
                            gpu_vram_pressure=_gpu_vram_pressure(self._device),
                        )
                    except Exception:
                        # 观测失败不得使已加载的模型不可用。
                        pass
        return Bgem3Embedder._model_cache[self._cache_key]

    def release(self) -> bool:
        """Release the shared model weights while keeping this embedder reusable."""

        with Bgem3Embedder._load_lock:
            model = Bgem3Embedder._model_cache.pop(self._cache_key, None)
            if model is None:
                return False
            self.last_release_rss_before_mib = process_rss_mib()
            del model
            gc.collect()
            self.release_count += 1
            if self._device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    try:
                        ipc_collect()
                    except RuntimeError:
                        pass
            self.last_release_rss_after_mib = process_rss_mib()
            return True

    def runtime_snapshot(self) -> dict[str, object]:
        """Return non-sensitive residency telemetry for the local service API."""

        return {
            "resource_id": "bge-m3-model",
            "resident": self.is_loaded,
            "host_ram_mib": self.last_host_ram_mib if self.is_loaded else 0.0,
            "host_ram_measurement": self.host_ram_measurement if self.is_loaded else "not_resident",
            "load_rss_before_mib": self.last_rss_before_mib,
            "load_rss_after_mib": self.last_rss_after_mib,
            "load_rss_delta_mib": self.last_rss_delta_mib,
            "release_rss_before_mib": self.last_release_rss_before_mib,
            "release_rss_after_mib": self.last_release_rss_after_mib,
            "gpu_allocated_mib": _gpu_allocated_mib(self._device) if self.is_loaded else 0.0,
            "gpu_vram_pressure": _gpu_vram_pressure(self._device),
            "device": self._device,
            "load_count": self.load_count,
            "release_count": self.release_count,
            "last_load_seconds": self.last_load_seconds,
        }

    def embed_query(self, text: str) -> list[float]:
        try:
            started = recorder().begin_use(
                self._resource_key,
                gpu_vram_pressure=_gpu_vram_pressure(self._device),
            )
        except Exception:
            started = None
        success = False
        try:
            result = _to_list(self.model.encode(text, normalize_embeddings=True))
            success = True
            return result
        finally:
            if started is not None:
                try:
                    recorder().end_use(
                        self._resource_key,
                        started,
                        success=success,
                        gpu_vram_pressure=_gpu_vram_pressure(self._device),
                    )
                except Exception:
                    pass

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        try:
            started = recorder().begin_use(
                self._resource_key,
                gpu_vram_pressure=_gpu_vram_pressure(self._device),
            )
        except Exception:
            started = None
        success = False
        try:
            result = _to_list(
                self.model.encode(texts, normalize_embeddings=True, batch_size=batch_size)
            )
            success = True
            return result
        finally:
            if started is not None:
                try:
                    recorder().end_use(
                        self._resource_key,
                        started,
                        success=success,
                        gpu_vram_pressure=_gpu_vram_pressure(self._device),
                    )
                except Exception:
                    pass

    def get_embedding_dimension(self) -> int:
        model = self.model
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()
