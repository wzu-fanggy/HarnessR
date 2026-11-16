"""本地模型 backing resource 的 Capability adapter。"""

from __future__ import annotations

import gc
from time import monotonic
from typing import Any

from ..model import ResidencyLevel, ResourceVector
from .base import BaseCapabilityAdapter


class LocalModelCapabilityAdapter(BaseCapabilityAdapter):
    """按需加载/释放一个本地 SentenceTransformer 模型。

    该 adapter 面向 CWSR 的 reference implementation：模型在 BOUND+ 层级加载，
    降到 BOUND 以下时删除模型实例并清理 CUDA cache。资源 admission 使用显式声明的
    GPU VRAM 成本；加载后 measured cost 使用 torch CUDA allocated-memory 增量做二次复核。

    当前实现刻意不把进程内 host RSS 计入模型可回收成本：PyTorch/Python allocator 在
    删除模型后可能继续保留一部分主机内存，因此不能把这部分 RSS 伪装成已被 CWSR
    可靠释放。若部署要求连同模型 host RAM 一并回收，应使用独立进程或容器作为 backing。

    这不是通用模型服务调度器。当前实现假设 adapter 对该模型实例拥有生命周期控制权，
    因而不应包裹仍被其它组件共享引用的全局 singleton。
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cuda",
        declared_gpu_vram_mb: float,
        warmup_text: str = "sodium-ion battery electrolyte",
    ) -> None:
        if not str(model_name).strip():
            raise ValueError("model_name is required")
        if declared_gpu_vram_mb < 0:
            raise ValueError("declared_gpu_vram_mb must be non-negative")
        self.model_name = str(model_name)
        self.device = str(device or "cpu")
        self.declared_gpu_vram_mb = float(declared_gpu_vram_mb)
        self.warmup_text = str(warmup_text)
        self._model: Any | None = None
        self._torch: Any | None = None
        self._baseline_gpu_mb = 0.0
        self.last_load_latency_s = 0.0
        self.last_unload_latency_s = 0.0
        self.embedding_dimension: int | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND:
            return ResourceVector()
        return ResourceVector(gpu_vram_mb=self.declared_gpu_vram_mb)

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND or not self.is_loaded:
            return ResourceVector()
        gpu_delta = max(0.0, self._current_gpu_mb() - self._baseline_gpu_mb)
        return ResourceVector(gpu_vram_mb=gpu_delta)

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current < ResidencyLevel.BOUND <= target and not self.is_loaded:
            self._load_model()

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current >= ResidencyLevel.BOUND > target:
            self._unload_model()

    def close(self) -> None:
        self._unload_model()

    def embed_query(self, text: str) -> list[float]:
        if self._model is None:
            raise RuntimeError("model capability is not loaded")
        vector = self._model.encode(str(text), normalize_embeddings=True)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        result = [float(item) for item in vector]
        if self.embedding_dimension is None:
            self.embedding_dimension = len(result)
        return result

    def _load_model(self) -> None:
        # Lazy imports keep the base CWSR package free of a mandatory torch/ST dependency.
        import torch
        from sentence_transformers import SentenceTransformer

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA model capability requested but torch.cuda is unavailable")
        self._torch = torch
        self._baseline_gpu_mb = self._current_gpu_mb()
        started = monotonic()
        model = SentenceTransformer(
            self.model_name,
            device=self.device,
            model_kwargs={"low_cpu_mem_usage": False},
        )
        # A real encode makes CUDA allocations observable before admission reconciliation.
        warm = model.encode(self.warmup_text, normalize_embeddings=True)
        if hasattr(warm, "shape") and len(getattr(warm, "shape", ())) > 0:
            self.embedding_dimension = int(warm.shape[-1])
        elif hasattr(warm, "__len__"):
            self.embedding_dimension = len(warm)
        self._model = model
        self.last_load_latency_s = monotonic() - started

    def _unload_model(self) -> None:
        if self._model is None:
            return
        started = monotonic()
        self._model = None
        gc.collect()
        torch = self._torch
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                try:
                    ipc_collect()
                except RuntimeError:
                    pass
        self.last_unload_latency_s = monotonic() - started

    def _current_gpu_mb(self) -> float:
        torch = self._torch
        if torch is None or not self.device.startswith("cuda"):
            return 0.0
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.memory_allocated(self.device)) / (1024.0 * 1024.0)
