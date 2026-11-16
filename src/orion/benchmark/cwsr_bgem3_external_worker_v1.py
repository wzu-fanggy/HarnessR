"""Persistent external BGE-M3 service shell for real CWSR RAG/KG replay.

The process models an already-running local RAG service. Importing Python, torch,
and sentence-transformers belongs to service infrastructure and is kept outside
the capability-residency timing window. The BGE-M3 model itself is managed
explicitly through a small JSON-lines protocol:

- TRANSPORT_READY: service shell is alive, but the model is not loaded;
- MODEL_READY: BGE-M3 has been loaded and warmed up;
- RESULT: one embedding request completed;
- MODEL_UNLOADED: model tensors and CUDA cache were released while the shell stays alive;
- STOPPED: the service shell is terminating.

This separation prevents model reload measurements from being inflated by repeated
Python interpreter and framework startup.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from time import monotonic
from typing import Any


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _cuda_mib(torch, kind: str, device: str) -> float:
    if not device.startswith("cuda"):
        return 0.0
    if kind == "allocated":
        value = torch.cuda.memory_allocated(device)
    elif kind == "reserved":
        value = torch.cuda.memory_reserved(device)
    elif kind == "peak_allocated":
        value = torch.cuda.max_memory_allocated(device)
    elif kind == "peak_reserved":
        value = torch.cuda.max_memory_reserved(device)
    else:  # pragma: no cover - internal guard
        raise ValueError(kind)
    return float(value) / (1024.0 * 1024.0)


def _cleanup_model(torch, model: Any, *, device: str) -> tuple[None, dict[str, float]]:
    started = monotonic()
    model = None
    gc.collect()
    if device.startswith("cuda"):
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
            except Exception:
                pass
    return None, {
        "cleanup_s": monotonic() - started,
        "torch_allocated_after_cleanup_mib": _cuda_mib(torch, "allocated", device),
        "torch_reserved_after_cleanup_mib": _cuda_mib(torch, "reserved", device),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-text", default="sodium-ion battery electrolyte")
    args = parser.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the external BGE worker")

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model: Any | None = None
    _emit(
        {
            "status": "TRANSPORT_READY",
            "pid": os.getpid(),
            "model": str(model_path),
            "device": args.device,
            "torch_allocated_mib": _cuda_mib(torch, "allocated", args.device),
            "torch_reserved_mib": _cuda_mib(torch, "reserved", args.device),
        }
    )

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            action = str(request.get("action") or "")

            if action == "load":
                if model is None:
                    before_allocated = _cuda_mib(torch, "allocated", args.device)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                        torch.cuda.reset_peak_memory_stats(args.device)
                    started = monotonic()
                    model = SentenceTransformer(
                        str(model_path),
                        device=args.device,
                        model_kwargs={"low_cpu_mem_usage": False},
                    )
                    warm = model.encode(args.warmup_text, normalize_embeddings=True)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    load_s = monotonic() - started
                    dimension = int(warm.shape[-1]) if hasattr(warm, "shape") else len(warm)
                else:
                    before_allocated = _cuda_mib(torch, "allocated", args.device)
                    load_s = 0.0
                    dimension = int(
                        model.get_sentence_embedding_dimension()
                        if hasattr(model, "get_sentence_embedding_dimension")
                        else 0
                    )
                allocated = _cuda_mib(torch, "allocated", args.device)
                reserved = _cuda_mib(torch, "reserved", args.device)
                _emit(
                    {
                        "status": "MODEL_READY",
                        "pid": os.getpid(),
                        "model": str(model_path),
                        "device": args.device,
                        "dimension": dimension,
                        "load_s": load_s,
                        "torch_allocated_delta_mib": max(0.0, allocated - before_allocated),
                        "torch_allocated_mib": allocated,
                        "torch_reserved_mib": reserved,
                        "torch_peak_allocated_mib": _cuda_mib(torch, "peak_allocated", args.device),
                        "torch_peak_reserved_mib": _cuda_mib(torch, "peak_reserved", args.device),
                    }
                )

            elif action == "embed":
                if model is None:
                    raise RuntimeError("BGE model is not loaded")
                text = str(request.get("text") or "")
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                    torch.cuda.reset_peak_memory_stats(args.device)
                embed_started = monotonic()
                vector = model.encode(text, normalize_embeddings=True)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                embed_s = monotonic() - embed_started
                if hasattr(vector, "tolist"):
                    vector = vector.tolist()
                _emit(
                    {
                        "status": "RESULT",
                        "request_id": request.get("request_id"),
                        "embed_s": embed_s,
                        "dimension": len(vector),
                        "vector": [float(item) for item in vector],
                        "torch_allocated_mib": _cuda_mib(torch, "allocated", args.device),
                        "torch_reserved_mib": _cuda_mib(torch, "reserved", args.device),
                        "torch_peak_allocated_mib": _cuda_mib(torch, "peak_allocated", args.device),
                        "torch_peak_reserved_mib": _cuda_mib(torch, "peak_reserved", args.device),
                    }
                )

            elif action == "unload":
                if model is None:
                    cleanup = {
                        "cleanup_s": 0.0,
                        "torch_allocated_after_cleanup_mib": _cuda_mib(
                            torch, "allocated", args.device
                        ),
                        "torch_reserved_after_cleanup_mib": _cuda_mib(
                            torch, "reserved", args.device
                        ),
                    }
                else:
                    # Drop the last model reference before forcing Python/CUDA cleanup.
                    loaded_model = model
                    model = None
                    del loaded_model
                    gc.collect()
                    cleanup_started = monotonic()
                    if args.device.startswith("cuda"):
                        try:
                            torch.cuda.synchronize(args.device)
                        except Exception:
                            pass
                        torch.cuda.empty_cache()
                        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                        if callable(ipc_collect):
                            try:
                                ipc_collect()
                            except Exception:
                                pass
                    cleanup = {
                        "cleanup_s": monotonic() - cleanup_started,
                        "torch_allocated_after_cleanup_mib": _cuda_mib(
                            torch, "allocated", args.device
                        ),
                        "torch_reserved_after_cleanup_mib": _cuda_mib(
                            torch, "reserved", args.device
                        ),
                    }
                _emit({"status": "MODEL_UNLOADED", **cleanup})

            elif action == "stop":
                if model is not None:
                    loaded_model = model
                    model = None
                    del loaded_model
                    gc.collect()
                    if args.device.startswith("cuda"):
                        try:
                            torch.cuda.synchronize(args.device)
                        except Exception:
                            pass
                        torch.cuda.empty_cache()
                _emit(
                    {
                        "status": "STOPPED",
                        "torch_allocated_mib": _cuda_mib(
                            torch, "allocated", args.device
                        ),
                        "torch_reserved_mib": _cuda_mib(
                            torch, "reserved", args.device
                        ),
                    }
                )
                return 0

            else:
                _emit(
                    {
                        "status": "ERROR",
                        "message": f"unsupported action: {action!r}",
                    }
                )
        except Exception as exc:
            _emit(
                {
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
