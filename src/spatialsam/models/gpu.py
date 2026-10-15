"""GPU状態の取得と Peak VRAM / 処理時間の計測。torch が無くても動く。"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType

MIB = 1024 * 1024


@dataclass(frozen=True)
class GpuStatus:
    available: bool
    name: str = ""
    total_mib: float = 0.0
    free_mib: float = 0.0
    reason: str = ""


@dataclass
class Measurement:
    elapsed_sec: float = 0.0
    peak_vram_mib: float | None = None
    extra: dict[str, float] = field(default_factory=dict)


def _import_torch() -> ModuleType | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def query_gpu_status(torch_module: ModuleType | None = None) -> GpuStatus:
    torch = torch_module or _import_torch()
    if torch is None:
        return GpuStatus(False, reason="torch がインストールされていません（uv sync --extra gpu）")
    if not torch.cuda.is_available():
        return GpuStatus(False, reason="CUDA が利用できません")
    free, total = torch.cuda.mem_get_info()
    return GpuStatus(True, torch.cuda.get_device_name(0), total / MIB, free / MIB)


def is_out_of_memory(error: BaseException) -> bool:
    """torch.cuda.OutOfMemoryError（torch を import せずに判定）。"""
    return type(error).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(error)


def empty_cuda_cache() -> None:
    torch = _import_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def measure(torch_module: ModuleType | None = None) -> Iterator[Measurement]:
    """with ブロックの経過時間と（CUDAがあれば）Peak VRAM を記録する。"""
    torch = torch_module or _import_torch()
    use_cuda = torch is not None and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    result = Measurement()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_sec = time.perf_counter() - start
        if use_cuda:
            result.peak_vram_mib = torch.cuda.max_memory_allocated() / MIB
