"""gsplat の CUDA カーネルを JIT ビルドするための環境設定。

システムに CUDA ツールチェーンが無くても、venv に入れた pip 版 nvcc（`nvidia-cuda-nvcc`）を
`CUDA_HOME` として使う。設定は gsplat の import より前に行う必要がある。
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

# pip 版 nvcc と cccl ヘッダでバージョン表記が食い違い、互換性チェックで止まるのを避ける
NVCC_FLAGS = "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
# 既定の ~/.cache/torch_extensions は他プロジェクト（別バージョンの gsplat / torch）と共有されるため分ける
EXTENSIONS_DIR = Path.home() / ".cache" / "spatialsam" / "torch_extensions"


def pip_cuda_home() -> Path | None:
    """venv 内の `nvidia/cu13`（nvcc を含むもの）。無ければ None。"""
    try:
        import nvidia
    except ImportError:
        return None
    for base in map(Path, nvidia.__path__):
        candidate = base / "cu13"
        if (candidate / "bin" / "nvcc").exists():
            return candidate
    return None


def ensure_cudart_link(cuda_home: Path) -> None:
    """リンカが `-lcudart` で見つけられるよう、`libcudart.so` → `libcudart.so.13` を張る。"""
    link = cuda_home / "lib" / "libcudart.so"
    if not link.exists():
        link.symlink_to("libcudart.so.13")


def build_env(cuda_home: Path, arch: str | None, current: MutableMapping[str, str]) -> dict[str, str]:
    """設定すべき環境変数（既にユーザーが設定している値は尊重する）。"""
    updates = {
        "CUDA_HOME": str(cuda_home),
        "PATH": f"{cuda_home / 'bin'}{os.pathsep}{current.get('PATH', '')}",
        "NVCC_PREPEND_FLAGS": f"{NVCC_FLAGS} {current.get('NVCC_PREPEND_FLAGS', '')}".strip(),
        "TORCH_EXTENSIONS_DIR": str(EXTENSIONS_DIR),
    }
    if arch:
        updates["TORCH_CUDA_ARCH_LIST"] = arch
    keep = {"CUDA_HOME", "TORCH_CUDA_ARCH_LIST", "TORCH_EXTENSIONS_DIR"}
    return {k: v for k, v in updates.items() if not (k in keep and current.get(k))}


def detect_arch() -> str | None:
    """ビルド対象をこの GPU（例: A1000 は 8.6）だけにして初回ビルドを短くする。"""
    import torch

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def configure_cuda_build_env(env: MutableMapping[str, str] = os.environ) -> Path | None:
    cuda_home = pip_cuda_home()
    if cuda_home is None or env.get("CUDA_HOME"):
        return Path(env["CUDA_HOME"]) if env.get("CUDA_HOME") else None
    ensure_cudart_link(cuda_home)
    env.update(build_env(cuda_home, detect_arch(), env))
    return cuda_home


def import_gsplat():
    """環境を整えてから gsplat を import する（初回は CUDA カーネルのビルドで約 2 分）。"""
    configure_cuda_build_env()
    import gsplat

    return gsplat
