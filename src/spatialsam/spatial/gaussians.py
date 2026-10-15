"""3D Gaussian の集合と、標準的な 3DGS PLY（INRIA 実装と同じ形式）の読み書き。

PLY には活性化前の値を保存する（scale は log、opacity は logit、色は SH 係数）。
`f_rest_*` はチャンネル優先（R の高次係数 → G → B）で並ぶ。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


@dataclass(frozen=True)
class GaussianScene:
    means: np.ndarray       # [N, 3]
    quats: np.ndarray       # [N, 4]（w, x, y, z。正規化前）
    log_scales: np.ndarray  # [N, 3]
    opacity_logits: np.ndarray  # [N]
    sh: np.ndarray          # [N, K, 3]（K = (degree + 1)^2。0 番目が DC 成分）

    def __post_init__(self) -> None:
        n = len(self.means)
        shapes = {"quats": (n, 4), "log_scales": (n, 3), "opacity_logits": (n,)}
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} の形状が不正: {getattr(self, name).shape}（期待値 {shape}）")
        if self.sh.ndim != 3 or self.sh.shape[0] != n or self.sh.shape[2] != 3:
            raise ValueError(f"sh の形状が不正: {self.sh.shape}")
        sh_degree_from_coeffs(self.sh.shape[1])

    def __len__(self) -> int:
        return len(self.means)

    @property
    def sh_degree(self) -> int:
        return sh_degree_from_coeffs(self.sh.shape[1])

    def bounds(self, quantile: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """中心座標の範囲。外れ値（浮遊 Gaussian）を除くには quantile を 0.01 などにする。"""
        return np.quantile(self.means, quantile, axis=0), np.quantile(self.means, 1 - quantile, axis=0)


def sh_degree_from_coeffs(num_coeffs: int) -> int:
    degree = math.isqrt(num_coeffs) - 1
    if degree < 0 or (degree + 1) ** 2 != num_coeffs:
        raise ValueError(f"SH 係数の数が平方数ではありません: {num_coeffs}")
    return degree


def _columns(vertex: PlyElement, prefix: str) -> np.ndarray:
    names = sorted((p.name for p in vertex.properties if p.name.startswith(prefix)),
                   key=lambda n: int(n[len(prefix):]))
    return np.stack([np.asarray(vertex[n], np.float32) for n in names], axis=1) if names else np.zeros(
        (vertex.count, 0), np.float32)


def _sh_from_columns(dc: np.ndarray, rest: np.ndarray) -> np.ndarray:
    n = len(dc)
    rest = rest.reshape(n, 3, rest.shape[1] // 3).transpose(0, 2, 1)  # [N, K-1, 3]
    return np.concatenate([dc[:, None, :], rest], axis=1)


def load_ply(path: Path) -> GaussianScene:
    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.stack([np.asarray(vertex[c], np.float32) for c in "xyz"], axis=1)
    return GaussianScene(
        means=xyz,
        quats=_columns(vertex, "rot_"),
        log_scales=_columns(vertex, "scale_"),
        opacity_logits=np.asarray(vertex["opacity"], np.float32),
        sh=_sh_from_columns(_columns(vertex, "f_dc_"), _columns(vertex, "f_rest_")),
    )


def _ply_fields(scene: GaussianScene) -> dict[str, np.ndarray]:
    n = len(scene)
    rest = scene.sh[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)
    fields = {c: scene.means[:, i] for i, c in enumerate("xyz")}
    fields |= {f"n{c}": np.zeros(n, np.float32) for c in "xyz"}
    fields |= {f"f_dc_{i}": scene.sh[:, 0, i] for i in range(3)}
    fields |= {f"f_rest_{i}": rest[:, i] for i in range(rest.shape[1])}
    fields["opacity"] = scene.opacity_logits
    fields |= {f"scale_{i}": scene.log_scales[:, i] for i in range(3)}
    fields |= {f"rot_{i}": scene.quats[:, i] for i in range(4)}
    return fields


def save_ply(scene: GaussianScene, path: Path) -> Path:
    fields = _ply_fields(scene)
    data = np.empty(len(scene), dtype=[(name, "f4") for name in fields])
    for name, values in fields.items():
        data[name] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))
    return path
