"""点群（色付き）から 3D Gaussian を作る。学習せずにそのまま描画するためのもの。

各点を等方的な Gaussian にし、大きさは近傍点までの平均距離にする（INRIA 3DGS の初期化と同じ考え方）。
学習済み 3DGS と違い隙間やぼけが残るので、画質比較には学習済みのシーンを使う。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData

from spatialsam.spatial.gaussians import GaussianScene, load_ply
from spatialsam.spatial.point_io import PointCloudFilter, filter_points, read_point_cloud

SH_C0 = 0.28209479177387814  # 0 次の球面調和関数の係数
DEFAULT_OPACITY = 0.9


def rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    """[0, 255] の色 → DC 成分。gsplat は SH の評価結果に 0.5 を足して色にする。"""
    return ((rgb.astype(np.float32) / 255.0 - 0.5) / SH_C0).astype(np.float32)


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def neighbor_distances(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """各点から近い k 点までの平均距離。重複点で 0 にならないよう下限を付ける。"""
    from scipy.spatial import cKDTree

    if len(xyz) <= 1:
        return np.full(len(xyz), 0.01, np.float32)
    k = min(k, len(xyz) - 1)
    distances, _ = cKDTree(xyz).query(xyz, k=k + 1)  # 先頭は自分自身
    mean = distances[:, 1:].mean(axis=1)
    return np.maximum(mean, 1e-7).astype(np.float32)


def clip_outlier_scales(scales: np.ndarray, quantile: float = 0.5) -> np.ndarray:
    """疎な領域の点は近傍距離が大きく（COLMAP の bonsai で最大 45）、カメラ手前で視野を覆うぼけになるので抑える。

    学習しないまま描くので、中央値で頭打ちにするのが見た目上いちばん良かった（bonsai で 0.5 / 0.75 / 0.9 を比較）。
    """
    return np.minimum(scales, np.quantile(scales, quantile)) if len(scales) else scales


def gaussians_from_points(xyz: np.ndarray, rgb: np.ndarray, scale_factor: float = 1.0,
                          opacity: float = DEFAULT_OPACITY) -> GaussianScene:
    n = len(xyz)
    scales = clip_outlier_scales(neighbor_distances(xyz)) * scale_factor
    quats = np.zeros((n, 4), np.float32)
    quats[:, 0] = 1.0
    return GaussianScene(
        means=xyz.astype(np.float32),
        quats=quats,
        log_scales=np.repeat(np.log(scales)[:, None], 3, axis=1),
        opacity_logits=np.full(n, logit(opacity), np.float32),
        sh=rgb_to_sh0(rgb)[:, None, :],
    )


def is_gaussian_ply(path: Path) -> bool:
    names = {p.name for p in PlyData.read(str(path))["vertex"].properties}
    return {"f_dc_0", "opacity", "scale_0", "rot_0"} <= names


def points_to_scene(xyz: np.ndarray, rgb: np.ndarray, point_filter: PointCloudFilter | None = None) -> GaussianScene:
    """点群を（必要なら Open3D で前処理してから）等方 Gaussian にする。"""
    if point_filter is not None:
        xyz, rgb, _ = filter_points(xyz, rgb, point_filter)
    return gaussians_from_points(xyz, rgb)


def load_scene(path: Path, point_filter: PointCloudFilter | None = None) -> GaussianScene:
    """3DGS の PLY ならそのまま、点群（PLY / PCD / XYZ など）なら Open3D で読んで Gaussian に変換する。

    前処理（point_filter）は点群にだけ効く。学習済みの 3DGS には適用しない。
    """
    if path.suffix.lower() == ".ply" and is_gaussian_ply(path):
        return load_ply(path)
    return points_to_scene(*read_point_cloud(path), point_filter)
