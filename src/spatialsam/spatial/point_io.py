"""Open3D による点群の読み書きと前処理 #43。

- 読み書き: PLY / PCD / XYZ / XYZRGB / PTS（拡張子で形式を決める）。3DGS の PLY は gaussians.py（plyfile）で扱う
- 前処理:
  - 統計的外れ値除去（各点の k 近傍の平均距離が、全体の平均 + std_ratio × 標準偏差を超える点を除く）
  - ボクセルでの間引き（1 ボクセルに 1 点、色は平均）
  COLMAP の疎な点群は遠くの孤立点を含み、描画すると視野を覆うぼけになるので、外れ値除去が効く
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

POINT_CLOUD_SUFFIXES = (".ply", ".pcd", ".xyz", ".xyzrgb", ".pts")
GRAY = 128


@dataclass(frozen=True)
class PointCloudFilter:
    remove_outliers: bool = True
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    voxel_size: float | None = None  # None なら間引かない（シーンの単位）

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FilterStats:
    before: int
    after_outliers: int
    after_voxel: int

    def to_dict(self) -> dict:
        return asdict(self)


def _o3d():
    import open3d as o3d

    return o3d


def to_open3d(xyz: np.ndarray, rgb: np.ndarray):
    o3d = _o3d()
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(xyz, np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(rgb, np.float64) / 255.0)
    return cloud


def from_open3d(cloud) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(cloud.points, np.float32)
    if cloud.has_colors():
        rgb = np.clip(np.asarray(cloud.colors) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), GRAY, np.uint8)  # 色が無ければ灰色
    return xyz, rgb


def read_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() not in POINT_CLOUD_SUFFIXES:
        raise ValueError(f"対応していない点群の形式です: {path.suffix}（{', '.join(POINT_CLOUD_SUFFIXES)}）")
    cloud = _o3d().io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"点群が空か、読み込めません: {path}")
    return from_open3d(cloud)


def write_point_cloud(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _o3d().io.write_point_cloud(str(path), to_open3d(xyz, rgb)):
        raise OSError(f"点群を書き出せません: {path}")
    return path


def filter_points(xyz: np.ndarray, rgb: np.ndarray,
                  settings: PointCloudFilter) -> tuple[np.ndarray, np.ndarray, FilterStats]:
    cloud = to_open3d(xyz, rgb)
    before = len(cloud.points)
    if settings.remove_outliers and before > settings.outlier_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=settings.outlier_neighbors,
                                                    std_ratio=settings.outlier_std_ratio)
    after_outliers = len(cloud.points)
    if settings.voxel_size:
        cloud = cloud.voxel_down_sample(settings.voxel_size)
    xyz_out, rgb_out = from_open3d(cloud)
    return xyz_out, rgb_out, FilterStats(before, after_outliers, len(xyz_out))
