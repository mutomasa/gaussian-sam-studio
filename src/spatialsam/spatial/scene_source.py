"""描画する 3D シーンの入力（PLY と COLMAP の組み合わせ）と既定の軌道。

- PLY のみ: 3DGS または点群。軌道はシーンの範囲から決める（上方向は指定）
- COLMAP のみ: points3D を Gaussian にして描く。撮影カメラから軌道を推定できる
- 両方: 3DGS（COLMAP 座標で学習済み）を、撮影カメラの視点や推定した軌道で描く
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from spatialsam.spatial.colmap import ColmapView, load_views, read_points3d
from spatialsam.spatial.gaussians import GaussianScene
from spatialsam.spatial.pointcloud import gaussians_from_points, load_scene
from spatialsam.spatial.trajectory import OrbitSpec, orbit_from_cameras

AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


@dataclass(frozen=True)
class SceneSource:
    ply_path: Path | None = None
    sparse_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.ply_path is None and self.sparse_dir is None:
            raise ValueError("PLY か COLMAP の sparse ディレクトリのどちらかを指定してください")
        for path in (self.ply_path, self.sparse_dir):
            if path is not None and not path.exists():
                raise FileNotFoundError(f"見つかりません: {path}")

    @property
    def has_cameras(self) -> bool:
        return self.sparse_dir is not None

    @property
    def label(self) -> str:
        return str(self.ply_path or self.sparse_dir / "points3D.bin")

    def to_dict(self) -> dict[str, str | None]:
        return {"ply_path": str(self.ply_path) if self.ply_path else None,
                "sparse_dir": str(self.sparse_dir) if self.sparse_dir else None}

    @classmethod
    def from_dict(cls, raw: dict[str, str | None]) -> SceneSource:
        return cls(*(Path(raw[k]) if raw.get(k) else None for k in ("ply_path", "sparse_dir")))

    @classmethod
    def from_label(cls, label: str) -> SceneSource:
        """scene_source を記録していない古い描画結果用（label は PLY か points3D.bin のパス）。"""
        path = Path(label)
        return cls(sparse_dir=path.parent) if path.name == "points3D.bin" else cls(ply_path=path)

    def load_scene(self) -> GaussianScene:
        if self.ply_path is not None:
            return load_scene(self.ply_path)
        points = read_points3d(self.sparse_dir / "points3D.bin")
        return gaussians_from_points(points.xyz, points.rgb)

    def load_views(self) -> list[ColmapView]:
        return load_views(self.sparse_dir) if self.sparse_dir is not None else []


def orbit_from_bounds(scene: GaussianScene, up: np.ndarray, distance_factor: float = 1.0) -> OrbitSpec:
    """カメラ情報が無いとき: 外れ値を除いた範囲の中心を見て、範囲の大きさに比例した半径で回る。"""
    low, high = scene.bounds(quantile=0.02)
    extent = float(np.linalg.norm(high - low))
    return OrbitSpec(center=(low + high) / 2, up=np.asarray(up, float), radius=max(extent, 1e-3) * distance_factor)


def default_orbit(scene: GaussianScene, views: list[ColmapView], up_axis: str = "+z", **overrides) -> OrbitSpec:
    spec = orbit_from_cameras([v.camera for v in views]) if views else orbit_from_bounds(scene, np.array(AXES[up_axis]))
    return replace(spec, **overrides)
