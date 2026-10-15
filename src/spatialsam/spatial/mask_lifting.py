"""2D マスクを 3D に持ち上げる（docs/spatialsam_design.md §7）。

- 逆投影: マスク内の画素を深度で 3D 点にする（p_world = T_world←cam [z K⁻¹(u, v, 1)ᵀ; 1]）
- Gaussian 単位の投票: 各 Gaussian の中心を各視点に投影し、描画深度と比べて見えている視点だけで
  「マスクの内側に写ったか」を数える。内側の割合（信頼度）と見えた視点数でラベルを決める。
  遮蔽された視点を数えないので、手前の物体に隠れた Gaussian が誤ってラベルされるのを防げる。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from spatialsam.spatial.camera import Camera


@dataclass(frozen=True, eq=False)
class MaskedView:
    """1 視点分の入力。masks は物体 ID → [H, W] bool、depth / alpha は同じ解像度の描画結果。"""

    camera: Camera
    masks: dict[int, np.ndarray]
    depth: np.ndarray
    alpha: np.ndarray


@dataclass(frozen=True)
class LiftSettings:
    depth_tolerance: float = 0.05  # 描画深度に対する相対誤差。これより奥は遮蔽とみなす
    min_alpha: float = 0.5         # 描画の不透明度がこれ未満の画素は深度が当てにならないので使わない
    min_ratio: float = 0.5         # 見えた視点のうちマスク内だった割合がこれ以上ならラベルする
    min_views: int = 2             # 見えた視点がこれ未満の Gaussian はラベルしない


def project(points: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """world 座標 [N, 3] → 画素座標 [N, 2]（u, v）とカメラ z [N]。"""
    cam = points @ camera.world_to_cam[:3, :3].T + camera.world_to_cam[:3, 3]
    z = cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uv = (cam[:, :2] / safe_z[:, None]) @ camera.K[:2, :2].T + camera.K[:2, 2]
    return uv, z


def backproject(mask: np.ndarray, depth: np.ndarray, camera: Camera, valid: np.ndarray | None = None) -> np.ndarray:
    """マスク内（かつ深度が有効）の画素を world 座標の点 [M, 3] にする。"""
    keep = mask & (depth > 0) & (valid if valid is not None else True)
    v, u = np.nonzero(keep)
    z = depth[v, u]
    rays = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=1) @ np.linalg.inv(camera.K).T  # 画素中心
    cam = rays * z[:, None]
    return cam @ camera.cam_to_world[:3, :3].T + camera.cam_to_world[:3, 3]


def visible_pixels(points: np.ndarray, view: MaskedView, settings: LiftSettings) -> tuple[np.ndarray, np.ndarray]:
    """この視点で見えている点の添字と、その画素位置（行, 列）。"""
    height, width = view.depth.shape
    uv, z = project(points, view.camera)
    cols, rows = np.floor(uv[:, 0]).astype(np.int64), np.floor(uv[:, 1]).astype(np.int64)
    inside = (z > 0) & (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    index = np.nonzero(inside)[0]
    r, c = rows[index], cols[index]
    surface = view.depth[r, c]
    seen = (view.alpha[r, c] >= settings.min_alpha) & (z[index] <= surface * (1 + settings.depth_tolerance))
    return index[seen], np.stack([r[seen], c[seen]], axis=1)


@dataclass
class VoteCounts:
    """Gaussian ごとの「見えた視点数」と、物体 ID ごとの「マスク内に写った視点数」。"""

    visible: np.ndarray
    inside: dict[int, np.ndarray] = field(default_factory=dict)

    def add(self, view: MaskedView, index: np.ndarray, pixels: np.ndarray) -> None:
        self.visible[index] += 1
        for object_id, mask in view.masks.items():
            counts = self.inside.setdefault(object_id, np.zeros_like(self.visible))
            counts[index] += mask[pixels[:, 0], pixels[:, 1]]

    def confidence(self, object_id: int) -> np.ndarray:
        """見えた視点のうちマスク内だった割合（見えなかった Gaussian は 0）。"""
        inside = self.inside.get(object_id, np.zeros_like(self.visible))
        return np.divide(inside, self.visible, out=np.zeros(len(self.visible)), where=self.visible > 0)


@dataclass(frozen=True)
class LiftResult:
    votes: VoteCounts
    settings: LiftSettings
    num_views: int

    @property
    def object_ids(self) -> list[int]:
        return sorted(self.votes.inside)

    def labels(self, object_id: int) -> np.ndarray:
        s = self.settings
        return (self.votes.visible >= s.min_views) & (self.votes.confidence(object_id) >= s.min_ratio)

    def any_labels(self) -> np.ndarray:
        """いずれかの物体に属する Gaussian（テキストプロンプトに当たったものすべて）。"""
        labels = np.zeros(len(self.votes.visible), bool)
        for object_id in self.object_ids:
            labels |= self.labels(object_id)
        return labels


def lift_masks(points: np.ndarray, views: Iterable[MaskedView], settings: LiftSettings = LiftSettings()) -> LiftResult:
    votes = VoteCounts(np.zeros(len(points), np.int32))
    count = 0
    for view in views:
        index, pixels = visible_pixels(points, view, settings)
        votes.add(view, index, pixels)
        count += 1
    return LiftResult(votes, settings, count)
