"""COLMAP の画像とカメラを学習用に読み込む。

Mip-NeRF 360 の慣例どおり、画像名順で 8 枚に 1 枚（0, 8, 16, ...）を評価用に除く。
評価用の視点は学習に使わないので、新規視点の描画品質（#16）の比較に使える。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spatialsam.spatial.camera import Camera
from spatialsam.spatial.colmap import ColmapView, load_views

HOLDOUT_EVERY = 8


@dataclass(frozen=True, eq=False)
class TrainView:
    name: str
    camera: Camera      # 読み込んだ画像の解像度に合わせたカメラ
    image: np.ndarray   # [H, W, 3] uint8 RGB


def split_views(views: list[ColmapView], holdout_every: int = HOLDOUT_EVERY) -> tuple[list, list]:
    """(学習用, 評価用)。holdout_every が 0 なら全部学習に使う。"""
    if holdout_every <= 0:
        return list(views), []
    train = [v for i, v in enumerate(views) if i % holdout_every != 0]
    test = [v for i, v in enumerate(views) if i % holdout_every == 0]
    return train, test


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"画像を読めません: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_train_view(view: ColmapView, images_dir: Path) -> TrainView:
    """縮小画像（images_4 など）でも、実際のサイズに合わせてカメラを縮める。"""
    image = read_rgb(images_dir / view.name)
    height, width = image.shape[:2]
    return TrainView(view.name, view.camera.resized(width, height), image)


@dataclass(frozen=True)
class ColmapDataset:
    train: list[TrainView]
    test: list[TrainView]

    @classmethod
    def load(cls, sparse_dir: Path, images_dir: Path, holdout_every: int = HOLDOUT_EVERY) -> ColmapDataset:
        train, test = split_views(load_views(sparse_dir), holdout_every)
        return cls([load_train_view(v, images_dir) for v in train], [load_train_view(v, images_dir) for v in test])

    def scene_scale(self) -> float:
        """カメラ中心のばらつき（最大距離）。学習率と初期化の基準にする（gsplat の simple_trainer と同じ）。"""
        centers = np.array([v.camera.position for v in self.train + self.test])
        return float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()) * 1.1
