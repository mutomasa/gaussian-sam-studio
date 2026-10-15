"""ピンホールカメラ（OpenCV 規約: x 右・y 下・z 前方）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)  # ndarray を持つので == は使わない
class Camera:
    K: np.ndarray             # [3, 3] 内部行列
    world_to_cam: np.ndarray  # [4, 4]
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.K.shape != (3, 3) or self.world_to_cam.shape != (4, 4):
            raise ValueError("K は 3x3、world_to_cam は 4x4")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("画像サイズは正")

    @property
    def cam_to_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_cam)

    @property
    def position(self) -> np.ndarray:
        return self.cam_to_world[:3, 3]

    def scaled(self, factor: float) -> Camera:
        """解像度を factor 倍にしたカメラ（内部行列も合わせて変える）。"""
        return self.resized(max(1, round(self.width * factor)), max(1, round(self.height * factor)))

    def resized(self, width: int, height: int) -> Camera:
        """縮小画像（例: Mip-NeRF 360 の images_4）に合わせたカメラ。縦横の倍率は別々に扱う。"""
        K = self.K.copy()
        K[0] *= width / self.width
        K[1] *= height / self.height
        return Camera(K, self.world_to_cam, width, height)

    def to_dict(self) -> dict:
        return {"K": self.K.tolist(), "world_to_cam": self.world_to_cam.tolist(),
                "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, raw: dict) -> Camera:
        return cls(np.asarray(raw["K"], float), np.asarray(raw["world_to_cam"], float),
                   int(raw["width"]), int(raw["height"]))


def intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def intrinsics_from_fov(fov_y_deg: float, width: int, height: int) -> np.ndarray:
    f = 0.5 * height / np.tan(np.radians(fov_y_deg) / 2)
    return intrinsics(f, f, width / 2, height / 2)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """eye から target を向くカメラの world_to_cam（OpenCV 規約。画像の上方向が up に近づく）。"""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("視線方向と up が平行です")
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])  # 行 = カメラ軸（world 座標）
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = rotation
    world_to_cam[:3, 3] = -rotation @ eye
    return world_to_cam
