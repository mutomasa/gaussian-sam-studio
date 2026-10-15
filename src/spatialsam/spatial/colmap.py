"""COLMAP の sparse モデル（cameras.bin / images.bin / points3D.bin）の読み込み。

3DGS は通常 COLMAP の座標系で学習されるので、ここで得たカメラでそのまま描画すれば
学習に使った実画像と同じ視点になる。歪み係数は無視する（undistort 済みの PINHOLE を想定）。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from spatialsam.spatial.camera import Camera, intrinsics

# model_id → (名前, パラメータ数)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
FISHEYE_MODELS = {"OPENCV_FISHEYE", "FOV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE"}


@dataclass(frozen=True, eq=False)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def K(self) -> np.ndarray:
        if self.model in FISHEYE_MODELS:
            raise ValueError(f"魚眼モデルは未対応: {self.model}")
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return intrinsics(p[0], p[0], p[1], p[2])
        return intrinsics(p[0], p[1], p[2], p[3])


@dataclass(frozen=True, eq=False)
class ColmapImage:
    image_id: int
    qvec: np.ndarray  # (w, x, y, z) world→cam の回転
    tvec: np.ndarray
    camera_id: int
    name: str

    def world_to_cam(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = quat_to_rotation(self.qvec)
        matrix[:3, 3] = self.tvec
        return matrix


@dataclass(frozen=True, eq=False)
class ColmapPoints:
    xyz: np.ndarray  # [N, 3]
    rgb: np.ndarray  # [N, 3] uint8


@dataclass(frozen=True, eq=False)
class ColmapView:
    """1 枚の登録画像とその視点のカメラ。"""

    name: str
    camera: Camera


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _read(f: BinaryIO, fmt: str) -> tuple:
    size = struct.calcsize("<" + fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("COLMAP バイナリが途中で終わっています")
    return struct.unpack("<" + fmt, data)


def _read_name(f: BinaryIO) -> str:
    chars = bytearray()
    while (c := f.read(1)) not in (b"\x00", b""):
        chars += c
    return chars.decode("utf-8")


def read_cameras(path: Path) -> dict[int, ColmapCamera]:
    cameras = {}
    with path.open("rb") as f:
        for _ in range(_read(f, "Q")[0]):
            camera_id, model_id, width, height = _read(f, "iiQQ")
            name, num_params = CAMERA_MODELS[model_id]
            cameras[camera_id] = ColmapCamera(camera_id, name, width, height, np.array(_read(f, "d" * num_params)))
    return cameras


def read_images(path: Path) -> dict[int, ColmapImage]:
    images = {}
    with path.open("rb") as f:
        for _ in range(_read(f, "Q")[0]):
            image_id, *pose, camera_id = _read(f, "i7di")
            name = _read_name(f)
            f.seek(_read(f, "Q")[0] * 24, 1)  # 2D 点（x, y, point3D_id）は使わない
            images[image_id] = ColmapImage(image_id, np.array(pose[:4]), np.array(pose[4:]), camera_id, name)
    return images


def read_points3d(path: Path) -> ColmapPoints:
    xyz, rgb = [], []
    with path.open("rb") as f:
        for _ in range(_read(f, "Q")[0]):
            _, x, y, z, r, g, b, _, track_length = _read(f, "Q3d3BdQ")
            f.seek(track_length * 8, 1)  # track（image_id, point2D_idx）は使わない
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return ColmapPoints(np.array(xyz, np.float32).reshape(-1, 3), np.array(rgb, np.uint8).reshape(-1, 3))


def load_views(sparse_dir: Path) -> list[ColmapView]:
    """画像名順の視点リスト（元解像度）。"""
    cameras = read_cameras(sparse_dir / "cameras.bin")
    images = sorted(read_images(sparse_dir / "images.bin").values(), key=lambda i: i.name)
    return [
        ColmapView(i.name, Camera(cameras[i.camera_id].K(), i.world_to_cam(),
                                  cameras[i.camera_id].width, cameras[i.camera_id].height))
        for i in images
    ]
