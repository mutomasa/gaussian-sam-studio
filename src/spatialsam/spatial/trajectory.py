"""描画用のカメラ軌道。

- オービット: 注視点の周りを一周する（静的シーンの動画はカメラを動かして作る）
- データセット軌道: COLMAP の撮影カメラを順にたどる（実画像と同じ視点になるので比較に使える #16）
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spatialsam.spatial.camera import Camera, intrinsics_from_fov, look_at
from spatialsam.spatial.colmap import ColmapView


@dataclass(frozen=True, eq=False)
class OrbitSpec:
    center: np.ndarray
    up: np.ndarray
    radius: float
    elevation_deg: float = 20.0  # 注視点から見たカメラの仰角
    num_frames: int = 120
    fov_y_deg: float = 50.0
    width: int = 854
    height: int = 480
    start_deg: float = 0.0


@dataclass(frozen=True)
class OrbitBasis:
    """up に直交する 2 軸（オービット平面）。"""

    u: np.ndarray
    v: np.ndarray


def orbit_basis(up: np.ndarray) -> OrbitBasis:
    up = up / np.linalg.norm(up)
    helper = np.eye(3)[int(np.argmin(np.abs(up)))]  # up と最も平行でない軸
    u = np.cross(up, helper)
    u /= np.linalg.norm(u)
    return OrbitBasis(u, np.cross(up, u))


def orbit_eyes(spec: OrbitSpec) -> np.ndarray:
    up = spec.up / np.linalg.norm(spec.up)
    basis = orbit_basis(up)
    angles = np.radians(spec.start_deg + 360.0 * np.arange(spec.num_frames) / spec.num_frames)
    elevation = np.radians(spec.elevation_deg)
    horizontal = np.cos(elevation) * (np.outer(np.cos(angles), basis.u) + np.outer(np.sin(angles), basis.v))
    return spec.center + spec.radius * (horizontal + np.sin(elevation) * up)


def orbit_trajectory(spec: OrbitSpec) -> list[Camera]:
    K = intrinsics_from_fov(spec.fov_y_deg, spec.width, spec.height)
    return [Camera(K, look_at(eye, spec.center, spec.up), spec.width, spec.height) for eye in orbit_eyes(spec)]


def dataset_trajectory(views: Sequence[ColmapView], stride: int = 1, max_frames: int | None = None,
                       width: int | None = None, height: int | None = None) -> list[tuple[str, Camera]]:
    """撮影順（画像名順）に間引いたカメラ。width/height を指定すると縮小画像に合わせる。"""
    if stride < 1:
        raise ValueError("stride は 1 以上")
    selected = list(views)[::stride][:max_frames]
    return [(v.name, v.camera.resized(width, height) if width and height else v.camera) for v in selected]


def estimate_up(cameras: Sequence[Camera]) -> np.ndarray:
    """撮影カメラの「画像の上方向」（OpenCV では -y 軸）の平均を world の上とみなす。"""
    ups = np.array([-c.cam_to_world[:3, 1] for c in cameras])
    up = ups.mean(axis=0)
    return up / np.linalg.norm(up)


def estimate_look_center(cameras: Sequence[Camera]) -> np.ndarray:
    """全カメラの光軸に最も近い点（最小二乗）。360° 撮影なら被写体の位置になる。"""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for camera in cameras:
        origin = camera.position
        direction = camera.cam_to_world[:3, 2]
        projector = np.eye(3) - np.outer(direction, direction)
        A += projector
        b += projector @ origin
    return np.linalg.lstsq(A, b, rcond=None)[0]


def orbit_from_cameras(cameras: Sequence[Camera], **overrides) -> OrbitSpec:
    """撮影カメラの配置を真似たオービット（注視点・上方向・半径・仰角を推定）。"""
    center = estimate_look_center(cameras)
    up = estimate_up(cameras)
    offsets = np.array([c.position for c in cameras]) - center
    radius = float(np.linalg.norm(offsets, axis=1).mean())
    elevation = float(np.degrees(np.arcsin(np.clip((offsets @ up).mean() / radius, -1, 1))))
    params = {"center": center, "up": up, "radius": radius, "elevation_deg": elevation} | overrides
    return OrbitSpec(**params)


def even(value: float) -> int:
    return max(2, 2 * round(value / 2))


def fit_max_side(camera: Camera, max_side: int | None) -> Camera:
    """長辺が max_side を超えるときだけ縮小する（SAM の入力解像度に合わせる）。

    H.264（yuv420p）は奇数サイズを書けず動画だけ 1px 伸びて PNG・姿勢とずれるので、縦横を偶数にする。
    """
    longest = max(camera.width, camera.height)
    if not max_side or longest <= max_side:
        return camera
    factor = max_side / longest
    return camera.resized(even(camera.width * factor), even(camera.height * factor))


def orbit_summary(spec: OrbitSpec) -> dict:
    """metadata.json に残す軌道の設定。"""
    return {"type": "orbit", "center": spec.center.tolist(), "up": spec.up.tolist(), "radius": spec.radius,
            "elevation_deg": spec.elevation_deg, "num_frames": spec.num_frames, "fov_y_deg": spec.fov_y_deg,
            "width": spec.width, "height": spec.height, "start_deg": spec.start_deg}
