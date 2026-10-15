import numpy as np
import pytest

from spatialsam.spatial.camera import Camera, intrinsics, look_at
from spatialsam.spatial.colmap import ColmapView
from spatialsam.spatial.trajectory import (
    OrbitSpec,
    dataset_trajectory,
    estimate_look_center,
    estimate_up,
    orbit_eyes,
    orbit_from_cameras,
    orbit_trajectory,
)

UP = np.array([0.0, 0.0, 1.0])
CENTER = np.array([1.0, 2.0, 0.5])


def ring_cameras(n=12, radius=4.0, height=1.0):
    K = intrinsics(100, 100, 50, 40)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    eyes = CENTER + np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n, height)], axis=1)
    return [Camera(K, look_at(eye, CENTER, UP), 100, 80) for eye in eyes]


def test_orbit_keeps_radius_and_elevation():
    spec = OrbitSpec(CENTER, UP, radius=3.0, elevation_deg=30.0, num_frames=8)
    offsets = orbit_eyes(spec) - CENTER
    assert np.allclose(np.linalg.norm(offsets, axis=1), 3.0)
    assert np.allclose(offsets[:, 2], 3.0 * np.sin(np.radians(30)))
    assert not np.allclose(offsets[0], offsets[4])


def test_orbit_cameras_look_at_center():
    cameras = orbit_trajectory(OrbitSpec(CENTER, UP, radius=3.0, num_frames=5, width=64, height=48))
    assert len(cameras) == 5
    for camera in cameras:
        p = camera.world_to_cam @ np.append(CENTER, 1.0)
        assert np.allclose(p[:2], 0, atol=1e-9) and p[2] > 0


def test_estimates_from_ring_of_cameras():
    cameras = ring_cameras()
    assert np.allclose(estimate_look_center(cameras), CENTER, atol=1e-6)
    assert np.dot(estimate_up(cameras), UP) > 0.9
    spec = orbit_from_cameras(cameras, num_frames=10)
    assert spec.radius == pytest.approx(np.hypot(4.0, 1.0))
    assert spec.elevation_deg == pytest.approx(np.degrees(np.arctan2(1.0, 4.0)), abs=1.0)
    assert spec.num_frames == 10


def test_dataset_trajectory_strides_and_resizes():
    views = [ColmapView(f"{i:02d}.jpg", c) for i, c in enumerate(ring_cameras())]
    picked = dataset_trajectory(views, stride=3, max_frames=3, width=50, height=40)
    assert [name for name, _ in picked] == ["00.jpg", "03.jpg", "06.jpg"]
    assert picked[0][1].width == 50 and picked[0][1].K[0, 0] == 50
    with pytest.raises(ValueError):
        dataset_trajectory(views, stride=0)
    train_only = dataset_trajectory(views, exclude_holdout_every=4)
    assert [name for name, _ in train_only] == ["01.jpg", "02.jpg", "03.jpg", "05.jpg", "06.jpg", "07.jpg",
                                                "09.jpg", "10.jpg", "11.jpg"]
