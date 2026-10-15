import numpy as np
from plyfile import PlyData, PlyElement

from spatialsam.spatial.gaussians import save_ply
from spatialsam.spatial.pointcloud import (
    SH_C0,
    clip_outlier_scales,
    gaussians_from_points,
    is_gaussian_ply,
    load_scene,
    neighbor_distances,
    rgb_to_sh0,
)


def write_points(path, with_color=True):
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")] + ([("red", "u1"), ("green", "u1"), ("blue", "u1")] if with_color else [])
    data = np.zeros(3, dtype=fields)
    data["x"] = [0, 1, 3]
    if with_color:
        data["red"] = 255
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))
    return path


def test_rgb_to_sh0_roundtrip():
    rgb = np.array([[0, 128, 255]], np.uint8)
    assert np.allclose(rgb_to_sh0(rgb) * SH_C0 + 0.5, rgb / 255.0)


def test_neighbor_distances():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], np.float32)
    assert np.allclose(neighbor_distances(xyz, k=1), [1, 1, 2])
    assert neighbor_distances(np.zeros((1, 3))).shape == (1,)
    assert (neighbor_distances(np.zeros((2, 3))) > 0).all()  # 重複点でも 0 にしない


def test_clip_outlier_scales():
    scales = np.array([1.0] * 99 + [100.0])
    assert clip_outlier_scales(scales).max() < 2.0
    assert clip_outlier_scales(np.array([])).size == 0


def test_gaussians_from_points():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], np.float32)
    scene = gaussians_from_points(xyz, np.full((3, 3), 255, np.uint8), opacity=0.5)
    assert len(scene) == 3 and scene.sh_degree == 0
    assert np.allclose(scene.opacity_logits, 0)
    assert np.allclose(scene.log_scales[:, 0], scene.log_scales[:, 2])


def test_load_scene_dispatches_by_format(tmp_path):
    points = load_scene(write_points(tmp_path / "p.ply"))
    assert not is_gaussian_ply(tmp_path / "p.ply")
    assert np.allclose(points.sh[:, 0, 0] * SH_C0 + 0.5, 1.0)
    gray = load_scene(write_points(tmp_path / "g.ply", with_color=False))
    assert np.allclose(gray.sh[:, 0] * SH_C0 + 0.5, 128 / 255)
    save_ply(points, tmp_path / "gs.ply")
    assert is_gaussian_ply(tmp_path / "gs.ply")
    assert len(load_scene(tmp_path / "gs.ply")) == 3
