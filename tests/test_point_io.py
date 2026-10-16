import numpy as np
import pytest

pytest.importorskip("open3d")

from spatialsam.spatial.point_io import (  # noqa: E402
    GRAY,
    PointCloudFilter,
    filter_points,
    read_point_cloud,
    write_point_cloud,
)
from spatialsam.spatial.pointcloud import SH_C0, load_scene  # noqa: E402
from spatialsam.spatial.scene_source import SceneSource  # noqa: E402


def cloud(n=200, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(scale=0.5, size=(n, 3)).astype(np.float32)
    rgb = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
    return xyz, rgb


@pytest.mark.parametrize("suffix", [".ply", ".pcd", ".xyzrgb"])
def test_roundtrip_formats(tmp_path, suffix):
    xyz, rgb = cloud()
    loaded_xyz, loaded_rgb = read_point_cloud(write_point_cloud(tmp_path / f"c{suffix}", xyz, rgb))
    assert np.allclose(loaded_xyz, xyz, atol=1e-5)
    assert np.abs(loaded_rgb.astype(int) - rgb).max() <= 1


def test_points_without_color_become_gray(tmp_path):
    path = tmp_path / "c.xyz"
    path.write_text("0 0 0\n1 0 0\n0 1 0\n")
    xyz, rgb = read_point_cloud(path)
    assert xyz.shape == (3, 3) and (rgb == GRAY).all()


def test_unsupported_and_empty(tmp_path):
    with pytest.raises(ValueError):
        read_point_cloud(tmp_path / "c.obj")
    empty = tmp_path / "e.xyz"
    empty.write_text("")
    with pytest.raises(ValueError):
        read_point_cloud(empty)


def test_statistical_outlier_removal_drops_far_points():
    xyz, rgb = cloud()
    far = np.array([[50.0, 50.0, 50.0], [-60.0, 40.0, 0.0]], np.float32)
    out_xyz, out_rgb, stats = filter_points(np.vstack([xyz, far]), np.vstack([rgb, np.zeros((2, 3), np.uint8)]),
                                            PointCloudFilter())
    assert stats.before == 202 and stats.after_outliers <= 200
    assert np.abs(out_xyz).max() < 10 and len(out_rgb) == len(out_xyz)


def test_voxel_downsampling_reduces_points():
    xyz = np.repeat(np.array([[0.0, 0, 0], [1.0, 0, 0]], np.float32), 50, axis=0)
    rgb = np.full((100, 3), 200, np.uint8)
    out_xyz, out_rgb, stats = filter_points(xyz, rgb, PointCloudFilter(remove_outliers=False, voxel_size=0.5))
    assert stats.after_voxel == 2 and len(out_xyz) == 2 and (out_rgb == 200).all()


def test_scene_source_applies_filter_and_roundtrips(tmp_path):
    xyz, rgb = cloud()
    path = write_point_cloud(tmp_path / "c.pcd", np.vstack([xyz, [[80.0, 0, 0]]]), np.vstack([rgb, [[0, 0, 0]]]))
    plain = SceneSource(path).load_scene()
    filtered_source = SceneSource(path, point_filter=PointCloudFilter())
    filtered = filtered_source.load_scene()
    assert len(plain) == 201 and len(filtered) < 201 and np.abs(filtered.means).max() < 10
    restored = SceneSource.from_dict(filtered_source.to_dict())
    assert restored.point_filter == PointCloudFilter() and restored.ply_path == path
    assert np.allclose(load_scene(path).sh[:, 0] * SH_C0 + 0.5, plain.sh[:, 0] * SH_C0 + 0.5)
