import numpy as np
import pytest

from spatialsam.spatial.gaussians import save_ply
from spatialsam.spatial.scene_source import SceneSource, default_orbit, orbit_from_bounds
from tests.test_colmap import write_sparse
from tests.test_spatial import make_scene


def test_requires_some_input(tmp_path):
    with pytest.raises(ValueError):
        SceneSource()
    with pytest.raises(FileNotFoundError):
        SceneSource(ply_path=tmp_path / "missing.ply")


def test_colmap_only_uses_points_and_cameras(tmp_path):
    source = SceneSource(sparse_dir=write_sparse(tmp_path / "sparse"))
    assert source.has_cameras and len(source.load_scene()) == 2
    assert [v.name for v in source.load_views()] == ["a.jpg", "b.jpg"]


def test_ply_only_orbits_scene_bounds(tmp_path):
    source = SceneSource(ply_path=save_ply(make_scene(n=50), tmp_path / "s.ply"))
    assert not source.has_cameras and source.load_views() == []
    spec = default_orbit(source.load_scene(), [], up_axis="-y", num_frames=7)
    assert np.allclose(spec.up, [0, -1, 0]) and spec.num_frames == 7 and spec.radius > 0


def test_orbit_from_bounds_centers_scene():
    scene = make_scene(n=200)
    shifted = type(scene)(scene.means + 10, scene.quats, scene.log_scales, scene.opacity_logits, scene.sh)
    spec = orbit_from_bounds(shifted, np.array([0, 0, 1.0]))
    assert np.allclose(spec.center, 10, atol=1.0)
