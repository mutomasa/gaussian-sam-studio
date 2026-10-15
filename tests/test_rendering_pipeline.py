import cv2
import numpy as np
import pytest

from spatialsam.experiments.renders import RenderStore
from spatialsam.pipeline.rendering import RenderJob, run_render
from spatialsam.spatial.camera import Camera, intrinsics
from spatialsam.spatial.renderer import RenderOutput
from spatialsam.spatial.trajectory import OrbitSpec, fit_max_side, orbit_summary
from spatialsam.video.depth import colorize_depth, depth_range
from spatialsam.video.io import read_video_info


class StubRenderer:
    """フレームごとに明るさと深度が変わる画像を返す。"""

    def __init__(self):
        self.calls = 0

    def render(self, camera):
        self.calls += 1
        shape = (camera.height, camera.width)
        rgb = np.full((*shape, 3), 10 * self.calls, np.uint8)
        alpha = np.ones(shape, np.float32)
        alpha[0, 0] = 0.0
        return RenderOutput(rgb, np.full(shape, float(self.calls), np.float32), alpha)


def cameras(n=3):
    return [(f"{i}.jpg", Camera(intrinsics(30, 30, 16, 12), np.eye(4), 32, 24)) for i in range(n)]


def test_run_render_writes_all_outputs(tmp_path):
    store = RenderStore(tmp_path)
    run = run_render(RenderJob(tmp_path / "scene.ply", cameras(), fps=10, trajectory={"type": "test"}),
                     StubRenderer(), store)
    assert store.list_ids() == [run.id]
    info = read_video_info(run.video_path)
    assert (info.frame_count, info.width, info.height) == (3, 32, 24)
    assert run.depth_video_path.exists()
    assert cv2.imread(str(run.rgb_path(2)))[5, 5].tolist() == [30, 30, 30]  # PNG は可逆
    depth, alpha = run.load_depth(1)
    assert depth[5, 5] == pytest.approx(2.0) and alpha[0, 0] == 0.0
    poses = run.load_poses()
    assert [name for name, _ in poses] == ["0.jpg", "1.jpg", "2.jpg"] and poses[0][1].width == 32
    meta = run.load_metadata()
    assert meta["num_frames"] == 3 and meta["resolution"] == [32, 24] and meta["trajectory"] == {"type": "test"}
    assert store.videos() == [run.video_path] and run.video_path.name == f"{run.id}.mp4"


def test_run_render_requires_frames(tmp_path):
    with pytest.raises(ValueError):
        run_render(RenderJob(tmp_path / "s.ply", []), StubRenderer(), RenderStore(tmp_path))


def test_depth_helpers():
    depths = [np.array([[1.0, 2.0], [3.0, 100.0]])]
    valids = [np.array([[True, True], [True, False]])]
    near, far = depth_range(depths, valids, 0, 100)
    assert (near, far) == (1.0, 3.0)
    colored = colorize_depth(depths[0], valids[0], near, far)
    assert colored.shape == (2, 2, 3) and colored[1, 1].tolist() == [0, 0, 0]
    assert depth_range([], []) == (0.0, 1.0)


def test_fit_max_side_and_summary():
    camera = Camera(intrinsics(100, 100, 400, 300), np.eye(4), 800, 600)
    small = fit_max_side(camera, 400)
    assert (small.width, small.height) == (400, 300) and small.K[0, 0] == 50
    odd = fit_max_side(Camera(intrinsics(100, 100, 50, 40), np.eye(4), 3118, 2078), 854)
    assert (odd.width, odd.height) == (854, 570)
    assert fit_max_side(camera, None) is camera and fit_max_side(camera, 1000) is camera
    summary = orbit_summary(OrbitSpec(np.zeros(3), np.array([0, 0, 1.0]), 2.0))
    assert summary["type"] == "orbit" and summary["center"] == [0, 0, 0]
