import cv2
import numpy as np
import pytest

from spatialsam.experiments.renders import RenderStore
from spatialsam.experiments.segmentation3d import Segmentation3DStore
from spatialsam.experiments.store import ExperimentStore
from spatialsam.models.types import FrameMasks, SegmentationResult
from spatialsam.pipeline.lifting import masked_views, render_for_experiment, run_lifting, scene_source_of
from spatialsam.spatial.camera import Camera, intrinsics, look_at
from spatialsam.spatial.gaussians import GaussianScene, load_ply, save_ply
from spatialsam.spatial.mask_lifting import LiftSettings, project
from spatialsam.spatial.scene_highlight import dim_others, subset_scene
from spatialsam.spatial.scene_source import SceneSource

RENDER_SIZE = (40, 60)  # 描画の (H, W)
MASK_SIZE = (20, 30)    # SAM は半分に縮小して推論した想定


def tiny_scene():
    means = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], np.float32)  # 0 が対象、1 は上にある別物
    n = len(means)
    return GaussianScene(means, np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
                         np.full((n, 3), -3.0, np.float32), np.zeros(n, np.float32), np.zeros((n, 1, 3), np.float32))


def cameras(n=4):
    K = intrinsics(50, 50, RENDER_SIZE[1] / 2, RENDER_SIZE[0] / 2)
    eyes = [[5 * np.cos(a), 5 * np.sin(a), 0.5] for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    return [Camera(K, look_at(np.array(e), np.zeros(3), np.array([0, 0, 1.0])), RENDER_SIZE[1], RENDER_SIZE[0])
            for e in eyes]


def setup(tmp_path):
    ply = save_ply(tiny_scene(), tmp_path / "scene.ply")
    renders = RenderStore(tmp_path / "out")
    run = renders.create("render-test")
    cams = cameras()
    run.save_poses([("", c) for c in cams])
    for i, camera in enumerate(cams):
        uv, z = project(tiny_scene().means, camera)
        depth = np.zeros(RENDER_SIZE, np.float32)
        for (u, v), d in zip(uv.astype(int), z, strict=True):
            depth[v - 1:v + 2, u - 1:u + 2] = d
        run.save_depth(i, depth, (depth > 0).astype(np.float32))
    run.save_metadata({"scene": str(ply), "scene_source": SceneSource(ply).to_dict()})
    experiment = ExperimentStore(tmp_path / "out").create("exp")
    experiment.frames_dir.mkdir(parents=True)
    frames = []
    for i, camera in enumerate(cams):
        cv2.imwrite(str(experiment.frames_dir / f"{i:05d}.jpg"), np.zeros((*MASK_SIZE, 3), np.uint8))
        uv, _ = project(tiny_scene().means[:1], camera.resized(MASK_SIZE[1], MASK_SIZE[0]))
        mask = np.zeros(MASK_SIZE, bool)
        u, v = uv[0].astype(int)
        mask[v - 1:v + 2, u - 1:u + 2] = True
        frames.append(FrameMasks(i, {3: mask}, {3: 0.9}))
    experiment.save_prediction(SegmentationResult(frames))
    experiment.save_metadata({"prompt": "thing", "input": {"video": "render-test.mp4"},
                              "frame_indices": list(range(len(cams))), "object_ids": [3]})
    return renders, experiment


def test_finds_render_and_scene(tmp_path):
    renders, experiment = setup(tmp_path)
    run = render_for_experiment(experiment, renders)
    assert run.id == "render-test"
    assert scene_source_of(run).ply_path.name == "scene.ply"
    views = list(masked_views(experiment, run))
    assert len(views) == 4 and views[0].depth.shape == MASK_SIZE and views[0].camera.width == MASK_SIZE[1]


def test_render_for_non_render_video(tmp_path):
    renders, experiment = setup(tmp_path)
    experiment.save_metadata(experiment.load_metadata() | {"input": {"video": "blackswan.mp4"}})
    with pytest.raises(ValueError):
        render_for_experiment(experiment, renders)


def test_run_lifting_labels_target_only(tmp_path):
    renders, experiment = setup(tmp_path)
    store = Segmentation3DStore(tmp_path / "out")
    run = run_lifting(experiment, renders, store, LiftSettings(min_views=2))
    summary = run.load_summary()
    assert summary["render"] == "render-test" and summary["num_views"] == 4
    assert summary["objects"] == [{"object_id": 3, "num_gaussians": 1, "mean_confidence": 1.0}]
    assert run.load_labels()[3].tolist() == [True, False]
    assert len(load_ply(run.object_ply_path)) == 1
    pytest.importorskip("open3d")
    from spatialsam.spatial.point_io import read_point_cloud

    for suffix in (".ply", ".pcd"):
        points, _ = read_point_cloud(run.object_points_path(suffix))
        assert np.allclose(points, [[0.0, 0.0, 0.0]])
    assert store.list_ids() == [run.id]


def test_highlight_helpers():
    scene = tiny_scene()
    keep = np.array([True, False])
    assert len(subset_scene(scene, keep)) == 1
    dimmed = dim_others(scene, keep, brightness=0.0, tint=None)
    from spatialsam.spatial.pointcloud import SH_C0
    colors = dimmed.sh[:, 0] * SH_C0 + 0.5
    assert np.allclose(colors[1], 0.0) and np.allclose(colors[0], 0.5)
