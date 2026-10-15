import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spatialsam.spatial.camera import Camera, intrinsics, look_at  # noqa: E402
from spatialsam.spatial.colmap import ColmapView  # noqa: E402
from spatialsam.training import run_job  # noqa: E402
from spatialsam.training.dataset import ColmapDataset, TrainView, split_views  # noqa: E402
from spatialsam.training.jobs import DONE, FAILED, QUEUED, TrainingStore, TrainRequest, pid_alive  # noqa: E402
from spatialsam.training.losses import photometric_loss, psnr, ssim  # noqa: E402
from spatialsam.training.trainer import (  # noqa: E402
    TrainConfig,
    create_optimizers,
    export_scene,
    init_params,
    learning_rates,
)

CUDA = torch.cuda.is_available()


def test_split_views_holds_out_every_eighth():
    views = [ColmapView(f"{i:03d}.jpg", None) for i in range(17)]
    train, test = split_views(views)
    assert [v.name for v in test] == ["000.jpg", "008.jpg", "016.jpg"]
    assert len(train) == 14
    assert split_views(views, 0) == (views, [])


def test_losses():
    a = torch.rand(32, 32, 3)
    assert ssim(a, a).item() == pytest.approx(1.0, abs=1e-5)
    assert psnr(a, a) == float("inf")
    assert psnr(torch.zeros(4, 4, 3), torch.full((4, 4, 3), 0.1)) == pytest.approx(20.0)
    assert photometric_loss(a, a).item() == pytest.approx(0.0, abs=1e-5)
    assert photometric_loss(a, 1 - a).item() > 0.1


def test_init_params_and_export():
    xyz = np.random.default_rng(0).normal(size=(20, 3)).astype(np.float32)
    rgb = np.full((20, 3), 200, np.uint8)
    config = TrainConfig(sh_degree=2)
    params = init_params(xyz, rgb, config, "cpu")
    assert params["shN"].shape == (20, 8, 3) and params["opacities"].shape == (20,)
    assert torch.sigmoid(params["opacities"]).allclose(torch.full((20,), 0.5))
    scene = export_scene(params)
    assert scene.sh_degree == 2 and np.allclose(scene.means, xyz)
    optimizers = create_optimizers(params, learning_rates(config, scene_scale=2.0))
    assert set(optimizers) == set(params)
    assert optimizers["means"].param_groups[0]["lr"] == pytest.approx(config.means_lr * 2.0)


def synthetic_dataset(num_views=8, size=(48, 64)):
    """既知の Gaussian 群を周囲から描いた画像を正解にする。"""
    from spatialsam.spatial.pointcloud import gaussians_from_points
    from spatialsam.spatial.renderer import GaussianRenderer

    rng = np.random.default_rng(0)
    xyz = rng.uniform(-1, 1, size=(400, 3)).astype(np.float32)
    rgb = rng.integers(0, 255, size=(400, 3)).astype(np.uint8)
    renderer = GaussianRenderer(gaussians_from_points(xyz, rgb))
    height, width = size
    views = []
    for i, angle in enumerate(np.linspace(0, 2 * np.pi, num_views, endpoint=False)):
        eye = np.array([4 * np.cos(angle), 4 * np.sin(angle), 1.0])
        camera = Camera(intrinsics(60, 60, width / 2, height / 2), look_at(eye, np.zeros(3), np.array([0, 0, 1.0])),
                        width, height)
        views.append(TrainView(f"{i}.png", camera, renderer.render(camera).rgb))
    return ColmapDataset(views[1:], views[:1]), xyz, rgb


@pytest.mark.skipif(not CUDA, reason="CUDA が無い")
def test_trainer_reduces_loss():
    from spatialsam.training.trainer import Trainer

    dataset, xyz, rgb = synthetic_dataset()
    noisy = xyz + np.random.default_rng(1).normal(scale=0.05, size=xyz.shape).astype(np.float32)
    trainer = Trainer(dataset, noisy, rgb, TrainConfig(steps=500, cap_max=600, refine_start=50, refine_every=50))
    history = []
    trainer.fit(history.append, report_every=25)
    assert history[-1].step == 500
    assert np.mean([p.loss for p in history[-3:]]) < np.mean([p.loss for p in history[:3]])
    assert history[-1].num_gaussians <= 600
    metrics = trainer.evaluate(dataset.test)
    assert metrics["num_views"] == 1 and metrics["psnr"] > 18  # 学習前は約 9.4dB


def test_store_and_status(tmp_path):
    store = TrainingStore(tmp_path)
    job = store.create(TrainRequest("/s", "/i", {"steps": 10}))
    assert store.list_ids() == [job.id]
    assert job.load_request().config == {"steps": 10}
    assert job.status()["state"] == QUEUED
    assert not job.is_active()  # PID が無い = 起動していない
    job.update_status(pid=os.getpid())
    assert job.is_active() and store.active_jobs()[0].id == job.id
    job.update_status(state=DONE)
    assert not job.is_active()


def test_pid_alive():
    assert pid_alive(os.getpid())
    assert not pid_alive(None)
    assert not pid_alive(2**22 + 12345)


def test_run_job_records_failure(tmp_path):
    job = TrainingStore(tmp_path).create(TrainRequest(str(tmp_path / "missing"), str(tmp_path / "images")))
    assert run_job.main([str(job.root)]) == 1
    status = job.status()
    assert status["state"] == FAILED and "FileNotFoundError" in status["message"]
    assert not (job.root / "status.json.tmp").exists()
    assert json.loads(Path(job.root / "request.json").read_text())["sparse_dir"].endswith("missing")
