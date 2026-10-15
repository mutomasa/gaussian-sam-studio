"""gsplat による 3DGS 学習（MCMC 戦略、8GB VRAM 向け）。

ハイパーパラメータは gsplat の examples/simple_trainer.py（MCMC 設定）に合わせている。
Gaussian 数は `cap_max` で頭打ちにし、`packed=True` で描画の中間バッファを抑える。
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from spatialsam.spatial.gaussians import GaussianScene
from spatialsam.spatial.pointcloud import logit, neighbor_distances, rgb_to_sh0
from spatialsam.training.dataset import ColmapDataset, TrainView
from spatialsam.training.losses import photometric_loss, psnr, ssim


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 7000
    cap_max: int = 500_000
    sh_degree: int = 3
    sh_degree_interval: int = 1000   # この step ごとに SH 次数を 1 つ上げる
    init_opacity: float = 0.5
    init_scale: float = 0.1          # 近傍距離に掛ける倍率
    ssim_weight: float = 0.2
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    means_lr: float = 1.6e-4         # scene_scale 倍して使う。終盤は 1/100 まで下げる
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    refine_start: int = 500
    refine_stop: int = 25_000
    refine_every: int = 100
    noise_lr: float = 5e5
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainProgress:
    step: int
    total: int
    loss: float
    psnr: float
    num_gaussians: int
    elapsed_sec: float


ProgressCallback = Callable[[TrainProgress], None]


def init_params(xyz: np.ndarray, rgb: np.ndarray, config: TrainConfig, device: str) -> torch.nn.ParameterDict:
    """点群から初期 Gaussian を作る（MCMC 設定: 不透明度 0.5、scale は近傍距離の 0.1 倍）。"""
    n = len(xyz)
    scales = np.log(neighbor_distances(xyz) * config.init_scale)
    num_rest = (config.sh_degree + 1) ** 2 - 1
    quats = torch.rand(n, 4)
    tensors = {
        "means": torch.as_tensor(xyz, dtype=torch.float32),
        "scales": torch.as_tensor(np.repeat(scales[:, None], 3, axis=1), dtype=torch.float32),
        "quats": quats,
        "opacities": torch.full((n,), logit(config.init_opacity)),
        "sh0": torch.as_tensor(rgb_to_sh0(rgb), dtype=torch.float32)[:, None, :],
        "shN": torch.zeros(n, num_rest, 3),
    }
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in tensors.items()})


def learning_rates(config: TrainConfig, scene_scale: float) -> dict[str, float]:
    return {"means": config.means_lr * scene_scale, "scales": config.scales_lr, "quats": config.quats_lr,
            "opacities": config.opacities_lr, "sh0": config.sh0_lr, "shN": config.shN_lr}


def create_optimizers(params: torch.nn.ParameterDict, lrs: dict[str, float]) -> dict[str, torch.optim.Optimizer]:
    """MCMC 戦略は 1 パラメータ 1 オプティマイザを前提にする。"""
    return {name: torch.optim.Adam([{"params": params[name], "lr": lrs[name], "name": name}], eps=1e-15)
            for name in params}


def export_scene(params: torch.nn.ParameterDict) -> GaussianScene:
    def numpy(name: str) -> np.ndarray:
        return params[name].detach().cpu().numpy()

    return GaussianScene(
        means=numpy("means"), quats=numpy("quats"), log_scales=numpy("scales"),
        opacity_logits=numpy("opacities"), sh=np.concatenate([numpy("sh0"), numpy("shN")], axis=1),
    )


def to_device_image(image: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(image, device=device).float() / 255.0


class Trainer:
    def __init__(self, dataset: ColmapDataset, xyz: np.ndarray, rgb: np.ndarray,
                 config: TrainConfig = TrainConfig(), device: str = "cuda", rasterizer: Callable | None = None) -> None:
        from spatialsam.spatial.cuda_env import import_gsplat

        gsplat = import_gsplat()  # JIT ビルド用の環境変数を設定してから import する
        self.dataset, self.config, self.device = dataset, config, device
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        self.params = init_params(xyz, rgb, config, device)
        self.optimizers = create_optimizers(self.params, learning_rates(config, dataset.scene_scale()))
        gamma = 0.01 ** (1.0 / max(1, config.steps))
        self.means_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizers["means"], gamma)
        self.strategy = gsplat.MCMCStrategy(cap_max=config.cap_max, noise_lr=config.noise_lr,
                                     refine_start_iter=config.refine_start, refine_stop_iter=config.refine_stop,
                                     refine_every=config.refine_every)
        self.strategy.check_sanity(self.params, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()
        self._rasterize = rasterizer or gsplat.rasterization

    def render(self, view: TrainView, sh_degree: int) -> tuple[torch.Tensor, dict]:
        p, camera = self.params, view.camera
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)[None]  # noqa: E731
        colors, _, info = self._rasterize(
            p["means"], torch.nn.functional.normalize(p["quats"], dim=-1), torch.exp(p["scales"]),
            torch.sigmoid(p["opacities"]), torch.cat([p["sh0"], p["shN"]], dim=1),
            tensor(camera.world_to_cam), tensor(camera.K), camera.width, camera.height,
            sh_degree=sh_degree, packed=True,
        )
        return colors[0].clamp(0.0, 1.0), info

    def regularization(self) -> torch.Tensor:
        """MCMC 論文の正則化: 不透明度と大きさを小さく保ち、不要な Gaussian を再配置しやすくする。"""
        opacity = torch.sigmoid(self.params["opacities"]).mean()
        scale = torch.exp(self.params["scales"]).mean()
        return self.config.opacity_reg * opacity + self.config.scale_reg * scale

    def train_step(self, step: int, view: TrainView) -> tuple[float, float]:
        target = to_device_image(view.image, self.device)
        rendered, info = self.render(view, min(step // self.config.sh_degree_interval, self.config.sh_degree))
        loss = photometric_loss(rendered, target, self.config.ssim_weight) + self.regularization()
        loss.backward()
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.means_scheduler.step()
        self.strategy.step_post_backward(self.params, self.optimizers, self.strategy_state, step, info,
                                         lr=self.means_scheduler.get_last_lr()[0])
        return loss.item(), psnr(rendered.detach(), target)

    def fit(self, progress: ProgressCallback | None = None, report_every: int = 100) -> None:
        start = time.perf_counter()
        order: list[TrainView] = []
        for step in range(self.config.steps):
            if not order:  # エポックごとに順番を混ぜる
                order = random.sample(self.dataset.train, len(self.dataset.train))
            loss, train_psnr = self.train_step(step, order.pop())
            last = step == self.config.steps - 1
            if progress and (step % report_every == 0 or last):
                progress(TrainProgress(step + 1, self.config.steps, loss, train_psnr, len(self.params["means"]),
                                       time.perf_counter() - start))

    @torch.no_grad()
    def evaluate(self, views: list[TrainView]) -> dict[str, float]:
        """評価用視点（学習に使っていない）の PSNR / SSIM の平均。"""
        if not views:
            return {}
        scores = [self._score(view) for view in views]
        return {"psnr": float(np.mean([s[0] for s in scores])), "ssim": float(np.mean([s[1] for s in scores])),
                "num_views": len(views)}

    def _score(self, view: TrainView) -> tuple[float, float]:
        rendered, _ = self.render(view, self.config.sh_degree)
        target = to_device_image(view.image, self.device)
        return psnr(rendered, target), ssim(rendered, target).item()

    def export(self) -> GaussianScene:
        return export_scene(self.params)

