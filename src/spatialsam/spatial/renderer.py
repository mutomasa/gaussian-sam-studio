"""gsplat による RGB / Depth 描画。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from spatialsam.spatial.camera import Camera
from spatialsam.spatial.gaussians import GaussianScene


@dataclass(frozen=True)
class RenderOutput:
    rgb: np.ndarray    # [H, W, 3] uint8（RGB 順）
    depth: np.ndarray  # [H, W] float32 カメラ z 方向の期待深度。何も無い画素は 0
    alpha: np.ndarray  # [H, W] float32

    def valid_depth(self, min_alpha: float = 0.5) -> np.ndarray:
        """不透明度が低い画素の深度は当てにならないので除く。"""
        return self.alpha >= min_alpha


@dataclass(frozen=True)
class RenderSettings:
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    near_plane: float = 0.01
    far_plane: float = 1e10


def to_rgb_uint8(colors: np.ndarray) -> np.ndarray:
    return (np.clip(colors, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def composite(rgb: np.ndarray, alpha: np.ndarray, background: tuple[float, float, float]) -> np.ndarray:
    """背景色を合成する。gsplat 1.5.3 は packed=True で backgrounds を渡すと形状チェックで落ちるため自前で行う。"""
    return rgb + (1.0 - alpha[..., None]) * np.asarray(background, np.float32)


def expected_depth(accumulated: np.ndarray, alpha: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """gsplat の "ED" は α で正規化済みだが、α≈0 の画素は 0 にそろえる。"""
    return np.where(alpha > eps, accumulated, 0.0).astype(np.float32)


class GaussianTensors:
    """GPU 上に置いた活性化済みパラメータ。同じシーンを何視点も描くので一度だけ転送する。"""

    def __init__(self, scene: GaussianScene, device: str = "cuda") -> None:
        import torch

        def tensor(values: np.ndarray) -> Any:
            return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32, device=device)

        self.device = device
        self.means = tensor(scene.means)
        self.quats = torch.nn.functional.normalize(tensor(scene.quats), dim=-1)
        self.scales = torch.exp(tensor(scene.log_scales))
        self.opacities = torch.sigmoid(tensor(scene.opacity_logits))
        self.sh = tensor(scene.sh)
        self.sh_degree = scene.sh_degree


Rasterizer = Callable[..., tuple[Any, Any, dict]]


def _default_rasterizer() -> Rasterizer:
    from spatialsam.spatial.cuda_env import import_gsplat

    return import_gsplat().rasterization


class GaussianRenderer:
    def __init__(self, scene: GaussianScene, settings: RenderSettings = RenderSettings(),
                 device: str = "cuda", rasterizer: Rasterizer | None = None) -> None:
        self._gaussians = GaussianTensors(scene, device)
        self._settings = settings
        self._rasterize = rasterizer or _default_rasterizer()

    def render(self, camera: Camera) -> RenderOutput:
        import torch

        g, s = self._gaussians, self._settings
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=g.device)[None]  # noqa: E731
        with torch.no_grad():
            colors, alphas, _ = self._rasterize(
                g.means, g.quats, g.scales, g.opacities, g.sh,
                tensor(camera.world_to_cam), tensor(camera.K), camera.width, camera.height,
                sh_degree=g.sh_degree, render_mode="RGB+ED",
                near_plane=s.near_plane, far_plane=s.far_plane,
            )
        image = colors[0].cpu().numpy()
        alpha = alphas[0, ..., 0].cpu().numpy()
        rgb = composite(image[..., :3], alpha, s.background)
        return RenderOutput(to_rgb_uint8(rgb), expected_depth(image[..., 3], alpha), alpha)
