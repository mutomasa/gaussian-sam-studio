"""学習の損失（torch。画像は [H, W, 3] で値域 [0, 1]）。

SSIM は 3DGS（INRIA 実装）と同じゼロ詰めの 11x11 ガウス窓で、損失として微分する。
評価の PSNR / SSIM / LPIPS は evaluation/image_quality.py（TorchMetrics）で測る。
ここの psnr は学習中の進捗表示用。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5


def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """3DGS 論文と同じ 11x11 ガウス窓の SSIM（画像全体の平均）。"""
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    window = _gaussian_window(SSIM_WINDOW, SSIM_SIGMA, a.device)
    pad = SSIM_WINDOW // 2

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, window, padding=pad, groups=3)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x**2
    var_y = blur(y * y) - mu_y**2
    cov = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return value.mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def photometric_loss(rendered: torch.Tensor, target: torch.Tensor, ssim_weight: float = 0.2) -> torch.Tensor:
    """3DGS の標準損失: (1 - λ)·L1 + λ·(1 - SSIM)。"""
    l1 = (rendered - target).abs().mean()
    return (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim(rendered, target))
