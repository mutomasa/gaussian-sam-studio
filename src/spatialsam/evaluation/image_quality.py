"""描画画像の画質指標（実画像との PSNR / SSIM / LPIPS）。TorchMetrics で計算する #41。

- PSNR: 値域 [0, 1]
- SSIM: 11x11 ガウス窓、σ=1.5（3DGS 論文と同じ窓）。画像の端は窓が収まる範囲だけで平均する（TorchMetrics の標準）。
  学習の損失（training/losses.py）はゼロ詰めの 3DGS 実装のままなので、値はわずかに違う（bonsai で約 0.0014 低い）
- LPIPS: AlexNet 版（値が小さいほど似ている）。lpips パッケージと同じ重みで、値も一致する
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SSIM_KERNEL = 11
SSIM_SIGMA = 1.5


@dataclass(frozen=True)
class QualityScore:
    psnr: float
    ssim: float
    lpips: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_quality(scores: list[QualityScore]) -> dict[str, float]:
    if not scores:
        return {}
    summary = {"psnr": float(np.mean([s.psnr for s in scores])), "ssim": float(np.mean([s.ssim for s in scores]))}
    lpips_values = [s.lpips for s in scores if s.lpips is not None]
    if lpips_values:
        summary["lpips"] = float(np.mean(lpips_values))
    return summary


def to_nchw(image):
    """[H, W, 3]（値域 [0, 1] の torch.Tensor）→ [1, 3, H, W]。"""
    return image.permute(2, 0, 1)[None]


def psnr(rendered, target) -> float:
    """画像は [H, W, 3]、値域 [0, 1] の torch.Tensor。完全一致は inf。"""
    import torch
    from torchmetrics.functional.image import peak_signal_noise_ratio

    if torch.equal(rendered, target):
        return float("inf")
    return float(peak_signal_noise_ratio(to_nchw(rendered), to_nchw(target), data_range=1.0))


def ssim(rendered, target) -> float:
    from torchmetrics.functional.image import structural_similarity_index_measure

    return float(structural_similarity_index_measure(to_nchw(rendered), to_nchw(target), data_range=1.0,
                                                     kernel_size=SSIM_KERNEL, sigma=SSIM_SIGMA))


class ImageQualityScorer:
    """画像は [H, W, 3] uint8 RGB。LPIPS のモデルは最初の呼び出しで一度だけ読み込む。"""

    def __init__(self, device: str = "cuda", use_lpips: bool = True) -> None:
        self.device = device
        self.use_lpips = use_lpips
        self._lpips = None

    def _tensor(self, image: np.ndarray):
        import torch

        return torch.as_tensor(image, device=self.device).float() / 255.0

    def _lpips_model(self):
        if self._lpips is None:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            # normalize=True で [0, 1] の入力を受ける（内部で [-1, 1] にする）
            self._lpips = LearnedPerceptualImagePatchSimilarity("alex", normalize=True).to(self.device).eval()
        return self._lpips

    def lpips(self, rendered, target) -> float:
        import torch

        with torch.no_grad():
            return float(self._lpips_model()(to_nchw(rendered), to_nchw(target)))

    def score(self, rendered: np.ndarray, target: np.ndarray) -> QualityScore:
        if rendered.shape != target.shape:
            raise ValueError(f"画像サイズが違います: {rendered.shape} と {target.shape}")
        a, b = self._tensor(rendered), self._tensor(target)
        return QualityScore(psnr(a, b), ssim(a, b), self.lpips(a, b) if self.use_lpips else None)
