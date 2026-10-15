"""描画画像の画質指標（実画像との PSNR / SSIM / LPIPS）。

PSNR・SSIM は学習と同じ実装（training/losses.py）を使い、学習時の評価値と比べられるようにする。
LPIPS は 3DGS 論文と同じく AlexNet 版（値が小さいほど似ている）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


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
            import lpips

            self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        return self._lpips

    def lpips(self, rendered, target) -> float:
        import torch

        to_nchw = lambda t: t.permute(2, 0, 1)[None] * 2 - 1  # noqa: E731 — LPIPS は [-1, 1]
        with torch.no_grad():
            return float(self._lpips_model()(to_nchw(rendered), to_nchw(target)).item())

    def score(self, rendered: np.ndarray, target: np.ndarray) -> QualityScore:
        from spatialsam.training.losses import psnr, ssim

        if rendered.shape != target.shape:
            raise ValueError(f"画像サイズが違います: {rendered.shape} と {target.shape}")
        a, b = self._tensor(rendered), self._tensor(target)
        return QualityScore(psnr(a, b), float(ssim(a, b).item()), self.lpips(a, b) if self.use_lpips else None)
