"""ラベルを使ったシーンの加工（対象だけ抜き出す・対象以外を暗くする）。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from spatialsam.spatial.gaussians import GaussianScene
from spatialsam.spatial.pointcloud import SH_C0


def subset_scene(scene: GaussianScene, keep: np.ndarray) -> GaussianScene:
    return GaussianScene(scene.means[keep], scene.quats[keep], scene.log_scales[keep],
                         scene.opacity_logits[keep], scene.sh[keep])


def dim_others(scene: GaussianScene, highlight: np.ndarray, brightness: float = 0.25,
               tint: tuple[float, float, float] | None = (1.0, 0.25, 0.25), tint_strength: float = 0.35) -> GaussianScene:
    """対象以外を暗い灰色にし、対象には少し色を乗せる（対象の位置がひと目でわかるように）。"""
    sh = scene.sh.copy()
    colors = sh[:, 0] * SH_C0 + 0.5  # DC 成分 → [0, 1] の色
    gray = colors.mean(axis=1, keepdims=True) * brightness
    colors[~highlight] = gray[~highlight]
    sh[~highlight, 1:] = 0.0  # 視点による色の変化も消す
    if tint is not None:
        colors[highlight] = (1 - tint_strength) * colors[highlight] + tint_strength * np.asarray(tint)
    sh[:, 0] = (colors - 0.5) / SH_C0
    return replace(scene, sh=sh.astype(np.float32))
