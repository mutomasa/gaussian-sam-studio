"""深度マップの可視化。"""

from __future__ import annotations

import cv2
import numpy as np


def depth_range(depths: list[np.ndarray], valids: list[np.ndarray], low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    """全フレーム共通の表示範囲（外れ値を除いた分位点）。フレームごとに変えると色がちらつく。"""
    values = np.concatenate([d[v] for d, v in zip(depths, valids, strict=True)]) if depths else np.array([])
    if values.size == 0:
        return 0.0, 1.0
    near, far = np.percentile(values, [low, high])
    return float(near), float(max(far, near + 1e-6))


def colorize_depth(depth: np.ndarray, valid: np.ndarray, near: float, far: float) -> np.ndarray:
    """近い = 赤、遠い = 青（TURBO を反転）。無効画素は黒。BGR で返す。"""
    normalized = np.clip((depth - near) / (far - near), 0.0, 1.0)
    colored = cv2.applyColorMap(((1.0 - normalized) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored
