"""3D セグメンテーションの評価 #26。

- 3D IoU: Gaussian 単位のラベル同士（正解ラベルや別条件の結果との比較）
- 再投影 IoU: 3D のラベルを別の視点に描いたマスク（遮蔽込み）と、その視点の 2D マスクとの IoU。
  学習・持ち上げに使っていない視点の実画像で比べれば、3D の結果が新しい視点でも正しいかを測れる。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from spatialsam.evaluation.segmentation import iou
from spatialsam.spatial.camera import Camera


class FeatureRenderer(Protocol):
    def render_features(self, camera: Camera, features: np.ndarray) -> np.ndarray: ...


def iou_3d(predicted: np.ndarray, reference: np.ndarray) -> float:
    if predicted.shape != reference.shape:
        raise ValueError(f"Gaussian 数が違います: {predicted.shape} と {reference.shape}")
    return iou(predicted, reference)


def label_masks(renderer: FeatureRenderer, labels: np.ndarray, cameras: Sequence[Camera],
                threshold: float = 0.5) -> list[np.ndarray]:
    """ラベル（0/1）を α 合成し、対象が手前に見えている画素をマスクにする。"""
    features = labels.astype(np.float32)
    return [renderer.render_features(camera, features)[..., 0] >= threshold for camera in cameras]


def reprojection_ious(predicted: Sequence[np.ndarray], reference: Sequence[np.ndarray]) -> list[float | None]:
    """視点ごとの IoU。どちらも空の視点は None（比べる面積が無い）。"""
    return [None if not p.any() and not r.any() else iou(p, r)
            for p, r in zip(predicted, reference, strict=True)]


def summarize_reprojection(ious: Sequence[float | None], predicted: Sequence[np.ndarray],
                           reference: Sequence[np.ndarray]) -> dict[str, float | int | None]:
    """基準の 2D マスクが空なのに 3D 側に対象が写る視点（基準の見落としの可能性）は別に数える。"""
    values = [v for v in ious if v is not None]
    missed = [i for i, (p, r) in enumerate(zip(predicted, reference, strict=True)) if p.any() and not r.any()]
    detected = [v for i, v in enumerate(ious) if v is not None and i not in missed]
    return {"mean_iou": float(np.mean(values)) if values else None,
            "median_iou": float(np.median(values)) if values else None,
            "mean_iou_where_reference_detected": float(np.mean(detected)) if detected else None,
            "reference_empty_views": missed, "num_views": len(ious), "num_compared": len(values)}
