"""2D・動画セグメンテーションの評価指標（IoU, Dice, 輪郭F, J&F）。

前景は「全物体マスクの和集合」と GT 前景の比較とする（物体ID対応は扱わない）。
輪郭Fの許容距離は DAVIS と同様に画像対角長の 0.008 倍。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import cv2
import numpy as np

BOUNDARY_TOLERANCE_RATIO = 0.008


@dataclass(frozen=True)
class FrameScore:
    frame_index: int
    iou: float
    dice: float
    boundary_f: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _ratio(numerator: int, denominator: int, empty: float = 1.0) -> float:
    return empty if denominator == 0 else numerator / denominator


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    return _ratio(int((pred & gt).sum()), int((pred | gt).sum()))


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    return _ratio(2 * int((pred & gt).sum()), int(pred.sum() + gt.sum()))


def boundary_map(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (mask_u8 - eroded).astype(bool)


def boundary_tolerance(shape: tuple[int, ...], ratio: float = BOUNDARY_TOLERANCE_RATIO) -> int:
    return max(1, math.ceil(ratio * math.hypot(shape[0], shape[1])))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def boundary_f_measure(pred: np.ndarray, gt: np.ndarray, tolerance: int | None = None) -> float:
    pred_b, gt_b = boundary_map(pred), boundary_map(gt)
    if not pred_b.any() and not gt_b.any():
        return 1.0
    if not pred_b.any() or not gt_b.any():
        return 0.0
    radius = tolerance if tolerance is not None else boundary_tolerance(pred.shape)
    precision = (pred_b & _dilate(gt_b, radius)).sum() / pred_b.sum()
    recall = (gt_b & _dilate(pred_b, radius)).sum() / gt_b.sum()
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def evaluate_frame(frame_index: int, pred: np.ndarray, gt: np.ndarray) -> FrameScore:
    if pred.shape != gt.shape:
        raise ValueError(f"マスクの形状が一致しません: pred={pred.shape}, gt={gt.shape}")
    return FrameScore(frame_index, iou(pred, gt), dice(pred, gt), boundary_f_measure(pred, gt))


def summarize(scores: Sequence[FrameScore]) -> dict[str, float]:
    if not scores:
        raise ValueError("評価対象フレームがありません")
    j = float(np.mean([s.iou for s in scores]))
    f = float(np.mean([s.boundary_f for s in scores]))
    return {
        "J": j,
        "F": f,
        "J&F": (j + f) / 2,
        "Dice": float(np.mean([s.dice for s in scores])),
        "num_frames": float(len(scores)),
    }
