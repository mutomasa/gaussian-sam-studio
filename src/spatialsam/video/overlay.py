"""マスクの色付け・重畳表示。"""

from __future__ import annotations

import cv2
import numpy as np

from spatialsam.models.types import FrameMasks

# 視認性の高い固定パレット（BGR）
PALETTE: tuple[tuple[int, int, int], ...] = (
    (75, 25, 230),
    (75, 180, 60),
    (25, 225, 255),
    (200, 130, 0),
    (48, 130, 245),
    (180, 30, 145),
    (240, 240, 70),
    (230, 50, 240),
)


def color_for_id(obj_id: int) -> tuple[int, int, int]:
    return PALETTE[obj_id % len(PALETTE)]


def blend_mask(
    frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.5
) -> np.ndarray:
    out = frame.copy()
    region = mask.astype(bool)
    tint = np.array(color, dtype=np.float32)
    out[region] = (out[region] * (1 - alpha) + tint * alpha).astype(np.uint8)
    return out


def draw_contour(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(frame.copy(), contours, -1, color, thickness=2)


def overlay_masks(frame: np.ndarray, frame_masks: FrameMasks | None, alpha: float = 0.5) -> np.ndarray:
    if frame_masks is None:
        return frame
    out = frame
    for obj_id, mask in sorted(frame_masks.masks.items()):
        color = color_for_id(obj_id)
        out = draw_contour(blend_mask(out, mask, color, alpha), mask, color)
    return out


def render_mask_only(shape: tuple[int, int], frame_masks: FrameMasks | None) -> np.ndarray:
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    if frame_masks is None:
        return canvas
    for obj_id, mask in sorted(frame_masks.masks.items()):
        canvas[mask.astype(bool)] = color_for_id(obj_id)
    return canvas
