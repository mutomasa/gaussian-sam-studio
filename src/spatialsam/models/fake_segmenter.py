"""GPU・重み無しでパイプラインとUIを動かすためのダミーモデル。

プロンプトは無視し、各フレーム中央の楕円を物体ID 1として返す。精度評価には使わない。
"""

from __future__ import annotations

import cv2
import numpy as np

from spatialsam.models.types import FrameMasks, SegmentationRequest, SegmentationResult
from spatialsam.video.io import list_frame_files


def center_ellipse_mask(shape: tuple[int, int], scale: float = 0.25) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    axes = (max(1, int(width * scale)), max(1, int(height * scale)))
    cv2.ellipse(mask, (width // 2, height // 2), axes, 0, 0, 360, 1, thickness=-1)
    return mask.astype(bool)


class FakeSegmenter:
    name = "fake"

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        files = list_frame_files(request.frames_dir)
        frames = [self._segment_frame(i, cv2.imread(str(path))) for i, path in enumerate(files)]
        return SegmentationResult(frames)

    def release(self) -> None:
        return None

    @staticmethod
    def _segment_frame(index: int, image: np.ndarray) -> FrameMasks:
        mask = center_ellipse_mask(image.shape[:2])
        return FrameMasks(index, {1: mask}, {1: 1.0})
