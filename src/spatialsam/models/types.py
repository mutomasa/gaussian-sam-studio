"""セグメンテーション入出力のモデル非依存なデータ型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SegmentationRequest:
    """フレーム連番ディレクトリ（00000.jpg ...）とテキストプロンプト。"""

    frames_dir: Path
    prompt: str


@dataclass(frozen=True)
class FrameMasks:
    """1フレーム分の物体ID→2値マスク（H×W, bool）とスコア。"""

    frame_index: int
    masks: dict[int, np.ndarray]
    scores: dict[int, float] = field(default_factory=dict)

    def union(self, shape: tuple[int, int]) -> np.ndarray:
        """全物体のマスクを合成した前景マスクを返す。"""
        merged = np.zeros(shape, dtype=bool)
        for mask in self.masks.values():
            merged |= mask.astype(bool)
        return merged

    def with_index(self, frame_index: int) -> FrameMasks:
        return FrameMasks(frame_index, self.masks, self.scores)


@dataclass(frozen=True)
class SegmentationResult:
    frames: list[FrameMasks]

    def object_ids(self) -> list[int]:
        ids = {obj_id for frame in self.frames for obj_id in frame.masks}
        return sorted(ids)

    def by_frame(self) -> dict[int, FrameMasks]:
        return {frame.frame_index: frame for frame in self.frames}

    def remap_indices(self, indices: list[int]) -> SegmentationResult:
        """連番位置（0,1,2,...）を元動画のフレーム番号へ置き換える。"""
        return SegmentationResult([f.with_index(indices[f.frame_index]) for f in self.frames])
