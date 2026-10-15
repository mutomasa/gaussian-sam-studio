"""動画セグメンテーションモデルの共通インターフェース。"""

from __future__ import annotations

from typing import Protocol

from spatialsam.models.types import SegmentationRequest, SegmentationResult


class VideoSegmenter(Protocol):
    name: str

    def segment(self, request: SegmentationRequest) -> SegmentationResult: ...

    def release(self) -> None:
        """GPU上の重みなどを解放する。"""
        ...
