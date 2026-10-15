"""設定からセグメンタを生成する。"""

from __future__ import annotations

from spatialsam.config import AppConfig
from spatialsam.models.fake_segmenter import FakeSegmenter
from spatialsam.models.sam31_segmenter import Sam31Segmenter
from spatialsam.models.segmenter import VideoSegmenter

SEGMENTER_NAMES = ("sam3.1", "fake")


def create_segmenter(name: str, config: AppConfig) -> VideoSegmenter:
    if name == "sam3.1":
        return Sam31Segmenter(config.sam31)
    if name == "fake":
        return FakeSegmenter()
    raise ValueError(f"未知のセグメンタ: {name}（候補: {', '.join(SEGMENTER_NAMES)}）")
