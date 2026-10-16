"""設定からセグメンタを生成する。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from spatialsam.config import AppConfig
from spatialsam.models.checkpoint import resolve_checkpoint
from spatialsam.models.chunking import ChunkedSegmenter
from spatialsam.models.fake_segmenter import FakeSegmenter
from spatialsam.models.sam31_segmenter import Sam31Segmenter, Sam31Settings, build_sam31_predictor
from spatialsam.models.segmenter import VideoSegmenter

SEGMENTER_NAMES = ("sam3.1", "fake")


def pinned_sam31_factory(config: AppConfig):
    """固定 revision の重みで predictor を作る関数（重みの解決はモデル構築時まで遅延）。"""

    def build(settings: Sam31Settings) -> Any:
        source = config.sam31_source
        path = resolve_checkpoint(settings.checkpoint_path, source.repo_id, source.revision)
        return build_sam31_predictor(replace(settings, checkpoint_path=path))

    return build


def create_base_segmenter(name: str, config: AppConfig) -> VideoSegmenter:
    if name == "sam3.1":
        return Sam31Segmenter(config.sam31, pinned_sam31_factory(config))
    if name == "fake":
        return FakeSegmenter()
    raise ValueError(f"未知のセグメンタ: {name}（候補: {', '.join(SEGMENTER_NAMES)}）")


def create_segmenter(name: str, config: AppConfig) -> VideoSegmenter:
    """設定でチャンク分割が有効なら、長い動画を分割して推論するよう包む。"""
    segmenter = create_base_segmenter(name, config)
    return ChunkedSegmenter(segmenter, config.chunking) if config.chunking else segmenter
