"""GPU上に同時に1つのモデルだけを保持するレジストリ（8GB VRAM向け）。"""

from __future__ import annotations

from collections.abc import Callable

from spatialsam.models.gpu import empty_cuda_cache
from spatialsam.models.segmenter import VideoSegmenter


class ModelRegistry:
    def __init__(self, clear_cache: Callable[[], None] = empty_cuda_cache) -> None:
        self._clear_cache = clear_cache
        self._key: str | None = None
        self._model: VideoSegmenter | None = None

    @property
    def loaded_key(self) -> str | None:
        return self._key

    def get(self, key: str, loader: Callable[[], VideoSegmenter]) -> VideoSegmenter:
        """`key` のモデルを返す。別のモデルが載っていれば先に解放する。"""
        if self._key != key:
            self.release()
            self._model = loader()
            self._key = key
        assert self._model is not None
        return self._model

    def release(self) -> None:
        if self._model is not None:
            self._model.release()
        self._model = None
        self._key = None
        self._clear_cache()
