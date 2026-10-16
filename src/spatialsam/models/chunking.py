"""長い動画をチャンクに分けて推論し、物体 ID をチャンク間でつなぐ #9。

- 隣り合うチャンクは `overlap` フレーム重ねる
- 各チャンクは別セッションで推論するので、物体 ID はチャンクごとにばらばらになる
- 重なりフレームで「前のチャンクの物体」と「次のチャンクの物体」のマスク IoU（全重なりフレームの
  交差面積の和 / 和集合面積の和）を求め、ハンガリアン法で 1 対 1 に対応付けて前の ID を引き継ぐ
- 対応の取れない物体には新しい ID を振る。重なりフレームの結果は追跡の履歴が長い前のチャンクを使う
  （前のチャンクにマスクが無いフレームだけ次のチャンクの結果で埋める）
- SAM 3.1 は新しいセッションで物体を出力するまでに数〜数十フレームかかる（連続検出による確認）。
  重なりがこれより短いと対応付けられず ID が途切れるので、重なりは長めにする（既定 48）

モデルに依存しない（VideoSegmenter を包むだけ）ので、SAM 3.1 以外のセグメンタにも使える。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spatialsam.models.segmenter import VideoSegmenter
from spatialsam.models.types import FrameMasks, SegmentationRequest, SegmentationResult
from spatialsam.video.io import frame_path, list_frame_files


@dataclass(frozen=True)
class ChunkSettings:
    chunk_size: int = 200
    overlap: int = 48
    match_iou: float = 0.3  # これ未満の対応は別物体とみなす

    def __post_init__(self) -> None:
        if self.chunk_size < 2 or not 0 < self.overlap < self.chunk_size:
            raise ValueError("chunk_size >= 2 かつ 0 < overlap < chunk_size にしてください")


def chunk_ranges(num_frames: int, settings: ChunkSettings) -> list[range]:
    """[start, stop) の連続区間。隣り合う区間は overlap フレーム重なり、最後の区間は末尾まで。"""
    if num_frames <= settings.chunk_size:
        return [range(num_frames)]
    step = settings.chunk_size - settings.overlap
    ranges, start = [], 0
    while start + settings.overlap < num_frames:
        ranges.append(range(start, min(start + settings.chunk_size, num_frames)))
        if ranges[-1].stop == num_frames:
            break
        start += step
    return ranges


def overlap_iou(previous: dict[int, FrameMasks], following: dict[int, FrameMasks],
                frames: list[int]) -> tuple[list[int], list[int], np.ndarray]:
    """重なりフレームでの物体同士の IoU 行列（行 = 前の物体 ID、列 = 次の物体 ID）。"""
    prev_ids = sorted({i for f in frames if f in previous for i in previous[f].masks})
    next_ids = sorted({i for f in frames if f in following for i in following[f].masks})
    inter = np.zeros((len(prev_ids), len(next_ids)))
    union = np.zeros_like(inter)
    for f in frames:
        a = previous.get(f, FrameMasks(f, {})).masks
        b = following.get(f, FrameMasks(f, {})).masks
        for r, pid in enumerate(prev_ids):
            for c, nid in enumerate(next_ids):
                ma, mb = a.get(pid), b.get(nid)
                if ma is None and mb is None:
                    continue
                if ma is None or mb is None:
                    union[r, c] += (ma if mb is None else mb).sum()
                    continue
                inter[r, c] += np.logical_and(ma, mb).sum()
                union[r, c] += np.logical_or(ma, mb).sum()
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return prev_ids, next_ids, iou


def match_ids(prev_ids: list[int], next_ids: list[int], iou: np.ndarray, threshold: float) -> dict[int, int]:
    """次のチャンクの物体 ID → 前のチャンクの物体 ID（IoU が threshold 以上の 1 対 1 対応）。"""
    from scipy.optimize import linear_sum_assignment

    if iou.size == 0:
        return {}
    rows, cols = linear_sum_assignment(-iou)
    return {next_ids[c]: prev_ids[r] for r, c in zip(rows, cols, strict=True) if iou[r, c] >= threshold}


def relabel(frame: FrameMasks, mapping: dict[int, int]) -> FrameMasks:
    return FrameMasks(frame.frame_index, {mapping[k]: v for k, v in frame.masks.items()},
                      {mapping[k]: v for k, v in frame.scores.items()})


class Stitcher:
    """チャンクの結果を順に受け取り、全体の ID でつないだ結果を作る。"""

    def __init__(self, settings: ChunkSettings) -> None:
        self.settings = settings
        self.frames: dict[int, FrameMasks] = {}
        self.covered: set[int] = set()
        self.matches: list[dict[str, Any]] = []
        self._next_id = 0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def _mapping(self, chunk: dict[int, FrameMasks], overlap: list[int]) -> dict[int, int]:
        local_ids = sorted({i for f in chunk.values() for i in f.masks})
        if not self.covered:  # 最初のチャンクは ID をそのまま使う
            self._next_id = max(local_ids, default=-1) + 1
            return {i: i for i in local_ids}
        prev_ids, next_ids, iou = overlap_iou(self.frames, chunk, overlap)
        matched = match_ids(prev_ids, next_ids, iou, self.settings.match_iou)
        mapping = matched | {i: self._new_id() for i in local_ids if i not in matched}
        self.matches.append({"overlap_frames": len(overlap), "matched": len(matched),
                             "new_ids": len(mapping) - len(matched),
                             "iou": [round(float(iou[prev_ids.index(p), next_ids.index(n)]), 4)
                                     for n, p in matched.items()]})
        return mapping

    def add(self, chunk: dict[int, FrameMasks], positions: range) -> None:
        """重なりフレームは前のチャンクの結果を残し（空なら次で埋める）、それ以外は ID を付け替えて追加する。"""
        mapping = self._mapping(chunk, [f for f in positions if f in self.covered])
        for index, frame in chunk.items():
            previous = self.frames.get(index)
            if index not in self.covered or (previous is None or not previous.masks) and frame.masks:
                self.frames[index] = relabel(frame, mapping)
        self.covered.update(positions)

    def result(self) -> SegmentationResult:
        return SegmentationResult([self.frames[i] for i in sorted(self.frames)])


@contextmanager
def chunk_frames_dir(frames: list[Path], positions: range) -> Iterator[Path]:
    """チャンクのフレームだけを 00000.jpg から並べた一時ディレクトリ（シンボリックリンク、無理ならコピー）。"""
    directory = Path(tempfile.mkdtemp(prefix="spatialsam-chunk-"))
    try:
        for local, position in enumerate(positions):
            target = frame_path(directory, local)
            try:
                os.symlink(frames[position].resolve(), target)
            except OSError:
                shutil.copy2(frames[position], target)
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


class ChunkedSegmenter:
    """VideoSegmenter を包み、長い入力をチャンクに分けて推論する。短い入力はそのまま渡す。"""

    def __init__(self, inner: VideoSegmenter, settings: ChunkSettings = ChunkSettings()) -> None:
        self._inner = inner
        self.settings = settings
        self.name = inner.name
        self.info: dict[str, Any] = {}

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        frames = list_frame_files(request.frames_dir)
        ranges = chunk_ranges(len(frames), self.settings)
        if len(ranges) == 1:
            self.info = {"chunks": 1}
            return self._inner.segment(request)
        stitcher = Stitcher(self.settings)
        for positions in ranges:
            with chunk_frames_dir(frames, positions) as directory:
                local = self._inner.segment(SegmentationRequest(directory, request.prompt))
            stitcher.add({positions[f.frame_index]: f.with_index(positions[f.frame_index]) for f in local.frames},
                         positions)
        self.info = {"chunks": len(ranges), "chunk_size": self.settings.chunk_size,
                     "overlap": self.settings.overlap, "matches": stitcher.matches}
        return stitcher.result()

    def release(self) -> None:
        self._inner.release()
