"""動画の読み出し（間引き・リサイズ）とフレーム連番・MP4の書き出し。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

FRAME_SUFFIX = ".jpg"


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class FrameSampling:
    stride: int = 1
    max_frames: int | None = None
    max_side: int | None = None  # 長辺の上限 [px]。None なら元解像度

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError("stride は1以上")

    def output_fps(self, source_fps: float) -> float:
        return source_fps / self.stride


def read_video_info(path: Path) -> VideoInfo:
    capture = _open(path)
    try:
        return VideoInfo(
            fps=capture.get(cv2.CAP_PROP_FPS) or 30.0,
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def resize_to_max_side(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    height, width = frame.shape[:2]
    if max_side is None or max(height, width) <= max_side:
        return frame
    scale = max_side / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def iter_sampled_frames(path: Path, sampling: FrameSampling) -> Iterator[tuple[int, np.ndarray]]:
    """(元動画のフレーム番号, BGR画像) を間引き・リサイズして返す。"""
    capture = _open(path)
    emitted = 0
    try:
        for index, frame in _iter_raw(capture):
            if sampling.max_frames is not None and emitted >= sampling.max_frames:
                break
            if index % sampling.stride == 0:
                emitted += 1
                yield index, resize_to_max_side(frame, sampling.max_side)
    finally:
        capture.release()


def extract_frames(path: Path, sampling: FrameSampling, out_dir: Path) -> list[int]:
    """フレームを `out_dir/00000.jpg` 形式で保存し、元のフレーム番号の列を返す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    indices: list[int] = []
    for position, (index, frame) in enumerate(iter_sampled_frames(path, sampling)):
        cv2.imwrite(str(frame_path(out_dir, position)), frame)
        indices.append(index)
    if not indices:
        raise ValueError(f"フレームを読み出せません: {path}")
    return indices


def frame_path(frames_dir: Path, position: int) -> Path:
    return frames_dir / f"{position:05d}{FRAME_SUFFIX}"


def list_frame_files(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob(f"*{FRAME_SUFFIX}"), key=lambda p: int(p.stem))


def write_video(frames_bgr: Iterable[np.ndarray], path: Path, fps: float) -> Path:
    """ブラウザで再生できる H.264 (yuv420p) の MP4 を書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=2
    )
    try:
        for frame in frames_bgr:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    return path


def _open(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"動画を開けません: {path}")
    return capture


def _iter_raw(capture: cv2.VideoCapture) -> Iterator[tuple[int, np.ndarray]]:
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield index, frame
        index += 1
