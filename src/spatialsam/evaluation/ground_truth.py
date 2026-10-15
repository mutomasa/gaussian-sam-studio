"""GTマスク（PNG連番）の読み込みと予測とのフレーム対応付け。

ファイル名末尾の数字を元動画のフレーム番号とみなす（例: 00012.png, frame_000012.png）。
画素値 > 0 を前景とする。
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np

_TRAILING_DIGITS = re.compile(r"(\d+)$")


def parse_frame_index(filename: str) -> int:
    match = _TRAILING_DIGITS.search(Path(filename).stem)
    if match is None:
        raise ValueError(f"ファイル名からフレーム番号を取得できません: {filename}")
    return int(match.group(1))


def decode_mask(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("マスク画像をデコードできません")
    return image > 0


def load_gt_dir(directory: Path) -> dict[int, np.ndarray]:
    return {parse_frame_index(p.name): decode_mask(p.read_bytes()) for p in sorted(directory.glob("*.png"))}


def load_gt_zip(data: bytes) -> dict[int, np.ndarray]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".png") and not n.startswith("__MACOSX")]
        return {parse_frame_index(n): decode_mask(archive.read(n)) for n in names}


def align_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """GTを予測マスクの解像度へ最近傍補間で合わせる。"""
    if mask.shape == shape:
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)
