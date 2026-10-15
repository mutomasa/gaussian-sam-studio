from pathlib import Path

import numpy as np
import pytest

from spatialsam.video.io import write_video

FRAME_SHAPE = (64, 96)


def square_frame(index: int, shape: tuple[int, int] = FRAME_SHAPE) -> np.ndarray:
    """黒背景に右へ移動する白い正方形（BGR）。"""
    frame = np.zeros((*shape, 3), np.uint8)
    x = 10 + 2 * index
    frame[20:40, x : x + 20] = 255
    return frame


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    return write_video((square_frame(i) for i in range(12)), tmp_path / "clip.mp4", fps=12)
