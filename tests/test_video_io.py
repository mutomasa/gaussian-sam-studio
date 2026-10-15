import numpy as np
import pytest

from tests.conftest import FRAME_SHAPE
from spatialsam.video.io import (
    FrameSampling,
    extract_frames,
    iter_sampled_frames,
    list_frame_files,
    read_video_info,
    resize_to_max_side,
)


def test_read_video_info(synthetic_video):
    info = read_video_info(synthetic_video)
    assert (info.height, info.width) == FRAME_SHAPE
    assert info.frame_count == 12
    assert info.fps == pytest.approx(12)


def test_sampling_stride_and_max_frames(synthetic_video):
    indices = [i for i, _ in iter_sampled_frames(synthetic_video, FrameSampling(stride=3, max_frames=3))]
    assert indices == [0, 3, 6]


def test_sampling_rejects_invalid_stride():
    with pytest.raises(ValueError):
        FrameSampling(stride=0)


def test_resize_to_max_side_keeps_aspect():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert resize_to_max_side(frame, 50).shape == (25, 50, 3)
    assert resize_to_max_side(frame, None) is frame
    assert resize_to_max_side(frame, 400) is frame


def test_extract_frames_writes_sequential_jpegs(synthetic_video, tmp_path):
    out = tmp_path / "frames"
    indices = extract_frames(synthetic_video, FrameSampling(stride=2), out)
    files = list_frame_files(out)
    assert indices == [0, 2, 4, 6, 8, 10]
    assert [f.name for f in files] == [f"{i:05d}.jpg" for i in range(6)]


def test_read_missing_video_raises(tmp_path):
    with pytest.raises(ValueError):
        read_video_info(tmp_path / "missing.mp4")
