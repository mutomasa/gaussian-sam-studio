import cv2
import numpy as np
import pytest

from spatialsam.models.chunking import (
    ChunkedSegmenter,
    ChunkSettings,
    Stitcher,
    chunk_ranges,
    match_ids,
    overlap_iou,
)
from spatialsam.models.types import FrameMasks, SegmentationRequest
from spatialsam.video.io import frame_path, list_frame_files

SHAPE = (24, 40)


def square(position: int, offset: int = 0) -> np.ndarray:
    mask = np.zeros(SHAPE, bool)
    x = (position + offset) % 30
    mask[4:12, x:x + 8] = True
    return mask


class SessionFakeSegmenter:
    """全体の何フレーム目かをリンク先のファイル名から知り、セッションごとに違う ID を振る。"""

    name = "fake-session"

    def __init__(self, second_object_from=None):
        self.sessions = 0
        self.second_object_from = second_object_from
        self.chunk_lengths = []

    def segment(self, request):
        self.sessions += 1
        base = 100 * self.sessions  # セッションごとに ID が変わる
        frames = []
        files = list_frame_files(request.frames_dir)
        self.chunk_lengths.append(len(files))
        for local, path in enumerate(files):
            position = int(path.resolve().stem)
            masks = {base + 7: square(position)}
            if self.second_object_from is not None and position >= self.second_object_from:
                second = np.zeros(SHAPE, bool)
                second[16:22, 2:8] = True
                masks[base + 3] = second
            frames.append(FrameMasks(local, masks, {k: 0.9 for k in masks}))
        from spatialsam.models.types import SegmentationResult
        return SegmentationResult(frames)

    def release(self):
        pass


def write_frames(directory, count):
    directory.mkdir()
    for i in range(count):
        cv2.imwrite(str(frame_path(directory, i)), np.zeros((*SHAPE, 3), np.uint8))
    return directory


def test_chunk_ranges_cover_with_overlap():
    settings = ChunkSettings(chunk_size=10, overlap=3)
    ranges = chunk_ranges(25, settings)
    assert [(r.start, r.stop) for r in ranges] == [(0, 10), (7, 17), (14, 24), (21, 25)]
    assert chunk_ranges(10, settings) == [range(10)]
    assert [(r.start, r.stop) for r in chunk_ranges(17, settings)] == [(0, 10), (7, 17)]
    with pytest.raises(ValueError):
        ChunkSettings(chunk_size=5, overlap=5)


def test_overlap_iou_and_matching():
    a = {0: FrameMasks(0, {1: square(0), 2: square(0, 10)}), 1: FrameMasks(1, {1: square(1)})}
    b = {0: FrameMasks(0, {9: square(0, 10), 8: square(0)}), 1: FrameMasks(1, {8: square(1)})}
    prev_ids, next_ids, iou = overlap_iou(a, b, [0, 1])
    assert prev_ids == [1, 2] and next_ids == [8, 9]
    assert iou[0, 0] == pytest.approx(1.0) and iou[1, 1] == pytest.approx(1.0) and iou[0, 1] == 0
    assert match_ids(prev_ids, next_ids, iou, 0.3) == {8: 1, 9: 2}
    assert match_ids([1], [5], np.array([[0.1]]), 0.3) == {}
    assert match_ids([], [], np.zeros((0, 0)), 0.3) == {}


def test_chunked_segmenter_keeps_ids_across_chunks(tmp_path):
    frames_dir = write_frames(tmp_path / "frames", 25)
    inner = SessionFakeSegmenter(second_object_from=18)
    segmenter = ChunkedSegmenter(inner, ChunkSettings(chunk_size=10, overlap=3))
    result = segmenter.segment(SegmentationRequest(frames_dir, "thing"))
    assert inner.sessions == 4 and inner.chunk_lengths == [10, 10, 10, 4]
    assert [f.frame_index for f in result.frames] == list(range(25))
    assert result.object_ids() == [107, 108]  # 最初の物体は全チャンクで同じ ID、後から現れた物体は続きの新しい ID
    assert all(107 in f.masks for f in result.frames)
    assert 108 not in result.by_frame()[17].masks and 108 in result.by_frame()[18].masks
    assert np.array_equal(result.by_frame()[12].masks[107], square(12))
    assert segmenter.info["chunks"] == 4 and [m["matched"] for m in segmenter.info["matches"]] == [1, 1, 2]
    assert not any(p.name.startswith("spatialsam-chunk-") for p in tmp_path.iterdir())


def test_short_input_is_passed_through(tmp_path):
    frames_dir = write_frames(tmp_path / "frames", 5)
    inner = SessionFakeSegmenter()
    segmenter = ChunkedSegmenter(inner, ChunkSettings(chunk_size=10, overlap=3))
    result = segmenter.segment(SegmentationRequest(frames_dir, "thing"))
    assert inner.sessions == 1 and result.object_ids() == [107] and segmenter.info == {"chunks": 1}


def test_unmatched_objects_get_new_ids():
    stitcher = Stitcher(ChunkSettings(chunk_size=4, overlap=2, match_iou=0.5))
    stitcher.add({i: FrameMasks(i, {0: square(i)}) for i in range(4)}, range(0, 4))
    stitcher.add({i: FrameMasks(i, {5: square(i, 15)}) for i in range(2, 6)}, range(2, 6))  # 別の場所 → 別物体
    result = stitcher.result()
    assert result.object_ids() == [0, 1]
    assert list(result.by_frame()[3].masks) == [0] and list(result.by_frame()[4].masks) == [1]


def test_overlap_gap_in_previous_chunk_is_filled_by_next():
    stitcher = Stitcher(ChunkSettings(chunk_size=4, overlap=2))
    stitcher.add({0: FrameMasks(0, {0: square(0)}), 1: FrameMasks(1, {0: square(1)}),
                  2: FrameMasks(2, {0: square(2)}), 3: FrameMasks(3, {})}, range(0, 4))
    stitcher.add({i: FrameMasks(i, {9: square(i)}) for i in range(2, 6)}, range(2, 6))
    result = stitcher.result()
    assert result.object_ids() == [0]
    assert list(result.by_frame()[3].masks) == [0]  # 前のチャンクが見失ったフレームを次のチャンクで埋める
