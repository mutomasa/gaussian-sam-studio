import io
import zipfile

import cv2
import numpy as np
import pytest

from spatialsam.evaluation.ground_truth import align_mask, load_gt_dir, load_gt_zip, parse_frame_index


def png_bytes(mask: np.ndarray) -> bytes:
    return cv2.imencode(".png", mask.astype(np.uint8) * 255)[1].tobytes()


@pytest.mark.parametrize(("name", "expected"), [("00012.png", 12), ("frame_000003.png", 3), ("a/b/7.png", 7)])
def test_parse_frame_index(name, expected):
    assert parse_frame_index(name) == expected


def test_parse_frame_index_without_digits():
    with pytest.raises(ValueError):
        parse_frame_index("mask.png")


def test_load_gt_dir(tmp_path):
    mask = np.zeros((4, 4), bool)
    mask[1, 1] = True
    (tmp_path / "00002.png").write_bytes(png_bytes(mask))
    loaded = load_gt_dir(tmp_path)
    assert list(loaded) == [2]
    assert loaded[2].dtype == bool and loaded[2][1, 1] and loaded[2].sum() == 1


def test_load_gt_zip_ignores_non_png():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("gt/00005.png", png_bytes(np.ones((3, 3), bool)))
        archive.writestr("gt/readme.txt", "x")
    assert list(load_gt_zip(buffer.getvalue())) == [5]


def test_align_mask_resizes_with_nearest():
    mask = np.zeros((4, 4), bool)
    mask[:2, :2] = True
    aligned = align_mask(mask, (8, 8))
    assert aligned.shape == (8, 8)
    assert aligned[:4, :4].all() and not aligned[4:, 4:].any()
