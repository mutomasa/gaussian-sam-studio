import zipfile

import cv2
import numpy as np
import pytest

from spatialsam.evaluation.ground_truth import load_gt_dir
from spatialsam.samples.davis import (
    ANNOTATION_DIR,
    JPEG_DIR,
    SampleSpec,
    default_gt_dir,
    load_manifest,
    prepare_sample,
    sequence_members,
    sha256sum,
    verify_sha256,
)
from spatialsam.video.io import read_video_info


def encode(image: np.ndarray, ext: str) -> bytes:
    return cv2.imencode(ext, image)[1].tobytes()


@pytest.fixture
def davis_zip(tmp_path):
    path = tmp_path / "davis.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(3):
            frame = np.full((48, 64, 3), 40 * i, np.uint8)
            label = np.zeros((48, 64), np.uint8)
            label[10:20, 10:20] = 1
            label[30:40, 40:50] = 2
            archive.writestr(f"{JPEG_DIR}/swan/{i:05d}.jpg", encode(frame, ".jpg"))
            archive.writestr(f"{ANNOTATION_DIR}/swan/{i:05d}.png", encode(label, ".png"))
        archive.writestr(f"{JPEG_DIR}/other/00000.jpg", encode(np.zeros((8, 8, 3), np.uint8), ".jpg"))
    return path


def test_sequence_members_filters_and_sorts():
    names = [f"{JPEG_DIR}/a/00010.jpg", f"{JPEG_DIR}/a/00002.jpg", f"{JPEG_DIR}/ab/00000.jpg",
             f"{JPEG_DIR}/a/sub/00000.jpg", f"{ANNOTATION_DIR}/a/00000.png"]
    assert sequence_members(names, JPEG_DIR, "a", ".jpg") == [f"{JPEG_DIR}/a/00002.jpg", f"{JPEG_DIR}/a/00010.jpg"]


def test_prepare_sample_writes_video_and_gt(davis_zip, tmp_path):
    out = tmp_path / "samples"
    sample = prepare_sample(davis_zip, SampleSpec("swan", "swan"), out, fps=24)
    info = read_video_info(sample.video)
    assert (sample.num_frames, info.frame_count, info.width, info.height) == (3, 3, 64, 48)
    gt = load_gt_dir(sample.gt_dir)
    assert sorted(gt) == [0, 1, 2]
    assert gt[0].sum() == 200  # 物体ID 1 と 2 を前景として読む
    assert default_gt_dir(out, "swan.mp4") == sample.gt_dir
    assert default_gt_dir(out, "missing.mp4") is None


def test_prepare_sample_unknown_sequence(davis_zip, tmp_path):
    with pytest.raises(ValueError):
        prepare_sample(davis_zip, SampleSpec("nope", "x"), tmp_path, fps=24)


def test_load_manifest(tmp_path):
    path = tmp_path / "samples.yaml"
    path.write_text(
        "url: https://example.com/data/DAVIS.zip\nfps: 24\nlicense: CC BY-NC 4.0\n"
        "samples:\n  - sequence: swan\n    prompt: swan\n",
        encoding="utf-8",
    )
    manifest = load_manifest(path)
    assert manifest.archive_name == "DAVIS.zip"
    assert manifest.samples == [SampleSpec("swan", "swan")]


def test_verify_sha256(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    verify_sha256(path, sha256sum(path))
    verify_sha256(path, "")  # 未指定なら検証しない
    with pytest.raises(ValueError):
        verify_sha256(path, "0" * 64)
