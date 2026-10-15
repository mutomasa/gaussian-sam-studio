import numpy as np
import pytest

from spatialsam.evaluation.geometry import iou_3d, label_masks, reprojection_ious, summarize_reprojection
from spatialsam.spatial.camera import Camera, intrinsics


def test_iou_3d():
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 1, 0], bool)
    assert iou_3d(a, b) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        iou_3d(a, b[:3])


class StubFeatureRenderer:
    def render_features(self, camera, features):
        image = np.zeros((camera.height, camera.width, 1), np.float32)
        image[:2, :2, 0] = features.max()  # 対象が左上に写る
        return image


def test_label_masks_thresholds_composited_labels():
    camera = Camera(intrinsics(10, 10, 2, 2), np.eye(4), 4, 4)
    masks = label_masks(StubFeatureRenderer(), np.array([True, False]), [camera, camera])
    assert len(masks) == 2 and masks[0].sum() == 4 and masks[0][0, 0]
    assert not label_masks(StubFeatureRenderer(), np.array([False]), [camera])[0].any()


def test_reprojection_summary_separates_reference_misses():
    full, empty = np.ones((2, 2), bool), np.zeros((2, 2), bool)
    half = np.array([[True, True], [False, False]])
    predicted, reference = [full, full, empty], [half, empty, empty]
    ious = reprojection_ious(predicted, reference)
    assert ious == [0.5, 0.0, None]
    summary = summarize_reprojection(ious, predicted, reference)
    assert summary["mean_iou"] == pytest.approx(0.25)
    assert summary["mean_iou_where_reference_detected"] == pytest.approx(0.5)
    assert summary["reference_empty_views"] == [1] and summary["num_compared"] == 2
