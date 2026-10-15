import numpy as np
import pytest

from spatialsam.evaluation.segmentation import (
    FrameScore,
    boundary_f_measure,
    boundary_tolerance,
    dice,
    evaluate_frame,
    iou,
    summarize,
)


def box(shape=(20, 20), rows=slice(5, 15), cols=slice(5, 15)) -> np.ndarray:
    mask = np.zeros(shape, bool)
    mask[rows, cols] = True
    return mask


def test_iou_and_dice_identical_masks_are_one():
    assert iou(box(), box()) == 1.0
    assert dice(box(), box()) == 1.0


def test_iou_and_dice_partial_overlap():
    a = box(cols=slice(0, 10), rows=slice(0, 10))
    b = box(cols=slice(5, 15), rows=slice(0, 10))
    assert iou(a, b) == pytest.approx(50 / 150)
    assert dice(a, b) == pytest.approx(100 / 200)


def test_both_empty_counts_as_perfect():
    empty = np.zeros((8, 8), bool)
    assert iou(empty, empty) == 1.0
    assert dice(empty, empty) == 1.0
    assert boundary_f_measure(empty, empty) == 1.0


def test_one_empty_is_zero():
    empty = np.zeros((20, 20), bool)
    assert iou(box(), empty) == 0.0
    assert boundary_f_measure(box(), empty) == 0.0


def test_boundary_f_tolerates_small_shift():
    shifted = box(cols=slice(6, 16))
    assert boundary_f_measure(box(), shifted, tolerance=1) == 1.0
    assert boundary_f_measure(box(), box(cols=slice(10, 20)), tolerance=1) < 1.0


def test_boundary_tolerance_uses_diagonal():
    assert boundary_tolerance((480, 854)) == 8
    assert boundary_tolerance((10, 10)) == 1


def test_evaluate_frame_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        evaluate_frame(0, np.zeros((4, 4), bool), np.zeros((5, 5), bool))


def test_summarize_computes_j_and_f():
    scores = [FrameScore(0, 1.0, 1.0, 0.5), FrameScore(1, 0.5, 0.6, 0.5)]
    summary = summarize(scores)
    assert summary["J"] == pytest.approx(0.75)
    assert summary["F"] == pytest.approx(0.5)
    assert summary["J&F"] == pytest.approx(0.625)
    assert summary["num_frames"] == 2


def test_summarize_requires_scores():
    with pytest.raises(ValueError):
        summarize([])
