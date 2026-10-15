"""保存済み実験の予測と GT マスクを比較して評価を保存する。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from spatialsam.evaluation.ground_truth import align_mask
from spatialsam.evaluation.segmentation import FrameScore, evaluate_frame, summarize
from spatialsam.experiments.store import Experiment
from spatialsam.models.types import FrameMasks, SegmentationResult
from spatialsam.video.io import list_frame_files


@dataclass(frozen=True)
class EvaluationReport:
    summary: dict[str, float]
    frame_scores: list[FrameScore]
    missing_gt_frames: list[int]


def predicted_foreground(frame: FrameMasks | None, shape: tuple[int, int]) -> np.ndarray:
    """物体が検出されなかったフレームは空マスクとして扱う。"""
    return np.zeros(shape, bool) if frame is None else frame.union(shape)


def score_frames(prediction: SegmentationResult, frame_indices: list[int],
                 gt_masks: dict[int, np.ndarray], shape: tuple[int, int]) -> list[FrameScore]:
    by_frame = prediction.by_frame()
    return [
        evaluate_frame(i, predicted_foreground(by_frame.get(i), shape), align_mask(gt_masks[i], shape))
        for i in frame_indices
        if i in gt_masks
    ]


def inference_frame_shape(experiment: Experiment) -> tuple[int, int]:
    """推論に使ったフレーム（リサイズ後）の (H, W)。"""
    first = list_frame_files(experiment.frames_dir)[0]
    return cv2.imread(str(first), cv2.IMREAD_GRAYSCALE).shape


def evaluate_experiment(experiment: Experiment, gt_masks: dict[int, np.ndarray]) -> EvaluationReport:
    """GT のあるフレームのみ評価し、結果を実験ディレクトリに保存する。"""
    shape = inference_frame_shape(experiment)
    frame_indices = experiment.load_metadata()["frame_indices"]
    scores = score_frames(experiment.load_prediction(), frame_indices, gt_masks, shape)
    report = EvaluationReport(
        summarize(scores),
        scores,
        [i for i in frame_indices if i not in gt_masks],
    )
    experiment.save_evaluation(report.summary, report.frame_scores)
    return report
