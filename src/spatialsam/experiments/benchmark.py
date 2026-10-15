"""GT 付きサンプルを一括で推論・評価し、1サンプル1行の表にまとめる。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from spatialsam.evaluation.ground_truth import load_gt_dir
from spatialsam.experiments.store import ExperimentStore
from spatialsam.models.segmenter import VideoSegmenter
from spatialsam.pipeline.evaluation import EvaluationReport, evaluate_experiment
from spatialsam.pipeline.segmentation import SegmentationJob, SegmentationOutcome, run_segmentation
from spatialsam.samples.davis import SampleSpec, default_gt_dir
from spatialsam.video.io import FrameSampling

ProgressCallback = Callable[[str], None]


def _round_or_none(value: float | None) -> int | None:
    return None if value is None else round(value)


def benchmark_row(spec: SampleSpec, outcome: SegmentationOutcome, report: EvaluationReport) -> dict[str, Any]:
    meta = outcome.experiment.load_metadata()
    return {
        "sequence": spec.sequence,
        "prompt": spec.prompt,
        "experiment": outcome.experiment.id,
        "frames": len(meta["frame_indices"]),
        "objects": len(meta["object_ids"]),
        "oom_retries": len(meta["oom_retries"]),
        "elapsed_sec": round(outcome.measurement.elapsed_sec, 2),
        "peak_vram_mib": _round_or_none(outcome.measurement.peak_vram_mib),
        **{key: round(value, 4) for key, value in report.summary.items() if key != "num_frames"},
    }


def run_sample(spec: SampleSpec, samples_dir: Path, sampling: FrameSampling, segmenter: VideoSegmenter,
               store: ExperimentStore, model_info: dict[str, Any], progress: ProgressCallback) -> dict[str, Any]:
    video = samples_dir / f"{spec.sequence}.mp4"
    gt_dir = default_gt_dir(samples_dir, video.name)
    if not video.exists() or gt_dir is None:
        raise FileNotFoundError(f"サンプルがありません（scripts/prepare_samples.py を実行）: {video}")
    job = SegmentationJob(video, spec.prompt, sampling, model_info)
    outcome = run_segmentation(job, segmenter, store, progress)
    return benchmark_row(spec, outcome, evaluate_experiment(outcome.experiment, load_gt_dir(gt_dir)))
