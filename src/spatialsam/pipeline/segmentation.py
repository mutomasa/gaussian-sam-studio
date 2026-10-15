"""動画 → フレーム抽出 → 推論 → 保存 → 重畳動画 の一連の処理。"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from spatialsam.experiments.store import Experiment, ExperimentStore
from spatialsam.models.gpu import Measurement, empty_cuda_cache, is_out_of_memory, measure
from spatialsam.models.segmenter import VideoSegmenter
from spatialsam.models.types import SegmentationRequest, SegmentationResult
from spatialsam.pipeline.retry import OomRetryPolicy
from spatialsam.video.io import FrameSampling, extract_frames, list_frame_files, read_video_info, write_video
from spatialsam.video.overlay import overlay_masks

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class SegmentationJob:
    video_path: Path
    prompt: str
    sampling: FrameSampling = field(default_factory=FrameSampling)
    model_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceRun:
    """推論に実際に使った入力と結果。OOM で縮小した試行は `oom_attempts` に残す。"""

    sampling: FrameSampling
    indices: list[int]
    result: SegmentationResult
    measurement: Measurement
    oom_attempts: list[dict[str, Any]]


@dataclass(frozen=True)
class SegmentationOutcome:
    experiment: Experiment
    result: SegmentationResult
    measurement: Measurement


def _noop(_: str) -> None:
    return None


def iter_overlay_frames(frames_dir: Path, result: SegmentationResult, indices: list[int]) -> Iterator[np.ndarray]:
    by_frame = result.by_frame()
    for path, index in zip(list_frame_files(frames_dir), indices, strict=True):
        yield overlay_masks(cv2.imread(str(path)), by_frame.get(index))


def build_metadata(job: SegmentationJob, segmenter: VideoSegmenter, run: InferenceRun, fps: float) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt": job.prompt,
        "model": {"segmenter": segmenter.name, **job.model_info},
        "input": {"video": job.video_path.name, "fps": fps},
        "sampling": asdict(run.sampling),
        "requested_sampling": asdict(job.sampling),
        "oom_retries": run.oom_attempts,
        "frame_indices": run.indices,
        "object_ids": run.result.object_ids(),
        "performance": {
            "elapsed_sec": run.measurement.elapsed_sec,
            "peak_vram_mib": run.measurement.peak_vram_mib,
        },
    }


def copy_input(video_path: Path, experiment: Experiment) -> Path:
    experiment.input_dir.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy2(video_path, experiment.input_dir / video_path.name))


def extract_fresh(video: Path, sampling: FrameSampling, frames_dir: Path) -> list[int]:
    """以前の試行のフレームを消してから抽出し直す。"""
    shutil.rmtree(frames_dir, ignore_errors=True)
    return extract_frames(video, sampling, frames_dir)


def segment_with_retry(video: Path, job: SegmentationJob, segmenter: VideoSegmenter, frames_dir: Path,
                       policy: OomRetryPolicy, progress: ProgressCallback) -> InferenceRun:
    sampling, attempts = job.sampling, []
    while True:
        progress("フレームを抽出しています")
        indices = extract_fresh(video, sampling, frames_dir)
        progress(f"{segmenter.name} で推論しています（{len(indices)} フレーム）")
        try:
            with measure() as measurement:
                raw = segmenter.segment(SegmentationRequest(frames_dir, job.prompt))
            return InferenceRun(sampling, indices, raw.remap_indices(indices), measurement, attempts)
        except Exception as error:
            if not is_out_of_memory(error) or len(attempts) >= policy.max_retries or not policy.can_degrade(len(indices)):
                raise
            empty_cuda_cache()
            attempts.append({"sampling": asdict(sampling), "num_frames": len(indices), "error": "out_of_memory"})
            sampling = policy.degrade(sampling, len(indices))
            progress(f"GPU メモリ不足のため {sampling.max_frames} フレームに減らして再実行します")


def run_segmentation(job: SegmentationJob, segmenter: VideoSegmenter, store: ExperimentStore,
                     progress: ProgressCallback = _noop,
                     retry_policy: OomRetryPolicy = OomRetryPolicy()) -> SegmentationOutcome:
    experiment = store.create()
    video = copy_input(job.video_path, experiment)
    info = read_video_info(video)
    run = segment_with_retry(video, job, segmenter, experiment.frames_dir, retry_policy, progress)
    progress("結果を保存しています")
    experiment.save_prediction(run.result)
    fps = run.sampling.output_fps(info.fps)
    write_video(iter_overlay_frames(experiment.frames_dir, run.result, run.indices), experiment.overlay_path, fps)
    experiment.save_metadata(build_metadata(job, segmenter, run, fps))
    return SegmentationOutcome(experiment, run.result, run.measurement)
