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
from spatialsam.models.gpu import Measurement, measure
from spatialsam.models.segmenter import VideoSegmenter
from spatialsam.models.types import SegmentationRequest, SegmentationResult
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


def build_metadata(job: SegmentationJob, segmenter: VideoSegmenter, indices: list[int], fps: float,
                   measurement: Measurement, result: SegmentationResult) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt": job.prompt,
        "model": {"segmenter": segmenter.name, **job.model_info},
        "input": {"video": job.video_path.name, "fps": fps},
        "sampling": asdict(job.sampling),
        "frame_indices": indices,
        "object_ids": result.object_ids(),
        "performance": {"elapsed_sec": measurement.elapsed_sec, "peak_vram_mib": measurement.peak_vram_mib},
    }


def copy_input(video_path: Path, experiment: Experiment) -> Path:
    experiment.input_dir.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy2(video_path, experiment.input_dir / video_path.name))


def run_segmentation(job: SegmentationJob, segmenter: VideoSegmenter, store: ExperimentStore,
                     progress: ProgressCallback = _noop) -> SegmentationOutcome:
    experiment = store.create()
    video = copy_input(job.video_path, experiment)
    info = read_video_info(video)
    progress("フレームを抽出しています")
    indices = extract_frames(video, job.sampling, experiment.frames_dir)
    progress(f"{segmenter.name} で推論しています（{len(indices)} フレーム）")
    with measure() as measurement:
        raw = segmenter.segment(SegmentationRequest(experiment.frames_dir, job.prompt))
    result = raw.remap_indices(indices)
    progress("結果を保存しています")
    experiment.save_prediction(result)
    fps = job.sampling.output_fps(info.fps)
    write_video(iter_overlay_frames(experiment.frames_dir, result, indices), experiment.overlay_path, fps)
    experiment.save_metadata(build_metadata(job, segmenter, indices, fps, measurement, result))
    return SegmentationOutcome(experiment, result, measurement)
