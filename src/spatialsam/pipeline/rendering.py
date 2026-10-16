"""3D シーン → カメラ軌道に沿った RGB / Depth / Pose 付きフレーム → MP4。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from spatialsam.experiments.renders import RenderRun, RenderStore
from spatialsam.experiments.tracking_records import record_render
from spatialsam.models.gpu import measure
from spatialsam.spatial.camera import Camera
from spatialsam.spatial.renderer import RenderOutput
from spatialsam.video.depth import colorize_depth, depth_range
from spatialsam.video.io import write_video

ProgressCallback = Callable[[str], None]
MIN_ALPHA = 0.5  # これ未満の画素の深度は無効（背景）とみなす


class Renderer(Protocol):
    def render(self, camera: Camera) -> RenderOutput: ...


@dataclass(frozen=True)
class RenderJob:
    scene_path: Path
    frames: list[tuple[str, Camera]]  # (元画像名 or "", カメラ)
    fps: float = 24.0
    trajectory: dict[str, Any] = field(default_factory=dict)
    scene_source: dict[str, str | None] = field(default_factory=dict)  # 3D Segmentation でシーンを復元するため


def _noop(_: str) -> None:
    return None


def render_frames(renderer: Renderer, run: RenderRun, frames: list[tuple[str, Camera]],
                  progress: ProgressCallback) -> None:
    run.rgb_dir.mkdir(parents=True, exist_ok=True)
    for index, (_, camera) in enumerate(frames):
        output = renderer.render(camera)
        cv2.imwrite(str(run.rgb_path(index)), cv2.cvtColor(output.rgb, cv2.COLOR_RGB2BGR))
        run.save_depth(index, output.depth, output.alpha)
        if index % 20 == 0:
            progress(f"描画中 {index + 1}/{len(frames)}")


def iter_rgb(run: RenderRun, count: int) -> Iterator[np.ndarray]:
    for index in range(count):
        yield cv2.imread(str(run.rgb_path(index)))


def iter_depth_preview(run: RenderRun, count: int, near: float, far: float) -> Iterator[np.ndarray]:
    for index in range(count):
        depth, alpha = run.load_depth(index)
        yield colorize_depth(depth, alpha >= MIN_ALPHA, near, far)


def shared_depth_range(run: RenderRun, count: int, samples: int = 16) -> tuple[float, float]:
    indices = np.linspace(0, count - 1, min(samples, count)).astype(int)
    loaded = [run.load_depth(i) for i in indices]
    return depth_range([d for d, _ in loaded], [a >= MIN_ALPHA for _, a in loaded])


def build_metadata(job: RenderJob, elapsed_sec: float, peak_vram_mib: float | None,
                   depth_limits: tuple[float, float]) -> dict[str, Any]:
    first = job.frames[0][1]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scene": str(job.scene_path),
        "scene_source": job.scene_source,
        "num_frames": len(job.frames),
        "fps": job.fps,
        "resolution": [first.width, first.height],
        "trajectory": job.trajectory,
        "depth_preview_range": list(depth_limits),
        "performance": {"elapsed_sec": elapsed_sec, "peak_vram_mib": peak_vram_mib},
    }


def run_render(job: RenderJob, renderer: Renderer, store: RenderStore,
               progress: ProgressCallback = _noop) -> RenderRun:
    if not job.frames:
        raise ValueError("描画するカメラがありません")
    run = store.create()
    run.save_poses(job.frames)
    with measure() as measurement:
        render_frames(renderer, run, job.frames, progress)
    progress("動画を書き出しています")
    count = len(job.frames)
    write_video(iter_rgb(run, count), run.video_path, job.fps)
    limits = shared_depth_range(run, count)
    write_video(iter_depth_preview(run, count, *limits), run.depth_video_path, job.fps)
    run.save_metadata(build_metadata(job, measurement.elapsed_sec, measurement.peak_vram_mib, limits))
    record_render(run)
    return run
