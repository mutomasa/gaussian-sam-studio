"""実画像 vs 3D シーンの描画画像（同じ視点）を、画質と SAM 3.1 の結果で比べる #16。

1. 評価用視点（学習に使っていない、8 枚に 1 枚）の実画像を集める
2. 各シーン（学習済み 3DGS、点群など）を同じカメラで描く
3. 画質: 描画 vs 実画像の PSNR / SSIM / LPIPS
4. SAM: 実画像・各描画を同じ順の動画にして同じプロンプトで推論し、描画側のマスクを実画像側のマスクと比べる
   （bonsai には正解マスクが無いので、実画像での SAM の結果を基準にする）
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import pandas as pd

from spatialsam.evaluation.image_quality import QualityScore, summarize_quality
from spatialsam.evaluation.segmentation import iou
from spatialsam.experiments.comparisons import REAL, Comparison, ComparisonStore
from spatialsam.experiments.store import Experiment, ExperimentStore
from spatialsam.models.gpu import empty_cuda_cache
from spatialsam.models.segmenter import VideoSegmenter
from spatialsam.pipeline.evaluation import inference_frame_shape, predicted_foreground
from spatialsam.pipeline.segmentation import SegmentationJob, run_segmentation
from spatialsam.spatial.camera import Camera
from spatialsam.spatial.renderer import RenderOutput
from spatialsam.spatial.scene_source import SceneSource
from spatialsam.training.dataset import TrainView
from spatialsam.video.io import write_video

ProgressCallback = Callable[[str], None]


class Renderer(Protocol):
    def render(self, camera: Camera) -> RenderOutput: ...


class QualityScorer(Protocol):
    def score(self, rendered: np.ndarray, target: np.ndarray) -> QualityScore: ...


@dataclass(frozen=True)
class ComparisonJob:
    views: list[TrainView]                   # 評価用視点（実画像とカメラ）
    sources: dict[str, SceneSource]          # 名前 → 描画するシーン（例: {"3dgs": ..., "points": ...}）
    prompts: list[str]
    fps: float = 8.0                         # 視点が飛び飛びなので遅めにする
    info: dict[str, Any] = field(default_factory=dict)


def _noop(_: str) -> None:
    return None


def crop_even(view: TrainView) -> TrainView:
    """H.264 で書けるよう右端・下端を 1px 削る（主点は変わらないので K はそのまま）。"""
    height, width = view.image.shape[:2]
    even_h, even_w = height - height % 2, width - width % 2
    if (even_h, even_w) == (height, width):
        return view
    return TrainView(view.name, Camera(view.camera.K, view.camera.world_to_cam, even_w, even_h),
                     view.image[:even_h, :even_w])


def render_views(renderer: Renderer, views: list[TrainView]) -> list[np.ndarray]:
    return [renderer.render(view.camera).rgb for view in views]


def score_quality(source: str, rendered: list[np.ndarray], views: list[TrainView],
                  scorer: QualityScorer) -> list[dict[str, Any]]:
    return [{"source": source, "index": i, "view": view.name, **scorer.score(image, view.image).as_dict()}
            for i, (image, view) in enumerate(zip(rendered, views, strict=True))]


def save_frames(comparison: Comparison, source: str, images: Iterable[np.ndarray], fps: float) -> Path:
    """PNG（可逆）と、SAM の入力にする動画を書く。"""
    comparison.frames_dir(source).mkdir(parents=True, exist_ok=True)
    bgr = []
    for index, image in enumerate(images):
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(comparison.frame_path(source, index)), frame)
        bgr.append(frame)
    return write_video(bgr, comparison.video_path(source), fps)


def foregrounds(experiment: Experiment) -> list[np.ndarray]:
    """フレーム順の前景マスク（全物体の和集合）。検出が無いフレームは空。"""
    shape = inference_frame_shape(experiment)
    by_frame = experiment.load_prediction().by_frame()
    return [predicted_foreground(by_frame.get(i), shape) for i in experiment.load_metadata()["frame_indices"]]


def mask_rows(prompt: str, source: str, masks: list[np.ndarray], reference: list[np.ndarray]) -> list[dict[str, Any]]:
    """実画像のマスクとの IoU。どちらにもマスクが無い視点は IoU を空にする（一致はしているが比べる面積が無い）。"""
    rows = []
    for index, (mask, ref) in enumerate(zip(masks, reference, strict=True)):
        both_empty = not mask.any() and not ref.any()
        rows.append({"prompt": prompt, "source": source, "index": index, "detected": bool(mask.any()),
                     "area": float(mask.mean()), "iou_vs_real": None if both_empty else iou(mask, ref)})
    return rows


def summarize_masks(table: pd.DataFrame) -> list[dict[str, Any]]:
    if table.empty:
        return []
    grouped = table.groupby(["prompt", "source"], sort=False)
    return [{"prompt": prompt, "source": source, "detection_rate": float(group["detected"].mean()),
             "mean_iou_vs_real": None if group["iou_vs_real"].isna().all() else float(group["iou_vs_real"].mean())}
            for (prompt, source), group in grouped]


def segment_sources(comparison: Comparison, sources: list[str], prompts: list[str], segmenter: VideoSegmenter,
                    experiments: ExperimentStore, progress: ProgressCallback) -> tuple[pd.DataFrame, dict]:
    rows, experiment_ids = [], {}
    for prompt in prompts:
        masks = {}
        for source in sources:
            progress(f"SAM: 「{prompt}」/ {source}")
            outcome = run_segmentation(SegmentationJob(comparison.video_path(source), prompt), segmenter, experiments)
            experiment_ids[f"{prompt}/{source}"] = outcome.experiment.id
            masks[source] = foregrounds(outcome.experiment)
        for source in sources:
            rows += mask_rows(prompt, source, masks[source], masks[REAL])
    return pd.DataFrame(rows), experiment_ids


def render_and_score(job: ComparisonJob, comparison: Comparison, renderer_factory: Callable[[SceneSource], Renderer],
                     scorer: QualityScorer, progress: ProgressCallback) -> pd.DataFrame:
    rows = []
    for name, source in job.sources.items():
        progress(f"描画: {name}")
        renderer = renderer_factory(source)
        rendered = render_views(renderer, job.views)
        del renderer
        empty_cuda_cache()
        save_frames(comparison, name, rendered, job.fps)
        rows += score_quality(name, rendered, job.views, scorer)
    return pd.DataFrame(rows)


def run_comparison(job: ComparisonJob, store: ComparisonStore, experiments: ExperimentStore,
                   segmenter: VideoSegmenter, scorer: QualityScorer,
                   renderer_factory: Callable[[SceneSource], Renderer], progress: ProgressCallback = _noop) -> Comparison:
    if REAL in job.sources:
        raise ValueError(f"ソース名 {REAL!r} は実画像用に予約されています")
    comparison = store.create()
    views = [crop_even(v) for v in job.views]
    job = ComparisonJob(views, job.sources, job.prompts, job.fps, job.info)
    save_frames(comparison, REAL, (v.image for v in views), job.fps)
    quality = render_and_score(job, comparison, renderer_factory, scorer, progress)
    comparison.save_table("quality", quality)
    masks, experiment_ids = segment_sources(comparison, [REAL, *job.sources], job.prompts, segmenter,
                                            experiments, progress)
    comparison.save_table("masks", masks)
    comparison.save_summary(build_summary(job, quality, masks, experiment_ids))
    return comparison


def build_summary(job: ComparisonJob, quality: pd.DataFrame, masks: pd.DataFrame,
                  experiment_ids: dict[str, str]) -> dict[str, Any]:
    by_source = {name: summarize_quality([QualityScore(r.psnr, r.ssim, r.lpips)
                                          for r in quality[quality["source"] == name].itertuples()])
                 for name in job.sources}
    first = job.views[0].camera
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_views": len(job.views),
        "resolution": [first.width, first.height],
        "views": [v.name for v in job.views],
        "sources": {name: source.label for name, source in job.sources.items()},
        "prompts": job.prompts,
        "quality": by_source,
        "segmentation": summarize_masks(masks),
        "experiments": experiment_ids,
        **job.info,
    }


def comparison_cameras(comparison: Comparison) -> list[Camera]:
    """比較に使った評価用視点のカメラ（実画像と同じく偶数サイズに切ったもの）を作り直す。"""
    from spatialsam.training.dataset import ColmapDataset

    info = comparison.load_summary()
    views = ColmapDataset.load(Path(info["sparse_dir"]), Path(info["images_dir"]), info["holdout_every"]).test
    return [crop_even(v).camera for v in views]
