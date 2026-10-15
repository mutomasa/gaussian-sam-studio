"""SAM の実験（3D 描画動画が入力）→ 描画時の姿勢・深度で Gaussian にラベルを付ける #24。

描画動画のフレーム i は、描画結果の poses.json / depth の i 番目と同じ視点。
SAM はフレームを間引いたり縮小したりして推論するので、frame_indices で対応を取り、
カメラと深度をマスクの解像度に合わせる。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from spatialsam.experiments.renders import RenderRun, RenderStore
from spatialsam.experiments.segmentation3d import Segmentation3D, Segmentation3DStore
from spatialsam.experiments.store import Experiment
from spatialsam.pipeline.evaluation import inference_frame_shape
from spatialsam.spatial.gaussians import GaussianScene, save_ply
from spatialsam.spatial.mask_lifting import LiftResult, LiftSettings, MaskedView, lift_masks
from spatialsam.spatial.scene_highlight import subset_scene
from spatialsam.spatial.scene_source import SceneSource


def render_for_experiment(experiment: Experiment, renders: RenderStore) -> RenderRun:
    """SAM の入力動画（<render_id>.mp4）から描画結果を探す。"""
    video = experiment.load_metadata()["input"]["video"]
    render_id = video.removesuffix(".mp4")
    try:
        return renders.open(render_id)
    except FileNotFoundError as error:
        raise ValueError(f"この実験の入力は 3D 描画の動画ではありません: {video}") from error


def scene_source_of(run: RenderRun) -> SceneSource:
    meta = run.load_metadata()
    return SceneSource.from_dict(meta["scene_source"]) if meta.get("scene_source") else SceneSource.from_label(meta["scene"])


def resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == shape:
        return image
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def masked_views(experiment: Experiment, run: RenderRun) -> Iterator[MaskedView]:
    shape = inference_frame_shape(experiment)
    poses = run.load_poses()
    by_frame = experiment.load_prediction().by_frame()
    for index in experiment.load_metadata()["frame_indices"]:
        frame = by_frame.get(index)
        depth, alpha = run.load_depth(index)
        masks = {} if frame is None else frame.masks
        yield MaskedView(poses[index][1].resized(shape[1], shape[0]), masks,
                         resize_nearest(depth, shape), resize_nearest(alpha, shape))


def object_summary(result: LiftResult) -> list[dict[str, Any]]:
    rows = []
    for object_id in result.object_ids:
        labels = result.labels(object_id)
        confidence = result.votes.confidence(object_id)
        rows.append({"object_id": object_id, "num_gaussians": int(labels.sum()),
                     "mean_confidence": float(confidence[labels].mean()) if labels.any() else None})
    return rows


def save_result(run: Segmentation3D, scene: GaussianScene, result: LiftResult) -> None:
    run.save_labels(result.votes.visible, {k: result.votes.confidence(k) for k in result.object_ids},
                    {k: result.labels(k) for k in result.object_ids})
    save_ply(subset_scene(scene, result.any_labels()), run.object_ply_path)


def run_lifting(experiment: Experiment, renders: RenderStore, store: Segmentation3DStore,
                settings: LiftSettings = LiftSettings()) -> Segmentation3D:
    render_run = render_for_experiment(experiment, renders)
    source = scene_source_of(render_run)
    scene = source.load_scene()
    result = lift_masks(scene.means, masked_views(experiment, render_run), settings)
    run = store.create()
    save_result(run, scene, result)
    run.save_summary({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment.id,
        "prompt": experiment.load_metadata()["prompt"],
        "render": render_run.id,
        "scene_source": source.to_dict(),
        "settings": settings.__dict__,
        "num_views": result.num_views,
        "num_gaussians": len(scene),
        "num_labeled": int(result.any_labels().sum()),
        "num_never_visible": int((result.votes.visible == 0).sum()),
        "objects": object_summary(result),
    })
    return run


def reference_masks(experiment: Experiment) -> list[np.ndarray]:
    """基準にする 2D の SAM 結果（フレーム順の前景マスク）。"""
    from spatialsam.pipeline.comparison import foregrounds

    return foregrounds(experiment)


def evaluate_reprojection(run: Segmentation3D, scene: GaussianScene, cameras: list, reference: Experiment,
                          renderer_factory) -> dict[str, Any]:
    """持ち上げた 3D ラベルを reference の視点に描き、reference の SAM マスクと比べて summary に保存する。"""
    from spatialsam.evaluation.geometry import label_masks, reprojection_ious, summarize_reprojection

    masks = reference_masks(reference)
    if len(masks) != len(cameras):
        raise ValueError(f"視点数が合いません: カメラ {len(cameras)} / マスク {len(masks)}")
    shape = masks[0].shape
    cameras = [camera.resized(shape[1], shape[0]) for camera in cameras]
    predicted = label_masks(renderer_factory(scene), run.any_labels(), cameras)
    ious = reprojection_ious(predicted, masks)
    evaluation = {"reference_experiment": reference.id, **summarize_reprojection(ious, predicted, masks),
                  "per_view_iou": ious}
    run.update_summary(reprojection=evaluation)
    return evaluation
