"""outputs/ に保存した各記録を、MLflow に送る形（params / metrics / 成果物）にする #45。

どれも保存し終えたあとに呼ぶ。MLflow が無効・未接続なら何もしない（spatialsam.tracking.safe_log）。
大きなファイル（学習した PLY、フレーム画像など）は送らない。
"""

from __future__ import annotations

from typing import Any

from spatialsam.tracking import safe_log


def record_segmentation(experiment) -> None:
    """SAM の実験: 設定と性能、重畳動画。"""
    meta = experiment.load_metadata()
    perf = meta.get("performance", {})
    info = meta.get("segmenter_info", {})
    safe_log(
        "segmentation", experiment.id,
        params={"prompt": meta["prompt"], "model": meta["model"], "input.video": meta["input"]["video"],
                "sampling": meta.get("sampling", {}), "chunks": info.get("chunks", 1)},
        metrics={"elapsed_sec": perf.get("elapsed_sec"), "peak_vram_mib": perf.get("peak_vram_mib"),
                 "num_frames": len(meta["frame_indices"]), "num_objects": len(meta["object_ids"]),
                 "oom_retries": len(meta.get("oom_retries", []))},
        artifacts=[experiment.root / "metadata.json", experiment.root / "prediction" / "tracks.json",
                   experiment.overlay_path],
    )


def record_evaluation(experiment, summary: dict[str, float]) -> None:
    """正解マスクとの評価: 同じ実験の run に J / F / J&F / Dice を足す。"""
    safe_log("segmentation", experiment.id, metrics={f"eval.{k}": v for k, v in summary.items()},
             artifacts=[experiment.evaluation_dir / "per_frame.csv"])


def record_render(run) -> None:
    meta = run.load_metadata()
    perf = meta.get("performance", {})
    safe_log(
        "render", run.id,
        params={"scene": meta.get("scene"), "scene_source": meta.get("scene_source") or {},
                "trajectory": meta.get("trajectory", {}), "fps": meta.get("fps"),
                "resolution": "x".join(map(str, meta.get("resolution", [])))},
        metrics={"num_frames": meta.get("num_frames"), "elapsed_sec": perf.get("elapsed_sec"),
                 "peak_vram_mib": perf.get("peak_vram_mib")},
        artifacts=[run.root / "metadata.json", run.root / "poses.json"],
    )


def record_training(job, config: dict[str, Any], history: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    """3DGS 学習: 設定、学習曲線（step ごと）、評価用視点の PSNR / SSIM。"""
    request = job.load_request()
    series = {name: [(h["step"], h[name]) for h in history]
              for name in ("loss", "psnr", "num_gaussians", "peak_vram_mib") if history and name in history[0]}
    safe_log(
        "training", job.id,
        params={"sparse_dir": request.sparse_dir, "images_dir": request.images_dir,
                "holdout_every": request.holdout_every, "config": config},
        metrics={f"eval.{k}": v for k, v in metrics.items()} | {"elapsed_sec": history[-1]["elapsed_sec"]}
        if history else {f"eval.{k}": v for k, v in metrics.items()},
        series={f"train.{k}": v for k, v in series.items()},
        artifacts=[job.root / "metrics.json", job.log_path],
    )


def record_comparison(comparison) -> None:
    """実画像 vs 3D 描画: ソースごとの画質と、プロンプト・ソースごとの SAM の一致度。"""
    summary = comparison.load_summary()
    metrics = {f"quality.{source}.{k}": v for source, values in summary["quality"].items() for k, v in values.items()}
    for row in summary["segmentation"]:
        prefix = f"sam.{row['prompt']}.{row['source']}"
        metrics |= {f"{prefix}.detection_rate": row["detection_rate"], f"{prefix}.iou_vs_real": row["mean_iou_vs_real"]}
    safe_log(
        "comparison", comparison.id,
        params={"sources": summary["sources"], "prompts": ",".join(summary["prompts"]),
                "num_views": summary["num_views"], "holdout_every": summary.get("holdout_every")},
        metrics=metrics,
        artifacts=[comparison.root / name for name in ("summary.json", "quality.csv", "masks.csv")],
    )


def record_segmentation3d(run) -> None:
    """3D セグメンテーション: 設定、ラベル数と信頼度、評価（あれば）。"""
    summary = run.load_summary()
    metrics = {"num_labeled": summary["num_labeled"], "num_gaussians": summary["num_gaussians"],
               "num_views": summary["num_views"], "num_never_visible": summary["num_never_visible"]}
    for obj in summary["objects"]:
        metrics |= {f"object.{obj['object_id']}.num_gaussians": obj["num_gaussians"],
                    f"object.{obj['object_id']}.mean_confidence": obj["mean_confidence"]}
    reprojection = summary.get("reprojection") or {}
    metrics |= {f"reprojection.{k}": v for k, v in reprojection.items()
                if k in ("mean_iou", "median_iou", "mean_iou_where_reference_detected", "num_compared")}
    safe_log(
        "segmentation3d", run.id,
        params={"prompt": summary["prompt"], "experiment": summary["experiment"], "render": summary["render"],
                "settings": summary["settings"]},
        metrics=metrics,
        artifacts=[run.root / "summary.json"],
    )
