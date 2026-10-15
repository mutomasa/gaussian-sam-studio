"""SAM の 2D マスク（3D 描画動画への推論結果）を Gaussian に持ち上げ、必要なら評価用視点で評価する。

    uv run python scripts/lift_masks.py --experiment <experiment_id> \
        [--compare <comparison_id>] [--min-ratio 0.5] [--min-views 2]

- `--experiment`: Video Segmentation の実験 ID（入力が 3D Rendering の動画であること）
- `--compare`: 実画像 vs 3D 描画の比較 ID。同じプロンプトの実画像での SAM 結果と、評価用視点で再投影 IoU を取る
- 結果は outputs/segmentation3d/<id>/（labels.npz、object.ply、summary.json）
"""

from __future__ import annotations

import argparse
import json

from spatialsam.config import load_config
from spatialsam.experiments.comparisons import REAL, ComparisonStore
from spatialsam.experiments.renders import RenderStore
from spatialsam.experiments.segmentation3d import Segmentation3DStore
from spatialsam.experiments.store import ExperimentStore
from spatialsam.pipeline.comparison import comparison_cameras
from spatialsam.pipeline.lifting import evaluate_reprojection, render_for_experiment, run_lifting, scene_source_of
from spatialsam.spatial.mask_lifting import LiftSettings
from spatialsam.spatial.renderer import GaussianRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--compare", help="評価に使う比較結果の ID")
    parser.add_argument("--min-ratio", type=float, default=LiftSettings.min_ratio)
    parser.add_argument("--min-views", type=int, default=LiftSettings.min_views)
    parser.add_argument("--depth-tolerance", type=float, default=LiftSettings.depth_tolerance)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    experiments, renders = ExperimentStore(config.outputs_dir), RenderStore(config.outputs_dir)
    experiment = experiments.open(args.experiment)
    settings = LiftSettings(args.depth_tolerance, LiftSettings.min_alpha, args.min_ratio, args.min_views)
    run = run_lifting(experiment, renders, Segmentation3DStore(config.outputs_dir), settings)
    if args.compare:
        comparison = ComparisonStore(config.outputs_dir).open(args.compare)
        prompt = experiment.load_metadata()["prompt"]
        reference_id = comparison.load_summary()["experiments"].get(f"{prompt}/{REAL}")
        if reference_id is None:
            raise SystemExit(f"比較結果にプロンプト「{prompt}」の実画像の SAM 結果がありません")
        scene = scene_source_of(render_for_experiment(experiment, renders)).load_scene()
        evaluate_reprojection(run, scene, comparison_cameras(comparison), experiments.open(reference_id),
                              GaussianRenderer)
    summary = run.load_summary()
    summary.get("reprojection", {}).pop("per_view_iou", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"→ {run.root}")


if __name__ == "__main__":
    main()
