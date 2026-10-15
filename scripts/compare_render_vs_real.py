"""実画像 vs 3D シーンの描画画像を、評価用視点で比べる（画質と SAM 3.1）#16。

    uv run python scripts/compare_render_vs_real.py --sparse <scene>/sparse/0 --images <scene>/images_4 \
        --source 3dgs=outputs/training/<job_id>/point_cloud.ply --points --prompt bonsai --prompt table

- 評価用視点は学習と同じ規則（画像名順で 8 枚に 1 枚）。3DGS の学習に使っていない視点で比べる
- `--points` で COLMAP 点群（学習なし）も比較に加える
- 結果は outputs/comparisons/<id>/（SAM の各実験は outputs/<experiment_id>/）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatialsam.config import load_config
from spatialsam.evaluation.image_quality import ImageQualityScorer
from spatialsam.experiments.comparisons import ComparisonStore
from spatialsam.experiments.store import ExperimentStore
from spatialsam.models.factory import SEGMENTER_NAMES, create_segmenter
from spatialsam.pipeline.comparison import ComparisonJob, run_comparison
from spatialsam.spatial.renderer import GaussianRenderer
from spatialsam.spatial.scene_source import SceneSource
from spatialsam.training.dataset import ColmapDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="実画像（学習に使ったのと同じ縮小率）")
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PLY", help="比べる 3D シーン（複数可）")
    parser.add_argument("--points", action="store_true", help="COLMAP 点群を学習せずに描いたものも比べる")
    parser.add_argument("--prompt", action="append", required=True, help="SAM のテキストプロンプト（複数可）")
    parser.add_argument("--holdout-every", type=int, default=8)
    parser.add_argument("--segmenter", choices=SEGMENTER_NAMES, default="sam3.1")
    parser.add_argument("--no-lpips", action="store_true")
    return parser.parse_args()


def build_sources(args: argparse.Namespace) -> dict[str, SceneSource]:
    sources = {}
    for item in args.source:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--source は NAME=PLY の形式で指定してください: {item}")
        sources[name] = SceneSource(Path(path), args.sparse)
    if args.points:
        sources["points"] = SceneSource(sparse_dir=args.sparse)
    if not sources:
        raise SystemExit("--source か --points を指定してください")
    return sources


def main() -> None:
    args = parse_args()
    config = load_config()
    views = ColmapDataset.load(args.sparse, args.images, args.holdout_every).test
    if not views:
        raise SystemExit("評価用視点がありません（--holdout-every を確認）")
    job = ComparisonJob(views, build_sources(args), args.prompt,
                        info={"sparse_dir": str(args.sparse), "images_dir": str(args.images),
                              "holdout_every": args.holdout_every, "segmenter": args.segmenter})
    segmenter = create_segmenter(args.segmenter, config)
    try:
        comparison = run_comparison(job, ComparisonStore(config.outputs_dir), ExperimentStore(config.outputs_dir),
                                    segmenter, ImageQualityScorer(use_lpips=not args.no_lpips),
                                    lambda source: GaussianRenderer(source.load_scene()), print)
    finally:
        segmenter.release()
    summary = comparison.load_summary()
    print(json.dumps({k: summary[k] for k in ("quality", "segmentation")}, ensure_ascii=False, indent=2))
    print(f"→ {comparison.root}")


if __name__ == "__main__":
    main()
