"""GT 付きサンプル（samples/）を一括で推論・評価して CSV にまとめる。

    uv run python scripts/benchmark_samples.py [--segmenter sam3.1] [--samples blackswan cows]
        [--max-frames 50] [--max-side 640] [--fp32-backbone]

結果は通常の実験として outputs/ に保存され（UI の評価ページでも見られる）、
一覧は outputs/benchmarks/<日時>.csv に書き出す。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime

import pandas as pd

from spatialsam.config import PROJECT_ROOT, AppConfig, load_config
from spatialsam.experiments.benchmark import run_sample
from spatialsam.experiments.store import ExperimentStore
from spatialsam.models.factory import SEGMENTER_NAMES, create_segmenter
from spatialsam.samples.davis import load_manifest
from spatialsam.video.io import FrameSampling

MANIFEST = PROJECT_ROOT / "configs" / "samples.yaml"
SAMPLES_DIR = PROJECT_ROOT / "samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segmenter", choices=SEGMENTER_NAMES, default="sam3.1")
    parser.add_argument("--samples", nargs="*", help="シーケンス名（省略時は全サンプル）")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-side", type=int)
    parser.add_argument("--fp32-backbone", action="store_true", help="bf16_backbone を無効にする（精度比較用）")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    config = load_config()
    if args.fp32_backbone:
        config = replace(config, sam31=replace(config.sam31, bf16_backbone=False))
    return config


def model_info(config: AppConfig, segmenter_name: str) -> dict:
    if segmenter_name != "sam3.1":
        return {}
    source = config.sam31_source
    return {"repo_id": source.repo_id, "revision": source.revision, "bf16_backbone": config.sam31.bf16_backbone}


def main() -> None:
    args = parse_args()
    config = build_config(args)
    specs = [s for s in load_manifest(MANIFEST).samples if not args.samples or s.sequence in args.samples]
    sampling = FrameSampling(args.stride, args.max_frames, args.max_side)
    segmenter = create_segmenter(args.segmenter, config)
    store = ExperimentStore(config.outputs_dir)
    rows = []
    try:
        for spec in specs:
            rows.append(run_sample(spec, SAMPLES_DIR, sampling, segmenter, store,
                                   model_info(config, args.segmenter), lambda m, s=spec: print(f"[{s.sequence}] {m}")))
            print(rows[-1])
    finally:
        segmenter.release()
    out = config.outputs_dir / "benchmarks" / f"{datetime.now():%Y%m%d-%H%M%S}-{args.segmenter}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
