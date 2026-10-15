"""COLMAP（画像 + sparse）から gsplat で 3DGS を学習する。

    uv run python scripts/train_3dgs.py --sparse data/bonsai/sparse/0 --images data/bonsai/images_4 \
        [--steps 7000] [--cap-max 1000000] [--background]

結果は outputs/training/<job_id>/point_cloud.ply。UI（3D Rendering → 3DGS 学習）からも同じジョブを起動できる。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialsam.config import load_config
from spatialsam.training import run_job
from spatialsam.training.jobs import TrainingStore, TrainRequest, launch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sparse", type=Path, required=True, help="cameras.bin / images.bin / points3D.bin のディレクトリ")
    parser.add_argument("--images", type=Path, required=True, help="学習画像のディレクトリ（縮小版でよい）")
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--cap-max", type=int, default=1_000_000, help="Gaussian 数の上限")
    parser.add_argument("--holdout-every", type=int, default=8, help="評価用に除く間隔（0 = 全部学習）")
    parser.add_argument("--background", action="store_true", help="別プロセスで起動してすぐ戻る")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = TrainRequest(str(args.sparse.resolve()), str(args.images.resolve()),
                           {"steps": args.steps, "cap_max": args.cap_max}, args.holdout_every)
    job = TrainingStore(load_config().outputs_dir).create(request)
    print(f"ジョブ: {job.root}")
    if args.background:
        print(f"PID {launch(job)}。進捗は {job.log_path}")
        return 0
    return run_job.main([str(job.root)])


if __name__ == "__main__":
    raise SystemExit(main())
