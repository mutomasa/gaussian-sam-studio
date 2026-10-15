"""学習ジョブの実行（子プロセスの入口）。

    python -m spatialsam.training.run_job outputs/training/<job_id>
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from spatialsam.spatial.colmap import read_points3d
from spatialsam.spatial.gaussians import save_ply
from spatialsam.training.jobs import DONE, FAILED, RUNNING, TrainingJob, write_json_atomic


def run(job: TrainingJob) -> None:
    import torch

    from spatialsam.training.dataset import ColmapDataset
    from spatialsam.training.trainer import TrainConfig, Trainer, TrainProgress

    request = job.load_request()
    config = TrainConfig(**request.config)
    job.update_status(state=RUNNING, pid=os.getpid(), config=config.to_dict(), message="データを読み込んでいます")
    sparse_dir, images_dir = Path(request.sparse_dir), Path(request.images_dir)
    dataset = ColmapDataset.load(sparse_dir, images_dir, request.holdout_every)
    points = read_points3d(sparse_dir / "points3D.bin")
    torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(dataset, points.xyz, points.rgb, config)

    def report(progress: TrainProgress) -> None:
        peak = torch.cuda.max_memory_allocated() / 2**20
        job.update_status(progress=asdict(progress) | {"peak_vram_mib": peak}, message="学習中")
        print(f"step {progress.step}/{progress.total} loss {progress.loss:.4f} psnr {progress.psnr:.2f} "
              f"gaussians {progress.num_gaussians} peak {peak:.0f}MiB", flush=True)

    trainer.fit(report)
    job.update_status(message="評価用視点で評価しています")
    metrics = trainer.evaluate(dataset.test) | {"num_train_views": len(dataset.train)}
    save_ply(trainer.export(), job.ply_path)
    write_json_atomic(job.root / "metrics.json", metrics)
    job.update_status(state=DONE, message="完了", peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20)


def main(argv: list[str] | None = None) -> int:
    job = TrainingJob(Path((argv or sys.argv[1:])[0]))
    try:
        run(job)
    except Exception as error:  # noqa: BLE001 — 失敗内容を UI に伝えて終了する
        traceback.print_exc()
        job.update_status(state=FAILED, message=f"{type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
