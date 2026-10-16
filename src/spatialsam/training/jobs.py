"""3DGS 学習ジョブ（共通の仕組みは spatialsam.jobs）。

`outputs/training/<job_id>/`
- `request.json`: 入力（COLMAP / 画像ディレクトリ）と学習設定
- `status.json`: 状態、PID、進捗、エラー
- `train.log`: 子プロセスの標準出力・標準エラー
- `point_cloud.ply`: 学習結果（標準 3DGS PLY）
- `metrics.json`: 評価用視点（学習に使っていない）の PSNR / SSIM
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from spatialsam.jobs import DONE, FAILED, QUEUED, RUNNING, BackgroundJob, JobStore, pid_alive, write_json_atomic
from spatialsam.jobs import launch as launch_module

__all__ = ["DONE", "FAILED", "QUEUED", "RUNNING", "TrainRequest", "TrainingJob", "TrainingStore", "launch",
           "pid_alive", "write_json_atomic"]

TRAINING_DIRNAME = "training"
RUN_MODULE = "spatialsam.training.run_job"


@dataclass(frozen=True)
class TrainRequest:
    sparse_dir: str
    images_dir: str
    config: dict[str, Any] = field(default_factory=dict)  # TrainConfig のうち既定値から変える項目
    holdout_every: int = 8


class TrainingJob(BackgroundJob):
    log_name = "train.log"

    @property
    def ply_path(self):
        return self.root / "point_cloud.ply"

    def load_request(self) -> TrainRequest:
        return TrainRequest(**self.raw_request())

    def metrics(self) -> dict[str, Any]:
        path = self.root / "metrics.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class TrainingStore(JobStore):
    dirname = TRAINING_DIRNAME
    prefix = "train"
    job_class = TrainingJob

    def create(self, request: TrainRequest, job_id: str | None = None) -> TrainingJob:
        return self.create_raw(asdict(request), job_id)

    def done_jobs(self) -> list[TrainingJob]:
        return [job for job in map(self.open, self.list_ids()) if job.status()["state"] == DONE]


def launch(job: TrainingJob) -> int:
    return launch_module(job, RUN_MODULE)
