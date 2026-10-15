"""3DGS 学習ジョブ。Streamlit とは別プロセスで動かし、進捗はファイルで受け渡す。

`outputs/training/<job_id>/`
- `request.json`: 入力（COLMAP / 画像ディレクトリ）と学習設定
- `status.json`: 状態（queued / running / done / failed）、PID、進捗、エラー
- `train.log`: 子プロセスの標準出力・標準エラー
- `point_cloud.ply`: 学習結果（標準 3DGS PLY）
- `metrics.json`: 評価用視点（学習に使っていない）の PSNR / SSIM
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from spatialsam.experiments.store import new_experiment_id

TRAINING_DIRNAME = "training"
QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


@dataclass(frozen=True)
class TrainRequest:
    sparse_dir: str
    images_dir: str
    config: dict[str, Any] = field(default_factory=dict)  # TrainConfig のうち既定値から変える項目
    holdout_every: int = 8


def write_json_atomic(path: Path, payload: Any) -> None:
    """読み手（UI）が書きかけのファイルを読まないよう、一時ファイルから置き換える。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TrainingJob:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def id(self) -> str:
        return self.root.name

    @property
    def ply_path(self) -> Path:
        return self.root / "point_cloud.ply"

    @property
    def log_path(self) -> Path:
        return self.root / "train.log"

    def load_request(self) -> TrainRequest:
        return TrainRequest(**json.loads((self.root / "request.json").read_text(encoding="utf-8")))

    def status(self) -> dict[str, Any]:
        path = self.root / "status.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"state": QUEUED}

    def update_status(self, **changes: Any) -> dict[str, Any]:
        status = self.status() | changes | {"updated_at": datetime.now().isoformat(timespec="seconds")}
        write_json_atomic(self.root / "status.json", status)
        return status

    def is_active(self) -> bool:
        """実行中で、プロセスも生きている（異常終了したまま running が残るのを除く）。"""
        status = self.status()
        return status["state"] in (QUEUED, RUNNING) and pid_alive(status.get("pid"))

    def metrics(self) -> dict[str, Any]:
        path = self.root / "metrics.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class TrainingStore:
    def __init__(self, outputs_dir: Path) -> None:
        self.root = outputs_dir / TRAINING_DIRNAME

    def create(self, request: TrainRequest, job_id: str | None = None) -> TrainingJob:
        job = TrainingJob(self.root / (job_id or f"train-{new_experiment_id()}"))
        job.root.mkdir(parents=True, exist_ok=False)
        write_json_atomic(job.root / "request.json", asdict(request))
        job.update_status(state=QUEUED)
        return job

    def open(self, job_id: str) -> TrainingJob:
        job = TrainingJob(self.root / job_id)
        if not (job.root / "request.json").exists():
            raise FileNotFoundError(f"学習ジョブが見つかりません: {job_id}")
        return job

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted((p.name for p in self.root.iterdir() if (p / "request.json").exists()), reverse=True)

    def active_jobs(self) -> list[TrainingJob]:
        return [job for job in map(self.open, self.list_ids()) if job.is_active()]


def launch(job: TrainingJob) -> int:
    """子プロセスで学習を始め、PID を返す。Streamlit の再実行やタブを閉じても止まらない。"""
    with job.log_path.open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "spatialsam.training.run_job", str(job.root)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    job.update_status(pid=process.pid)
    return process.pid
