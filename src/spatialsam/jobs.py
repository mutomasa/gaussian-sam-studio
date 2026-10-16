"""Streamlit とは別プロセスで動かす重い処理（3DGS 学習、実画像 vs 3D 描画の比較）の共通部分。

`outputs/<種類>/<job_id>/`
- `request.json`: 入力
- `status.json`: 状態（queued / running / done / failed）、PID、進捗、メッセージ
- `job.log`（学習は `train.log`）: 子プロセスの標準出力・標準エラー

子プロセスは `python -m <module> <job_dir>` で起動し、新しいセッションにするので Streamlit を止めても続く。
GPU を使うジョブと SAM の推論は同時に動かさない（8GB VRAM に収まらないため）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from spatialsam.experiments.store import new_experiment_id

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


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


class BackgroundJob:
    log_name = "job.log"

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def id(self) -> str:
        return self.root.name

    @property
    def log_path(self) -> Path:
        return self.root / self.log_name

    def raw_request(self) -> dict[str, Any]:
        return json.loads((self.root / "request.json").read_text(encoding="utf-8"))

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

    def log_tail(self, chars: int = 4000) -> str:
        return self.log_path.read_text(encoding="utf-8")[-chars:] if self.log_path.exists() else ""


class JobStore:
    """`outputs/<dirname>/` 以下のジョブ。job_class でジョブの種類ごとの機能を足す。"""

    dirname = "jobs"
    prefix = "job"
    job_class: type[BackgroundJob] = BackgroundJob

    def __init__(self, outputs_dir: Path) -> None:
        self.root = outputs_dir / self.dirname

    def create_raw(self, request: dict[str, Any], job_id: str | None = None) -> BackgroundJob:
        job = self.job_class(self.root / (job_id or f"{self.prefix}-{new_experiment_id()}"))
        job.root.mkdir(parents=True, exist_ok=False)
        write_json_atomic(job.root / "request.json", request)
        job.update_status(state=QUEUED)
        return job

    def open(self, job_id: str) -> BackgroundJob:
        job = self.job_class(self.root / job_id)
        if not (job.root / "request.json").exists():
            raise FileNotFoundError(f"ジョブが見つかりません: {job_id}")
        return job

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted((p.name for p in self.root.iterdir() if (p / "request.json").exists()), reverse=True)

    def active_jobs(self) -> list[BackgroundJob]:
        return [job for job in map(self.open, self.list_ids()) if job.is_active()]


def launch(job: BackgroundJob, module: str) -> int:
    """子プロセスでジョブを始め、PID を返す。Streamlit の再実行やタブを閉じても止まらない。"""
    with job.log_path.open("ab") as log:
        process = subprocess.Popen([sys.executable, "-m", module, str(job.root)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    job.update_status(pid=process.pid)
    return process.pid


def run_job_main(job: BackgroundJob, run) -> int:
    """子プロセスの入口の共通処理: 失敗したら status に残して終了コード 1 を返す。"""
    import traceback

    try:
        run(job)
    except Exception as error:  # noqa: BLE001 — 失敗内容を UI に伝えて終了する
        traceback.print_exc()
        job.update_status(state=FAILED, message=f"{type(error).__name__}: {error}")
        return 1
    return 0


def active_gpu_jobs(outputs_dir: Path) -> list[BackgroundJob]:
    """GPU を使っている別プロセスのジョブ（学習・比較）。SAM の推論や新しいジョブはこれが空のときだけ。"""
    from spatialsam.pipeline.comparison_job import ComparisonJobStore
    from spatialsam.training.jobs import TrainingStore

    return TrainingStore(outputs_dir).active_jobs() + ComparisonJobStore(outputs_dir).active_jobs()
