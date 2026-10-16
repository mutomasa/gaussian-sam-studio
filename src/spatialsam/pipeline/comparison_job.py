"""実画像 vs 3D 描画の比較を別プロセスで実行するジョブ #32（共通の仕組みは spatialsam.jobs）。

SAM 3.1 を「(ソース数 + 1) × プロンプト数」本回すので数分かかる。UI からはジョブとして起動し、
進捗は status.json で受け取る。結果そのものは outputs/comparisons/<comparison_id>/ に保存される。

    python -m spatialsam.pipeline.comparison_job outputs/comparison_jobs/<job_id>
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from spatialsam.jobs import DONE, RUNNING, BackgroundJob, JobStore, launch, run_job_main

RUN_MODULE = "spatialsam.pipeline.comparison_job"


@dataclass(frozen=True)
class ComparisonRequest:
    sparse_dir: str
    images_dir: str
    prompts: list[str]
    sources: dict[str, str] = field(default_factory=dict)  # 名前 → 3DGS の PLY
    include_points: bool = False                          # COLMAP 点群（学習なし）も比べる
    holdout_every: int = 8

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError("プロンプトを 1 つ以上指定してください")
        if not self.sources and not self.include_points:
            raise ValueError("比べるシーン（3DGS の PLY か点群）を 1 つ以上指定してください")

    def num_steps(self) -> int:
        """進捗表示用: 描画するシーン数 + SAM の実行回数。"""
        num_sources = len(self.sources) + int(self.include_points)
        return num_sources + len(self.prompts) * (num_sources + 1)


class ComparisonRunJob(BackgroundJob):
    def load_request(self) -> ComparisonRequest:
        return ComparisonRequest(**self.raw_request())

    @property
    def comparison_id(self) -> str | None:
        return self.status().get("comparison_id")


class ComparisonJobStore(JobStore):
    dirname = "comparison_jobs"
    prefix = "cmpjob"
    job_class = ComparisonRunJob

    def create(self, request: ComparisonRequest) -> ComparisonRunJob:
        return self.create_raw(asdict(request))


def launch_comparison(job: ComparisonRunJob) -> int:
    return launch(job, RUN_MODULE)


def build_sources(request: ComparisonRequest):
    from spatialsam.spatial.scene_source import SceneSource

    sparse = Path(request.sparse_dir)
    sources = {name: SceneSource(Path(ply), sparse) for name, ply in request.sources.items()}
    if request.include_points:
        sources["points"] = SceneSource(sparse_dir=sparse)
    return sources


def run(job: ComparisonRunJob) -> None:
    from spatialsam.config import load_config
    from spatialsam.evaluation.image_quality import ImageQualityScorer
    from spatialsam.experiments.comparisons import ComparisonStore
    from spatialsam.experiments.store import ExperimentStore
    from spatialsam.models.factory import create_segmenter
    from spatialsam.pipeline.comparison import ComparisonJob, run_comparison
    from spatialsam.spatial.renderer import GaussianRenderer
    from spatialsam.training.dataset import ColmapDataset

    request = job.load_request()
    config = load_config()
    job.update_status(state=RUNNING, message="評価用視点の画像を読み込んでいます", step=0, total=request.num_steps())
    views = ColmapDataset.load(Path(request.sparse_dir), Path(request.images_dir), request.holdout_every).test
    if not views:
        raise ValueError("評価用視点がありません（評価用に除く間隔を確認してください）")
    comparison_job = ComparisonJob(views, build_sources(request), request.prompts,
                                   info={"sparse_dir": request.sparse_dir, "images_dir": request.images_dir,
                                         "holdout_every": request.holdout_every, "segmenter": "sam3.1",
                                         "job": job.id})
    step = 0

    def progress(message: str) -> None:
        nonlocal step
        step += 1
        job.update_status(message=message, step=step)
        print(message, flush=True)

    segmenter = create_segmenter("sam3.1", config)
    try:
        comparison = run_comparison(comparison_job, ComparisonStore(config.outputs_dir),
                                    ExperimentStore(config.outputs_dir), segmenter, ImageQualityScorer(),
                                    lambda source: GaussianRenderer(source.load_scene()), progress)
    finally:
        segmenter.release()
    job.update_status(state=DONE, message="完了", comparison_id=comparison.id, step=request.num_steps())


def main(argv: list[str] | None = None) -> int:
    return run_job_main(ComparisonRunJob(Path((argv or sys.argv[1:])[0])), run)


if __name__ == "__main__":
    sys.exit(main())
