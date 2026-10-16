import os

import pytest

from spatialsam.jobs import DONE, FAILED, QUEUED, BackgroundJob, JobStore, active_gpu_jobs, run_job_main
from spatialsam.pipeline import comparison_job
from spatialsam.pipeline.comparison_job import ComparisonJobStore, ComparisonRequest
from spatialsam.training.jobs import TrainingStore, TrainRequest
from spatialsam.ui.comparison_panel import parse_prompts, trained_scenes


def test_job_store_lifecycle(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_raw({"x": 1})
    assert job.id.startswith("job-") and store.list_ids() == [job.id]
    assert job.raw_request() == {"x": 1} and job.status()["state"] == QUEUED
    assert not job.is_active() and job.log_tail() == ""
    job.update_status(pid=os.getpid())
    assert store.active_jobs()[0].id == job.id
    with pytest.raises(FileNotFoundError):
        store.open("missing")


def test_run_job_main_records_failure(tmp_path):
    job = JobStore(tmp_path).create_raw({})

    def boom(_):
        raise RuntimeError("壊れた")

    assert run_job_main(job, boom) == 1
    assert job.status()["state"] == FAILED and "RuntimeError: 壊れた" in job.status()["message"]
    assert run_job_main(job, lambda j: j.update_status(state=DONE)) == 0


def test_active_gpu_jobs_covers_training_and_comparison(tmp_path):
    assert active_gpu_jobs(tmp_path) == []
    training = TrainingStore(tmp_path).create(TrainRequest("/s", "/i"))
    comparison = ComparisonJobStore(tmp_path).create(ComparisonRequest("/s", "/i", ["p"], include_points=True))
    for job in (training, comparison):
        job.update_status(pid=os.getpid())
    assert sorted(j.id for j in active_gpu_jobs(tmp_path)) == sorted([training.id, comparison.id])
    comparison.update_status(state=DONE)
    assert [j.id for j in active_gpu_jobs(tmp_path)] == [training.id]


def test_comparison_request_validation_and_steps():
    with pytest.raises(ValueError):
        ComparisonRequest("/s", "/i", [], include_points=True)
    with pytest.raises(ValueError):
        ComparisonRequest("/s", "/i", ["p"])
    request = ComparisonRequest("/s", "/i", ["a", "b"], {"3dgs": "/x.ply"}, include_points=True)
    assert request.num_steps() == 2 + 2 * 3  # 描画 2 + SAM（実画像 + 2 シーン）× 2 プロンプト


def test_comparison_job_failure_is_recorded(tmp_path):
    job = ComparisonJobStore(tmp_path).create(ComparisonRequest(str(tmp_path / "none"), str(tmp_path), ["p"],
                                                                include_points=True))
    assert comparison_job.main([str(job.root)]) == 1
    assert job.status()["state"] == FAILED and job.comparison_id is None


def test_parse_prompts():
    assert parse_prompts("bonsai, table、 bonsai\nchair,,") == ["bonsai", "table", "chair"]
    assert parse_prompts("  ") == []


def test_trained_scenes_filters_by_colmap(tmp_path):
    store = TrainingStore(tmp_path)
    sparse = tmp_path / "scene" / "sparse"
    sparse.mkdir(parents=True)
    done = store.create(TrainRequest(str(sparse), "/i"))
    done.update_status(state=DONE)
    done.ply_path.write_bytes(b"ply")
    other = store.create(TrainRequest(str(tmp_path / "other"), "/i"))
    other.update_status(state=DONE)
    other.ply_path.write_bytes(b"ply")
    running = store.create(TrainRequest(str(sparse), "/i"))
    assert trained_scenes(store, str(sparse)) == {done.id: str(done.ply_path)}
    assert trained_scenes(store, "") == {}
    assert running.id not in trained_scenes(store, str(sparse))


def test_background_job_log_tail(tmp_path):
    job = BackgroundJob(tmp_path)
    job.log_path.write_text("a" * 10 + "b" * 5, encoding="utf-8")
    assert job.log_tail(5) == "bbbbb"
