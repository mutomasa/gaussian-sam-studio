import math
from types import SimpleNamespace

import pytest

from spatialsam import tracking
from spatialsam.tracking import (
    ID_TAG,
    MlflowTracker,
    NullTracker,
    TrackingSettings,
    flatten,
    resolve_uri,
    safe_log,
    to_metrics,
    to_params,
)


class FakeClient:
    """MlflowClient のうち使う部分だけ。"""

    def __init__(self):
        self.experiments, self.runs, self.batches, self.artifacts, self.terminated = {}, {}, [], [], []

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id=self.experiments[name]) if name in self.experiments else None

    def create_experiment(self, name):
        self.experiments[name] = str(len(self.experiments))
        return self.experiments[name]

    def search_runs(self, experiment_ids, filter_string, max_results):
        record_id = filter_string.split("= '")[1].rstrip("'")
        return [SimpleNamespace(info=SimpleNamespace(run_id=r)) for r, (e, tags) in self.runs.items()
                if e in experiment_ids and tags[ID_TAG] == record_id]

    def create_run(self, experiment_id, tags, run_name):
        run_id = f"run{len(self.runs)}"
        self.runs[run_id] = (experiment_id, tags)
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def log_batch(self, run_id, metrics, params):
        self.batches.append((run_id, metrics, params))

    def log_artifact(self, run_id, path):
        self.artifacts.append((run_id, path))

    def set_terminated(self, run_id):
        self.terminated.append(run_id)


def test_flatten_params_and_metrics():
    values = {"a": {"b": 1, "c": {"d": "x"}}, "e": None, "f": True, "g": float("nan"), "h": 2.5}
    assert flatten(values) == {"a.b": 1, "a.c.d": "x", "e": None, "f": True, "g": values["g"], "h": 2.5}
    assert to_params(values) == {"a.b": "1", "a.c.d": "x", "f": "True", "g": "nan", "h": "2.5"}
    assert to_metrics(values) == {"a.b": 1.0, "h": 2.5}
    assert len(to_params({"long": "x" * 1000})["long"]) == tracking.MAX_PARAM_LENGTH
    assert to_metrics({"eval": {"J&F": 0.9}, "sam.盆栽?": 1}) == {"eval.J_and_F": 0.9, "sam.盆栽_": 1.0}


def test_mlflow_tracker_reuses_run_and_logs_params_once(tmp_path):
    client = FakeClient()
    tracker = MlflowTracker("http://x", client=client)
    artifact = tmp_path / "a.json"
    artifact.write_text("{}")
    tracker.log("segmentation", "exp1", params={"prompt": "swan"}, metrics={"J": 0.9},
                artifacts=[artifact, tmp_path / "missing"], series={"loss": [(1, 0.5), (2, math.inf), (3, 0.2)]})
    tracker.log("segmentation", "exp1", params={"prompt": "changed"}, metrics={"eval.J": 0.95})
    assert list(client.experiments) == ["spatialsam/segmentation"] and len(client.runs) == 1
    (run1, metrics1, params1), (run2, metrics2, params2) = client.batches
    assert run1 == run2 == "run0"
    assert [p.key for p in params1] == ["prompt"] and params2 == []  # param は最初の 1 回だけ（変更できないため）
    assert sorted((m.key, m.step) for m in metrics1) == [("J", 0), ("loss", 1), ("loss", 3)]
    assert client.artifacts == [("run0", str(artifact))] and client.terminated == ["run0", "run0"]


def test_resolve_uri_priority(monkeypatch):
    settings = TrackingSettings(mlflow_uri="http://config:5000")
    monkeypatch.delenv(tracking.DISABLE_ENV, raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_uri(settings) == "http://config:5000"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")
    assert resolve_uri(settings) == "http://env:5000"
    monkeypatch.setenv(tracking.DISABLE_ENV, "1")
    assert resolve_uri(settings) is None


def test_unreachable_server_falls_back_to_null(monkeypatch):
    monkeypatch.delenv(tracking.DISABLE_ENV, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")  # 誰も待ち受けていないポート
    tracking.get_tracker.cache_clear()
    with pytest.warns(UserWarning, match="接続できない"):
        assert isinstance(tracking.get_tracker(), NullTracker)


def test_safe_log_never_raises(monkeypatch):
    class Broken:
        def log(self, *args, **kwargs):
            raise RuntimeError("down")

    monkeypatch.setattr(tracking, "get_tracker", lambda: Broken())
    with pytest.warns(UserWarning, match="記録に失敗"):
        safe_log("segmentation", "exp1", metrics={"J": 1.0})


def test_disabled_by_default_in_tests():
    assert isinstance(tracking.get_tracker(), NullTracker)
