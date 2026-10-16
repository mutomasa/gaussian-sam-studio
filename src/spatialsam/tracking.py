"""MLflow（Docker のトラッキングサーバー）への実験の記録 #45。

ローカルの outputs/ が正の保存先で、MLflow は一覧・比較のための追加の記録先。
- 種類ごとに MLflow の experiment を分ける（`spatialsam/segmentation`、`spatialsam/training` など）
- 1 つの記録（SAM の実験 ID、学習ジョブ ID など）を 1 つの run にし、タグ `spatialsam.id` で引き当てる。
  後から評価したときは同じ run に指標を足す
- 宛先は環境変数 MLFLOW_TRACKING_URI、無ければ configs/model.yaml の tracking.mlflow_uri
- 無効のとき、サーバーに届かないとき、記録に失敗したときは警告だけ出して続ける（アプリは止めない）
- テストなどで止めたいときは SPATIALSAM_DISABLE_TRACKING=1
"""

from __future__ import annotations

import math
import os
import re
import time
import urllib.request
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

EXPERIMENT_PREFIX = "spatialsam"
ID_TAG = "spatialsam.id"
DISABLE_ENV = "SPATIALSAM_DISABLE_TRACKING"
MAX_PARAM_LENGTH = 500


@dataclass(frozen=True)
class TrackingSettings:
    mlflow_uri: str | None = None
    log_artifacts: bool = True  # 重畳動画などのファイルも送る（大きい PLY は送らない）
    timeout_sec: float = 1.5    # サーバーの生存確認の待ち時間


def flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """入れ子の dict を `a.b` のキーに平らにする（MLflow の param / metric は平らな key-value）。"""
    flat = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat |= flatten(value, f"{name}.")
        else:
            flat[name] = value
    return flat


def safe_key(key: str) -> str:
    """MLflow のキーに使える文字（英数字・_ - . 空白 : /）だけにする（例: J&F → J_and_F）。"""
    return re.sub(r"[^\w\-. :/]", "_", key.replace("&", "_and_"))


def to_params(values: Mapping[str, Any]) -> dict[str, str]:
    return {safe_key(k): str(v)[:MAX_PARAM_LENGTH] for k, v in flatten(values).items() if v is not None}


def to_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    """数値だけ（bool・None・NaN・inf は除く）。"""
    return {safe_key(k): float(v) for k, v in flatten(values).items()
            if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)}


class Tracker(Protocol):
    def log(self, kind: str, record_id: str, params: Mapping[str, Any] = ..., metrics: Mapping[str, Any] = ...,
            artifacts: Iterable[Path] = ..., series: Mapping[str, list[tuple[int, float]]] = ...) -> None: ...


class NullTracker:
    """記録しない（無効のとき、サーバーに届かないとき）。"""

    enabled = False

    def log(self, kind, record_id, params=None, metrics=None, artifacts=(), series=None) -> None:
        return None


class MlflowTracker:
    enabled = True

    def __init__(self, uri: str, log_artifacts: bool = True, client=None) -> None:
        if client is None:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=uri)
        self.uri = uri
        self.client = client
        self.log_artifacts = log_artifacts

    def _experiment_id(self, kind: str) -> str:
        name = f"{EXPERIMENT_PREFIX}/{kind}"
        experiment = self.client.get_experiment_by_name(name)
        return experiment.experiment_id if experiment else self.client.create_experiment(name)

    def _run_id(self, experiment_id: str, record_id: str) -> tuple[str, bool]:
        """(run_id, 新しく作ったか)。同じ記録の run があればそれに足す。"""
        found = self.client.search_runs([experiment_id], filter_string=f"tags.`{ID_TAG}` = '{record_id}'",
                                        max_results=1)
        if found:
            return found[0].info.run_id, False
        run = self.client.create_run(experiment_id, tags={ID_TAG: record_id}, run_name=record_id)
        return run.info.run_id, True

    def log(self, kind, record_id, params=None, metrics=None, artifacts=(), series=None) -> None:
        from mlflow.entities import Metric, Param

        run_id, created = self._run_id(self._experiment_id(kind), record_id)
        now = int(time.time() * 1000)
        metric_list = [Metric(k, v, now, 0) for k, v in to_metrics(metrics or {}).items()]
        for name, points in (series or {}).items():
            metric_list += [Metric(safe_key(name), float(v), now, int(step)) for step, v in points if math.isfinite(v)]
        param_list = [Param(k, v) for k, v in to_params(params or {}).items()] if created else []
        self.client.log_batch(run_id, metrics=metric_list, params=param_list)
        if self.log_artifacts:
            for path in artifacts:
                if Path(path).exists():
                    self.client.log_artifact(run_id, str(path))
        self.client.set_terminated(run_id)


def server_alive(uri: str, timeout: float) -> bool:
    if not uri.startswith(("http://", "https://")):
        return True  # file: などローカルの宛先は確認しない
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout) as response:
            return response.status == 200
    except OSError:
        return False


def resolve_uri(settings: TrackingSettings) -> str | None:
    if os.environ.get(DISABLE_ENV):
        return None
    return os.environ.get("MLFLOW_TRACKING_URI") or settings.mlflow_uri


@lru_cache(maxsize=1)
def get_tracker() -> Tracker:
    """プロセスごとに 1 回だけ宛先を決めて生存を確認する。"""
    from spatialsam.config import load_config

    settings = load_config().tracking
    uri = resolve_uri(settings)
    if not uri:
        return NullTracker()
    if not server_alive(uri, settings.timeout_sec):
        warnings.warn(f"MLflow サーバーに接続できないので記録しません: {uri}"
                      "（docker compose -f docker/mlflow/compose.yaml up -d）", stacklevel=2)
        return NullTracker()
    return MlflowTracker(uri, settings.log_artifacts)


def safe_log(kind: str, record_id: str, **payload: Any) -> None:
    """記録に失敗しても処理は続ける（MLflow は追加の記録先なので）。"""
    try:
        get_tracker().log(kind, record_id, **payload)
    except Exception as error:  # noqa: BLE001
        warnings.warn(f"MLflow への記録に失敗しました（{kind}/{record_id}）: {error}", stacklevel=2)
