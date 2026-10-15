"""3D セグメンテーション（2D マスクを Gaussian に持ち上げた結果）を `outputs/segmentation3d/<id>/` に保存する。

- `labels.npz`: `visible`（見えた視点数）、`confidence_<物体ID>`（float16）、`labels_<物体ID>`（bool）
- `object.ply`: いずれかの物体に属する Gaussian だけを抜き出した 3DGS
- `summary.json`: 入力（SAM 実験・描画結果・シーン）、設定、物体ごとの Gaussian 数、評価
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from spatialsam.experiments.store import _write_json, new_experiment_id

SEGMENTATION3D_DIRNAME = "segmentation3d"


class Segmentation3D:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def id(self) -> str:
        return self.root.name

    @property
    def object_ply_path(self) -> Path:
        return self.root / "object.ply"

    def save_labels(self, visible: np.ndarray, confidence: dict[int, np.ndarray], labels: dict[int, np.ndarray]) -> None:
        arrays = {"visible": visible.astype(np.int32)}
        arrays |= {f"confidence_{k}": v.astype(np.float16) for k, v in confidence.items()}
        arrays |= {f"labels_{k}": v.astype(bool) for k, v in labels.items()}
        np.savez_compressed(self.root / "labels.npz", **arrays)

    def load_labels(self) -> dict[int, np.ndarray]:
        with np.load(self.root / "labels.npz") as data:
            return {int(k.split("_", 1)[1]): data[k] for k in data.files if k.startswith("labels_")}

    def load_confidence(self) -> dict[int, np.ndarray]:
        with np.load(self.root / "labels.npz") as data:
            return {int(k.split("_", 1)[1]): data[k].astype(np.float32) for k in data.files
                    if k.startswith("confidence_")}

    def any_labels(self) -> np.ndarray:
        labels = list(self.load_labels().values())
        return np.logical_or.reduce(labels) if labels else np.zeros(0, bool)

    def save_summary(self, summary: dict[str, Any]) -> None:
        _write_json(self.root / "summary.json", summary)

    def load_summary(self) -> dict[str, Any]:
        return json.loads((self.root / "summary.json").read_text(encoding="utf-8"))

    def update_summary(self, **changes: Any) -> None:
        self.save_summary(self.load_summary() | changes)


class Segmentation3DStore:
    def __init__(self, outputs_dir: Path) -> None:
        self.root = outputs_dir / SEGMENTATION3D_DIRNAME

    def create(self) -> Segmentation3D:
        run = Segmentation3D(self.root / f"seg3d-{new_experiment_id()}")
        run.root.mkdir(parents=True, exist_ok=False)
        return run

    def open(self, run_id: str) -> Segmentation3D:
        run = Segmentation3D(self.root / run_id)
        if not (run.root / "summary.json").exists():
            raise FileNotFoundError(f"3D セグメンテーションが見つかりません: {run_id}")
        return run

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted((p.name for p in self.root.iterdir() if (p / "summary.json").exists()), reverse=True)
