"""実験単位の入出力を `outputs/<experiment_id>/` に保存する。

レイアウトは docs/spatialsam_design.md §5 に準拠（frames/ は SAM 入力用の 00000.jpg 連番）。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from spatialsam.evaluation.segmentation import FrameScore
from spatialsam.experiments.serialization import build_tracks, load_frame_masks, save_frame_masks
from spatialsam.models.types import SegmentationResult


def new_experiment_id(now: Callable[[], datetime] = datetime.now) -> str:
    return f"{now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class Experiment:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def id(self) -> str:
        return self.root.name

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def masks_dir(self) -> Path:
        return self.root / "prediction" / "masks"

    @property
    def overlay_path(self) -> Path:
        return self.root / "prediction" / "overlay.mp4"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        _write_json(self.root / "metadata.json", metadata)

    def load_metadata(self) -> dict[str, Any]:
        return json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))

    def save_prediction(self, result: SegmentationResult) -> None:
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        for frame in result.frames:
            save_frame_masks(frame, self.masks_dir / f"{frame.frame_index:06d}.npz")
        _write_json(self.root / "prediction" / "tracks.json", build_tracks(result))

    def load_prediction(self) -> SegmentationResult:
        files = sorted(self.masks_dir.glob("*.npz"))
        return SegmentationResult([load_frame_masks(p) for p in files])

    def save_evaluation(self, summary: dict[str, Any], scores: Sequence[FrameScore]) -> None:
        _write_json(self.evaluation_dir / "metrics.json", summary)
        frame = pd.DataFrame([s.as_dict() for s in scores])
        frame.to_csv(self.evaluation_dir / "per_frame.csv", index=False)

    def has_evaluation(self) -> bool:
        return (self.evaluation_dir / "metrics.json").exists()


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def create(self, experiment_id: str | None = None) -> Experiment:
        experiment = Experiment(self.root / (experiment_id or new_experiment_id()))
        experiment.root.mkdir(parents=True, exist_ok=False)
        return experiment

    def open(self, experiment_id: str) -> Experiment:
        experiment = Experiment(self.root / experiment_id)
        if not (experiment.root / "metadata.json").exists():
            raise FileNotFoundError(f"実験が見つかりません: {experiment_id}")
        return experiment

    def list_ids(self) -> list[str]:
        """新しい順。metadata.json を持つものだけ。"""
        if not self.root.exists():
            return []
        ids = [p.name for p in self.root.iterdir() if (p / "metadata.json").exists()]
        return sorted(ids, reverse=True)
