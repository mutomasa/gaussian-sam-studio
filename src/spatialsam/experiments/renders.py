"""3D シーンから描画した動画を `outputs/renders/<render_id>/` に保存する。

- `<render_id>.mp4`: RGB 動画（Video Segmentation の入力になる。実験間で名前が衝突しないよう ID を付ける）
- `depth.mp4`: 深度の可視化動画
- `rgb/00000.png`: 可逆の RGB フレーム（画質比較用）
- `depth/00000.npz`: `depth`（カメラ z 方向の期待深度）と `alpha`（float16）
- `poses.json`: フレームごとのカメラ（K, world_to_cam, width, height）と元画像名
- `metadata.json`: シーン・軌道・性能
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from spatialsam.experiments.store import _write_json, new_experiment_id
from spatialsam.spatial.camera import Camera

RENDERS_DIRNAME = "renders"


class RenderRun:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def id(self) -> str:
        return self.root.name

    @property
    def video_path(self) -> Path:
        return self.root / f"{self.id}.mp4"

    @property
    def depth_video_path(self) -> Path:
        return self.root / "depth.mp4"

    @property
    def rgb_dir(self) -> Path:
        return self.root / "rgb"

    @property
    def depth_dir(self) -> Path:
        return self.root / "depth"

    def rgb_path(self, index: int) -> Path:
        return self.rgb_dir / f"{index:05d}.png"

    def depth_path(self, index: int) -> Path:
        return self.depth_dir / f"{index:05d}.npz"

    def save_depth(self, index: int, depth: np.ndarray, alpha: np.ndarray) -> None:
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.depth_path(index), depth=depth.astype(np.float16), alpha=alpha.astype(np.float16))

    def load_depth(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.depth_path(index)) as data:
            return data["depth"].astype(np.float32), data["alpha"].astype(np.float32)

    def save_poses(self, frames: list[tuple[str, Camera]]) -> None:
        _write_json(self.root / "poses.json",
                    [{"index": i, "source": name, **camera.to_dict()} for i, (name, camera) in enumerate(frames)])

    def load_poses(self) -> list[tuple[str, Camera]]:
        raw = json.loads((self.root / "poses.json").read_text(encoding="utf-8"))
        return [(entry["source"], Camera.from_dict(entry)) for entry in raw]

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        _write_json(self.root / "metadata.json", metadata)

    def load_metadata(self) -> dict[str, Any]:
        return json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))


class RenderStore:
    def __init__(self, outputs_dir: Path) -> None:
        self.root = outputs_dir / RENDERS_DIRNAME

    def create(self, render_id: str | None = None) -> RenderRun:
        run = RenderRun(self.root / (render_id or f"render-{new_experiment_id()}"))
        run.root.mkdir(parents=True, exist_ok=False)
        return run

    def open(self, render_id: str) -> RenderRun:
        run = RenderRun(self.root / render_id)
        if not (run.root / "metadata.json").exists():
            raise FileNotFoundError(f"描画結果が見つかりません: {render_id}")
        return run

    def list_ids(self) -> list[str]:
        """新しい順。完了した（metadata.json を持つ）ものだけ。"""
        if not self.root.exists():
            return []
        return sorted((p.name for p in self.root.iterdir() if (p / "metadata.json").exists()), reverse=True)

    def videos(self) -> list[Path]:
        return [self.open(i).video_path for i in self.list_ids()]
