"""実画像 vs 描画画像の比較結果を `outputs/comparisons/<comparison_id>/` に保存する。

- `frames/<source>/00000.png`: 各ソースの画像（`real` は実画像、ほかは同じ視点の描画）
- `videos/<source>.mp4`: SAM の入力にした動画（視点の順は画像名順）
- `quality.csv`: 描画ソースごと・視点ごとの PSNR / SSIM / LPIPS（実画像が基準）
- `masks.csv`: プロンプト・ソースごと・視点ごとの SAM マスク（実画像のマスクとの IoU）
- `summary.json`: 設定・集計・SAM 実験の ID
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from spatialsam.experiments.store import _write_json, new_experiment_id

COMPARISONS_DIRNAME = "comparisons"
REAL = "real"


class Comparison:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def id(self) -> str:
        return self.root.name

    def frames_dir(self, source: str) -> Path:
        return self.root / "frames" / source

    def frame_path(self, source: str, index: int) -> Path:
        return self.frames_dir(source) / f"{index:05d}.png"

    def video_path(self, source: str) -> Path:
        return self.root / "videos" / f"{self.id}-{source}.mp4"

    def save_table(self, name: str, table: pd.DataFrame) -> None:
        table.to_csv(self.root / f"{name}.csv", index=False)

    def load_table(self, name: str) -> pd.DataFrame:
        path = self.root / f"{name}.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def save_summary(self, summary: dict[str, Any]) -> None:
        _write_json(self.root / "summary.json", summary)

    def load_summary(self) -> dict[str, Any]:
        return json.loads((self.root / "summary.json").read_text(encoding="utf-8"))


class ComparisonStore:
    def __init__(self, outputs_dir: Path) -> None:
        self.root = outputs_dir / COMPARISONS_DIRNAME

    def create(self) -> Comparison:
        comparison = Comparison(self.root / f"compare-{new_experiment_id()}")
        comparison.root.mkdir(parents=True, exist_ok=False)
        return comparison

    def open(self, comparison_id: str) -> Comparison:
        comparison = Comparison(self.root / comparison_id)
        if not (comparison.root / "summary.json").exists():
            raise FileNotFoundError(f"比較結果が見つかりません: {comparison_id}")
        return comparison

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted((p.name for p in self.root.iterdir() if (p / "summary.json").exists()), reverse=True)
