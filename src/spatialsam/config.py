"""`configs/model.yaml` の読み込み。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from spatialsam.models.chunking import ChunkSettings
from spatialsam.models.sam31_segmenter import Sam31Settings
from spatialsam.tracking import TrackingSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


@dataclass(frozen=True)
class ModelSource:
    """再現性のため実験メタデータに記録するモデル配布元。"""

    repo_id: str = ""
    revision: str = ""


@dataclass(frozen=True)
class AppConfig:
    default_segmenter: str = "fake"
    sam31: Sam31Settings = field(default_factory=Sam31Settings)
    sam31_source: ModelSource = field(default_factory=ModelSource)
    chunking: ChunkSettings | None = None
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    max_upload_mib: int = 500


def _resolve(path: str | None, base: Path) -> Path:
    candidate = Path(path) if path else base / "outputs"
    return candidate if candidate.is_absolute() else base / candidate


def parse_config(raw: dict[str, Any], base: Path = PROJECT_ROOT) -> AppConfig:
    sam = dict(raw.get("sam31") or {})
    source = sam.pop("source", None) or {}
    return AppConfig(
        default_segmenter=raw.get("default_segmenter", "fake"),
        sam31=Sam31Settings(**sam),
        sam31_source=ModelSource(**source),
        chunking=ChunkSettings(**raw["chunking"]) if raw.get("chunking") else None,
        tracking=TrackingSettings(**(raw.get("tracking") or {})),
        outputs_dir=_resolve(raw.get("outputs_dir"), base),
        max_upload_mib=int(raw.get("max_upload_mib", 500)),
    )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not path.exists():
        return AppConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return parse_config(dict(raw))
