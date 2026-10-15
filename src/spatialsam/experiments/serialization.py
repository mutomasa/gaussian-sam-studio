"""FrameMasks と .npz / tracks.json の相互変換。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from spatialsam.models.types import FrameMasks, SegmentationResult


def save_frame_masks(frame: FrameMasks, path: Path) -> None:
    ids = np.array(sorted(frame.masks), dtype=np.int64)
    shape = next(iter(frame.masks.values())).shape if frame.masks else (0, 0)
    masks = np.stack([frame.masks[i] for i in ids]) if len(ids) else np.zeros((0, *shape), bool)
    scores = np.array([frame.scores.get(int(i), np.nan) for i in ids], dtype=np.float32)
    np.savez_compressed(path, frame_index=frame.frame_index, obj_ids=ids, masks=masks, scores=scores)


def load_frame_masks(path: Path) -> FrameMasks:
    with np.load(path) as data:
        ids = data["obj_ids"].tolist()
        return FrameMasks(
            int(data["frame_index"]),
            {obj_id: data["masks"][k].astype(bool) for k, obj_id in enumerate(ids)},
            {obj_id: float(data["scores"][k]) for k, obj_id in enumerate(ids)},
        )


def build_tracks(result: SegmentationResult) -> dict[str, Any]:
    """物体IDごとに出現フレームと平均スコアをまとめる。"""
    tracks: dict[int, dict[str, Any]] = {}
    for frame in result.frames:
        for obj_id, score in frame.scores.items():
            entry = tracks.setdefault(obj_id, {"frames": [], "scores": []})
            entry["frames"].append(frame.frame_index)
            entry["scores"].append(score)
    return {
        str(obj_id): {
            "frames": entry["frames"],
            "num_frames": len(entry["frames"]),
            "mean_score": float(np.nanmean(entry["scores"])),
        }
        for obj_id, entry in sorted(tracks.items())
    }
