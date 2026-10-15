"""アップロードファイルの検査と一時保存。"""

from __future__ import annotations

from pathlib import Path

ALLOWED_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")


def validate_video_upload(filename: str, size_bytes: int, max_mib: int) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError(f"未対応の形式です: {suffix}（対応: {', '.join(ALLOWED_VIDEO_SUFFIXES)}）")
    if size_bytes > max_mib * 1024 * 1024:
        raise ValueError(f"ファイルサイズが上限 {max_mib} MiB を超えています")


def save_upload(data: bytes, filename: str, out_dir: Path) -> Path:
    """ファイル名はパス成分を除いて保存する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / Path(filename).name
    path.write_bytes(data)
    return path
