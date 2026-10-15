"""DAVIS 2017（TrainVal 480p）から GT 付きサンプル動画を作る。

zip は展開せず、指定シーケンスの JPEG と GT PNG だけを読み出す。
- 動画: `samples/<sequence>.mp4`（全フレーム、フレーム番号 = DAVIS の画像番号）
- GT:   `samples/gt/<sequence>/00000.png ...`（DAVIS のパレット PNG。画素値 = 物体ID、0 = 背景）
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import yaml

from spatialsam.video.io import write_video

JPEG_DIR = "DAVIS/JPEGImages/480p"
ANNOTATION_DIR = "DAVIS/Annotations/480p"
GT_DIRNAME = "gt"


@dataclass(frozen=True)
class SampleSpec:
    sequence: str
    prompt: str
    note: str = ""


@dataclass(frozen=True)
class SampleManifest:
    url: str
    archive_name: str
    fps: float
    license: str
    samples: list[SampleSpec]
    sha256: str = ""


@dataclass(frozen=True)
class PreparedSample:
    video: Path
    gt_dir: Path
    num_frames: int


def load_manifest(path: Path) -> SampleManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SampleManifest(
        url=raw["url"],
        archive_name=PurePosixPath(raw["url"]).name,
        fps=float(raw.get("fps", 24)),
        license=raw.get("license", ""),
        samples=[SampleSpec(**s) for s in raw["samples"]],
        sha256=raw.get("sha256", ""),
    )


def download(url: str, dest: Path) -> Path:
    """既にあれば何もしない。途中で失敗しても壊れたファイルを残さない。"""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, partial)
    return partial.rename(dest)


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    if expected and sha256sum(path) != expected:
        raise ValueError(f"sha256 が一致しません（破損または別ファイル）: {path}")


def sequence_members(names: list[str], root: str, sequence: str, suffix: str) -> list[str]:
    """`root/sequence/` 直下の `suffix` ファイルを番号順に返す（他ディレクトリやパス操作は除外）。"""
    prefix = f"{root}/{sequence}/"
    members = [n for n in names if n.startswith(prefix) and n.endswith(suffix) and "/" not in n[len(prefix):]]
    return sorted(members, key=lambda n: int(PurePosixPath(n).stem))


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("画像をデコードできません")
    return image


def iter_frames(archive: zipfile.ZipFile, members: list[str]) -> Iterator[np.ndarray]:
    for name in members:
        yield decode_image(archive.read(name))


def write_ground_truth(archive: zipfile.ZipFile, members: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in members:
        (out_dir / PurePosixPath(name).name).write_bytes(archive.read(name))


def prepare_sample(archive_path: Path, spec: SampleSpec, samples_dir: Path, fps: float) -> PreparedSample:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        jpegs = sequence_members(names, JPEG_DIR, spec.sequence, ".jpg")
        pngs = sequence_members(names, ANNOTATION_DIR, spec.sequence, ".png")
        if not jpegs:
            raise ValueError(f"シーケンスが見つかりません: {spec.sequence}")
        video = write_video(iter_frames(archive, jpegs), samples_dir / f"{spec.sequence}.mp4", fps)
        gt_dir = samples_dir / GT_DIRNAME / spec.sequence
        write_ground_truth(archive, pngs, gt_dir)
    return PreparedSample(video, gt_dir, len(jpegs))


def default_gt_dir(samples_dir: Path, video_name: str) -> Path | None:
    """サンプル動画名（例: blackswan.mp4）に対応する GT ディレクトリ。無ければ None。"""
    candidate = samples_dir / GT_DIRNAME / Path(video_name).stem
    return candidate if candidate.is_dir() else None
