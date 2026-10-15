"""DAVIS 2017 から GT 付きサンプル動画を `samples/` に作る。

    uv run python scripts/prepare_samples.py [--cache-dir ~/.cache/spatialsam/davis]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialsam.config import PROJECT_ROOT
from spatialsam.samples.davis import download, load_manifest, prepare_sample, verify_sha256

MANIFEST = PROJECT_ROOT / "configs" / "samples.yaml"
SAMPLES_DIR = PROJECT_ROOT / "samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "spatialsam" / "davis")
    parser.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(MANIFEST)
    print(f"ライセンス: {manifest.license}")
    archive = download(manifest.url, args.cache_dir / manifest.archive_name)
    verify_sha256(archive, manifest.sha256)
    for spec in manifest.samples:
        sample = prepare_sample(archive, spec, args.samples_dir, manifest.fps)
        print(f"{spec.sequence}: {sample.num_frames} フレーム → {sample.video.name}（プロンプト例: {spec.prompt}）")


if __name__ == "__main__":
    main()
