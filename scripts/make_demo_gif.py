"""デモ動画（scripts/record_demo.py の出力）から、README に載せる見どころの GIF を作る #49。

    uv run python scripts/make_demo_gif.py [--video outputs/demo/bonsai_demo.mp4] [--out docs/media/bonsai_demo.gif]

区間（秒）は録画のたびに少しずれるので、変わったら --segment で指定し直す（例: --segment 36-42 --segment 64-71）。
GIF は ffmpeg の palettegen / paletteuse で減色して小さくする。
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from spatialsam.config import PROJECT_ROOT

# 既定: 描画結果 → SAM の重畳 → 3D セグメンテーション（視点を動かす）→ 実画像 vs 3D 描画
DEFAULT_SEGMENTS = ["36-42", "64-71", "78-92", "114-124"]


def parse_segment(text: str) -> tuple[float, float]:
    start, _, end = text.partition("-")
    start_sec, end_sec = float(start), float(end)
    if end_sec <= start_sec:
        raise argparse.ArgumentTypeError(f"区間は 開始-終了（開始 < 終了）で指定してください: {text}")
    return start_sec, end_sec


def filter_graph(segments: list[tuple[float, float]], width: int, fps: int, colors: int) -> str:
    """区間を切り出してつなぎ、縮小・間引きしてから、その動画専用のパレットで減色する。"""
    parts = [f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS[s{i}]" for i, (a, b) in enumerate(segments)]
    joined = "".join(f"[s{i}]" for i in range(len(segments)))
    return (";".join(parts) + f";{joined}concat=n={len(segments)}:v=1:a=0,fps={fps},"
            f"scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=PROJECT_ROOT / "outputs" / "demo" / "bonsai_demo.mp4")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "media" / "bonsai_demo.gif")
    parser.add_argument("--segment", type=parse_segment, action="append", help="切り出す区間（秒）。複数指定できる")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--colors", type=int, default=128)
    args = parser.parse_args()
    import imageio_ffmpeg

    segments = args.segment or [parse_segment(s) for s in DEFAULT_SEGMENTS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(args.video),
                    "-filter_complex", filter_graph(segments, args.width, args.fps, args.colors), str(args.out)],
                   check=True)
    total = sum(b - a for a, b in segments)
    print(f"→ {args.out}（{total:.0f} 秒、{args.out.stat().st_size / 2**20:.1f} MB）")


if __name__ == "__main__":
    main()
