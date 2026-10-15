"""GPU メモリ不足時に入力を縮小して再実行する方針。"""

from __future__ import annotations

from dataclasses import dataclass, replace

from spatialsam.video.io import FrameSampling


@dataclass(frozen=True)
class OomRetryPolicy:
    max_retries: int = 2
    min_frames: int = 8

    def degrade(self, sampling: FrameSampling, num_frames: int) -> FrameSampling:
        """フレーム数を半分にする（SAM 3.1 は内部で 1008px に変換するため解像度は効かない）。"""
        if not self.can_degrade(num_frames):
            raise ValueError(f"これ以上フレーム数を減らせません（{num_frames} フレーム）")
        return replace(sampling, max_frames=max(self.min_frames, num_frames // 2))

    def can_degrade(self, num_frames: int) -> bool:
        return num_frames > self.min_frames
