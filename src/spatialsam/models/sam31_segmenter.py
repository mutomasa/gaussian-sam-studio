"""SAM 3.1 Object Multiplex（facebook/sam3.1）をローカルGPUで動かすアダプタ。

`sam3` パッケージは `uv sync --extra gpu` で導入する。import は推論時まで遅延させる。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from spatialsam.models.sam3_compat import patch_init_state_kwargs
from spatialsam.models.types import FrameMasks, SegmentationRequest, SegmentationResult


@dataclass(frozen=True)
class Sam31Settings:
    checkpoint_path: str | None = None
    max_num_objects: int = 16
    use_fa3: bool = False  # FlashAttention 3 は Hopper 専用。Ampere 等では False
    score_threshold: float = 0.5
    # 何フレームまとめて画像エンコーダ・後処理に通すか。公式ビルダーは 16 固定（H100 前提）
    grounding_batch_size: int = 1
    postprocess_batch_size: int = 1
    # 画像・テキストエンコーダ（全パラメータの 94%）だけ bf16 で保持する。他は fp32 のまま
    bf16_backbone: bool = True
    # デコード済みフレームを GPU ではなく CPU メモリに置く
    offload_video_to_cpu: bool = True


def apply_memory_settings(predictor: Any, settings: Sam31Settings) -> Any:
    """ビルダー引数で変えられない省メモリ設定を、構築後に反映する（8GB VRAM 向け）。"""
    predictor.model.batched_grounding_batch_size = settings.grounding_batch_size
    predictor.model.postprocess_batch_size = settings.postprocess_batch_size
    if settings.bf16_backbone:
        import torch

        predictor.model.detector.backbone.to(torch.bfloat16)
        torch.cuda.empty_cache()
    return predictor


def build_sam31_predictor(settings: Sam31Settings) -> Any:
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=settings.checkpoint_path,
        max_num_objects=settings.max_num_objects,
        use_fa3=settings.use_fa3,
        default_output_prob_thresh=settings.score_threshold,
    )
    return apply_memory_settings(patch_init_state_kwargs(predictor), settings)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def convert_frame_output(frame_index: int, output: dict[str, Any]) -> FrameMasks:
    """SAM 3 の `outputs`（out_obj_ids / out_binary_masks / out_probs）を正規化する。"""
    obj_ids = _to_numpy(output["out_obj_ids"]).tolist()
    masks = _to_numpy(output["out_binary_masks"]).astype(bool)
    probs = _to_numpy(output.get("out_probs", np.ones(len(obj_ids))))
    kept = [i for i, _ in enumerate(obj_ids) if masks[i].any()]
    return FrameMasks(
        frame_index,
        {int(obj_ids[i]): masks[i] for i in kept},
        {int(obj_ids[i]): _score(probs[i]) for i in kept},
    )


def _score(prob: float) -> float:
    """検出スコアの無い出力（TRACKER_ONLY_SCORE）は NaN にする。"""
    return float("nan") if prob <= TRACKER_ONLY_SCORE else float(prob)


TRACKER_ONLY_SCORE = -10000.0  # 逆向き追跡など、検出スコアが無い出力に sam3 が入れる値


def first_detected_frame(frames: dict[int, FrameMasks]) -> int | None:
    detected = [index for index, frame in frames.items() if frame.masks]
    return min(detected) if detected else None


def backfill(frames: dict[int, FrameMasks], earlier: dict[int, FrameMasks]) -> dict[int, FrameMasks]:
    """最初の検出より前のフレームだけを逆向き追跡の結果で置き換える（それ以降は順方向の結果を残す）。"""
    first = first_detected_frame(frames)
    if first is None:
        return frames
    filled = {i: f for i, f in earlier.items() if i < first and f.masks}
    return frames | filled


class Sam31Segmenter:
    name = "sam3.1"

    def __init__(
        self,
        settings: Sam31Settings,
        predictor_factory: Callable[[Sam31Settings], Any] = build_sam31_predictor,
    ) -> None:
        self._settings = settings
        self._factory = predictor_factory
        self._predictor: Any | None = None

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        predictor = self._get_predictor()
        session_id = self._start_session(predictor, str(request.frames_dir))
        try:
            self._add_text_prompt(predictor, session_id, request.prompt)
            frames = self._convert(self._propagate(predictor, session_id, "forward"))
            frames = backfill(frames, self._backfill_frames(predictor, session_id, frames))
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        return SegmentationResult(sorted(frames.values(), key=lambda f: f.frame_index))

    def release(self) -> None:
        if self._predictor is not None and hasattr(self._predictor, "shutdown"):
            self._predictor.shutdown()
        self._predictor = None

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            self._predictor = self._factory(self._settings)
        return self._predictor

    def _start_session(self, predictor: Any, resource_path: str) -> str:
        response = predictor.handle_request({
            "type": "start_session",
            "resource_path": resource_path,
            "offload_video_to_cpu": self._settings.offload_video_to_cpu,
        })
        return response["session_id"]

    @staticmethod
    def _add_text_prompt(predictor: Any, session_id: str, prompt: str) -> None:
        predictor.handle_request(
            {"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt}
        )

    @staticmethod
    def _propagate(predictor: Any, session_id: str, direction: str,
                   start: int | None = None) -> Iterable[dict[str, Any]]:
        request = {"type": "propagate_in_video", "session_id": session_id, "propagation_direction": direction}
        if start is not None:
            request["start_frame_index"] = start
        return list(predictor.handle_stream_request(request))

    @staticmethod
    def _convert(responses: Iterable[dict[str, Any]]) -> dict[int, FrameMasks]:
        return {r["frame_index"]: convert_frame_output(r["frame_index"], r["outputs"]) for r in responses}

    def _backfill_frames(self, predictor: Any, session_id: str, frames: dict[int, FrameMasks]) -> dict[int, FrameMasks]:
        """最初に物体が出たフレームより前を、そこから逆向きに追跡して埋める #30。

        テキストプロンプトを付けたフレーム 0 で検出できず、数フレーム後から追跡が始まることがある
        （bonsai の描画動画など）。逆向きの追跡は同じセッションの状態をそのまま使える。
        """
        first = first_detected_frame(frames)
        if first is None or first == 0:
            return {}
        return self._convert(self._propagate(predictor, session_id, "backward", start=first))
