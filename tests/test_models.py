from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spatialsam.config import AppConfig, parse_config
from spatialsam.models.factory import create_segmenter
from spatialsam.models.fake_segmenter import FakeSegmenter, center_ellipse_mask
from spatialsam.models.gpu import measure, query_gpu_status
from spatialsam.models.registry import ModelRegistry
from spatialsam.models.sam31_segmenter import Sam31Segmenter, Sam31Settings, convert_frame_output
from spatialsam.models.types import FrameMasks, SegmentationRequest, SegmentationResult


def test_frame_masks_union():
    a = np.zeros((3, 3), bool)
    b = np.zeros((3, 3), bool)
    a[0, 0] = b[2, 2] = True
    assert FrameMasks(0, {1: a, 2: b}).union((3, 3)).sum() == 2


def test_result_remap_and_object_ids():
    mask = np.ones((2, 2), bool)
    result = SegmentationResult([FrameMasks(0, {2: mask}), FrameMasks(1, {5: mask})])
    remapped = result.remap_indices([10, 20])
    assert [f.frame_index for f in remapped.frames] == [10, 20]
    assert remapped.object_ids() == [2, 5]


def test_convert_frame_output_drops_empty_masks():
    masks = np.zeros((2, 4, 4), bool)
    masks[0, 1, 1] = True
    output = {"out_obj_ids": np.array([7, 8]), "out_binary_masks": masks, "out_probs": np.array([0.9, 0.8])}
    frame = convert_frame_output(3, output)
    assert frame.frame_index == 3
    assert list(frame.masks) == [7]
    assert frame.scores == {7: pytest.approx(0.9)}


class FakePredictor:
    def __init__(self):
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request["type"])
        return {"session_id": "s1", "outputs": {}}

    def handle_stream_request(self, request):
        self.requests.append(request["type"])
        for i in (1, 0):
            yield {"frame_index": i, "outputs": {
                "out_obj_ids": np.array([1]),
                "out_binary_masks": np.ones((1, 2, 2), bool),
                "out_probs": np.array([0.7]),
            }}


def test_sam31_segmenter_request_sequence(tmp_path):
    predictor = FakePredictor()
    segmenter = Sam31Segmenter(Sam31Settings(), predictor_factory=lambda _: predictor)
    result = segmenter.segment(SegmentationRequest(tmp_path, "cube"))
    assert predictor.requests == ["start_session", "add_prompt", "propagate_in_video", "close_session"]
    assert [f.frame_index for f in result.frames] == [0, 1]


def test_fake_segmenter_returns_one_object_per_frame(synthetic_video, tmp_path):
    from spatialsam.video.io import FrameSampling, extract_frames

    extract_frames(synthetic_video, FrameSampling(max_frames=3), tmp_path)
    result = FakeSegmenter().segment(SegmentationRequest(tmp_path, "anything"))
    assert len(result.frames) == 3
    assert result.object_ids() == [1]


def test_center_ellipse_mask_is_centered():
    mask = center_ellipse_mask((40, 40))
    assert mask[20, 20] and not mask[0, 0]


def test_factory_rejects_unknown_name():
    assert create_segmenter("fake", AppConfig()).name == "fake"
    assert create_segmenter("sam3.1", AppConfig()).name == "sam3.1"
    with pytest.raises(ValueError):
        create_segmenter("nope", AppConfig())


class Releasable:
    name = "r"

    def __init__(self):
        self.released = False

    def segment(self, request):
        raise NotImplementedError

    def release(self):
        self.released = True


def test_registry_keeps_single_model():
    cleared = []
    registry = ModelRegistry(clear_cache=lambda: cleared.append(True))
    first = registry.get("a", Releasable)
    assert registry.get("a", Releasable) is first
    second = registry.get("b", Releasable)
    assert first.released and second is not first
    assert registry.loaded_key == "b" and cleared


def test_parse_config_resolves_paths():
    raw = {
        "default_segmenter": "sam3.1",
        "sam31": {"source": {"repo_id": "facebook/sam3.1", "revision": "abc"}, "use_fa3": False},
        "outputs_dir": "out",
    }
    config = parse_config(raw, base=Path("/base"))
    assert config.sam31_source.repo_id == "facebook/sam3.1"
    assert config.sam31.use_fa3 is False
    assert config.outputs_dir == Path("/base/out")
    assert "source" in raw["sam31"]  # 入力を破壊しない


class NoCudaTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False


def test_gpu_status_and_measure_without_cuda():
    assert query_gpu_status(NoCudaTorch).available is False
    with measure(NoCudaTorch) as measurement:
        pass
    assert measurement.peak_vram_mib is None and measurement.elapsed_sec >= 0


def test_apply_memory_settings_overrides_batch_sizes():
    from types import SimpleNamespace

    from spatialsam.models.sam31_segmenter import apply_memory_settings

    predictor = SimpleNamespace(model=SimpleNamespace(batched_grounding_batch_size=16, postprocess_batch_size=16))
    settings = Sam31Settings(grounding_batch_size=2, postprocess_batch_size=4, bf16_backbone=False)
    apply_memory_settings(predictor, settings)
    assert predictor.model.batched_grounding_batch_size == 2
    assert predictor.model.postprocess_batch_size == 4


def test_apply_memory_settings_casts_only_backbone_to_bf16():
    import torch

    from spatialsam.models.sam31_segmenter import apply_memory_settings

    backbone, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    model = SimpleNamespace(detector=SimpleNamespace(backbone=backbone), head=head)
    apply_memory_settings(SimpleNamespace(model=model), Sam31Settings())
    assert backbone.weight.dtype == torch.bfloat16
    assert head.weight.dtype == torch.float32


def test_start_session_passes_offload_flag(tmp_path):
    predictor = FakePredictor()
    sent = []
    original = predictor.handle_request
    predictor.handle_request = lambda request: (sent.append(request), original(request))[1]
    Sam31Segmenter(Sam31Settings(offload_video_to_cpu=True), lambda _: predictor).segment(
        SegmentationRequest(tmp_path, "cube")
    )
    assert sent[0]["offload_video_to_cpu"] is True


def test_is_out_of_memory():
    from spatialsam.models.gpu import is_out_of_memory

    class OutOfMemoryError(RuntimeError):
        pass

    assert is_out_of_memory(OutOfMemoryError("x"))
    assert is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 82.00 MiB"))
    assert not is_out_of_memory(ValueError("other"))
