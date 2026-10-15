import pytest

from spatialsam.experiments.store import ExperimentStore
from spatialsam.models.fake_segmenter import FakeSegmenter
from spatialsam.pipeline.retry import OomRetryPolicy
from spatialsam.pipeline.segmentation import SegmentationJob, run_segmentation
from spatialsam.video.io import FrameSampling


class OutOfMemoryError(RuntimeError):
    """torch.cuda.OutOfMemoryError と同名の代役。"""


class FlakySegmenter(FakeSegmenter):
    """フレーム数が `limit` を超えると OOM を出す。"""

    name = "flaky"

    def __init__(self, limit: int, error: Exception | None = None):
        self.limit = limit
        self.error = error or OutOfMemoryError("CUDA out of memory")
        self.calls: list[int] = []

    def segment(self, request):
        num_frames = len(list(request.frames_dir.glob("*.jpg")))
        self.calls.append(num_frames)
        if num_frames > self.limit:
            raise self.error
        return super().segment(request)


def test_degrade_halves_frames_with_floor():
    policy = OomRetryPolicy(min_frames=4)
    assert policy.degrade(FrameSampling(stride=2), 12).max_frames == 6
    assert policy.degrade(FrameSampling(), 6).max_frames == 4
    assert policy.degrade(FrameSampling(stride=3), 12).stride == 3
    with pytest.raises(ValueError):
        policy.degrade(FrameSampling(), 4)


def job(video):
    return SegmentationJob(video, "square", FrameSampling())


def test_retries_with_fewer_frames_and_records_attempts(synthetic_video, tmp_path):
    segmenter = FlakySegmenter(limit=6)
    outcome = run_segmentation(job(synthetic_video), segmenter, ExperimentStore(tmp_path), retry_policy=OomRetryPolicy(min_frames=2))
    assert segmenter.calls == [12, 6]
    meta = outcome.experiment.load_metadata()
    assert meta["sampling"]["max_frames"] == 6
    assert meta["requested_sampling"]["max_frames"] is None
    assert meta["oom_retries"] == [{"sampling": meta["requested_sampling"], "num_frames": 12, "error": "out_of_memory"}]
    assert len(meta["frame_indices"]) == 6
    assert len(list(outcome.experiment.frames_dir.glob("*.jpg"))) == 6


def test_gives_up_after_max_retries(synthetic_video, tmp_path):
    segmenter = FlakySegmenter(limit=1)
    with pytest.raises(OutOfMemoryError):
        run_segmentation(job(synthetic_video), segmenter, ExperimentStore(tmp_path), retry_policy=OomRetryPolicy(max_retries=1, min_frames=2))
    assert segmenter.calls == [12, 6]


def test_non_oom_errors_are_not_retried(synthetic_video, tmp_path):
    segmenter = FlakySegmenter(limit=1, error=ValueError("bad frame"))
    with pytest.raises(ValueError):
        run_segmentation(job(synthetic_video), segmenter, ExperimentStore(tmp_path))
    assert segmenter.calls == [12]
