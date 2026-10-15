import json

import numpy as np
import pytest

from spatialsam.experiments.store import ExperimentStore
from spatialsam.models.fake_segmenter import FakeSegmenter, center_ellipse_mask
from spatialsam.pipeline.evaluation import evaluate_experiment
from spatialsam.pipeline.segmentation import SegmentationJob, run_segmentation
from spatialsam.video.io import FrameSampling
from tests.conftest import FRAME_SHAPE


@pytest.fixture
def outcome(synthetic_video, tmp_path):
    store = ExperimentStore(tmp_path / "outputs")
    job = SegmentationJob(synthetic_video, "square", FrameSampling(stride=2), {"repo_id": "x"})
    return run_segmentation(job, FakeSegmenter(), store)


def test_run_segmentation_saves_experiment(outcome):
    experiment = outcome.experiment
    meta = experiment.load_metadata()
    assert meta["prompt"] == "square"
    assert meta["model"] == {"segmenter": "fake", "repo_id": "x"}
    assert meta["frame_indices"] == [0, 2, 4, 6, 8, 10]
    assert meta["input"]["fps"] == pytest.approx(6)
    assert experiment.overlay_path.stat().st_size > 0
    assert json.loads((experiment.root / "prediction" / "tracks.json").read_text())["1"]["num_frames"] == 6


def test_prediction_roundtrip(outcome):
    loaded = outcome.experiment.load_prediction()
    assert [f.frame_index for f in loaded.frames] == [0, 2, 4, 6, 8, 10]
    np.testing.assert_array_equal(loaded.frames[0].masks[1], outcome.result.frames[0].masks[1])


def test_evaluate_experiment_with_matching_gt(outcome):
    gt = {i: center_ellipse_mask(FRAME_SHAPE) for i in (0, 2, 4)}
    report = evaluate_experiment(outcome.experiment, gt)
    assert report.summary["J&F"] == pytest.approx(1.0)
    assert report.missing_gt_frames == [6, 8, 10]
    assert outcome.experiment.has_evaluation()
    assert (outcome.experiment.evaluation_dir / "per_frame.csv").exists()


def test_evaluate_without_overlapping_frames_fails(outcome):
    with pytest.raises(ValueError):
        evaluate_experiment(outcome.experiment, {99: np.zeros(FRAME_SHAPE, bool)})


def test_store_lists_and_opens(outcome, tmp_path):
    store = ExperimentStore(tmp_path / "outputs")
    assert store.list_ids() == [outcome.experiment.id]
    with pytest.raises(FileNotFoundError):
        store.open("missing")
