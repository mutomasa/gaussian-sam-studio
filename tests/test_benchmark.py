from spatialsam.experiments.benchmark import benchmark_row
from spatialsam.experiments.store import ExperimentStore
from spatialsam.models.gpu import Measurement
from spatialsam.models.types import SegmentationResult
from spatialsam.pipeline.evaluation import EvaluationReport
from spatialsam.pipeline.segmentation import SegmentationOutcome
from spatialsam.samples.davis import SampleSpec


def test_benchmark_row(tmp_path):
    experiment = ExperimentStore(tmp_path).create("exp")
    experiment.save_metadata({"frame_indices": [0, 1, 2], "object_ids": [1, 4], "oom_retries": []})
    outcome = SegmentationOutcome(experiment, SegmentationResult([]), Measurement(1.234, 512.4))
    report = EvaluationReport({"J": 0.91234, "F": 0.8, "J&F": 0.85617, "Dice": 0.9, "num_frames": 3}, [], [])
    row = benchmark_row(SampleSpec("cows", "cow"), outcome, report)
    assert row == {"sequence": "cows", "prompt": "cow", "experiment": "exp", "frames": 3, "objects": 2,
                   "oom_retries": 0, "elapsed_sec": 1.23, "peak_vram_mib": 512,
                   "J": 0.9123, "F": 0.8, "J&F": 0.8562, "Dice": 0.9}
