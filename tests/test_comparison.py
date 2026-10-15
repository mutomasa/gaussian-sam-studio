import numpy as np
import pandas as pd
import pytest

from spatialsam.evaluation.image_quality import QualityScore, summarize_quality
from spatialsam.experiments.comparisons import REAL, ComparisonStore
from spatialsam.experiments.store import ExperimentStore
from spatialsam.models.fake_segmenter import FakeSegmenter
from spatialsam.pipeline.comparison import ComparisonJob, crop_even, mask_rows, run_comparison, summarize_masks
from spatialsam.spatial.camera import Camera, intrinsics
from spatialsam.spatial.renderer import RenderOutput
from spatialsam.spatial.scene_source import SceneSource
from spatialsam.training.dataset import TrainView


class FlatRenderer:
    def __init__(self, value):
        self.value = value

    def render(self, camera):
        shape = (camera.height, camera.width)
        return RenderOutput(np.full((*shape, 3), self.value, np.uint8), np.ones(shape, np.float32),
                            np.ones(shape, np.float32))


class MeanScorer:
    def score(self, rendered, target):
        diff = float(np.abs(rendered.astype(int) - target.astype(int)).mean())
        return QualityScore(psnr=100 - diff, ssim=1 - diff / 255, lpips=diff / 255)


def views(n=4, size=(33, 48)):
    camera = Camera(intrinsics(40, 40, 24, 16), np.eye(4), size[1], size[0])
    return [TrainView(f"{i:03d}.jpg", camera, np.full((*size, 3), 100, np.uint8)) for i in range(n)]


def test_crop_even_trims_odd_sizes():
    view = crop_even(views(1)[0])
    assert view.image.shape == (32, 48, 3) and (view.camera.width, view.camera.height) == (48, 32)
    assert crop_even(view) is view


def test_run_comparison_end_to_end(tmp_path):
    for name in ("a.ply", "b.ply"):
        (tmp_path / name).write_bytes(b"")  # 描画はスタブなので中身は読まない
    sources = {"close": SceneSource(tmp_path / "a.ply"), "far": SceneSource(tmp_path / "b.ply")}
    renderers = {"a.ply": FlatRenderer(110), "b.ply": FlatRenderer(200)}
    job = ComparisonJob(views(), sources, ["thing"], fps=4, info={"scene": "test"})
    comparison = run_comparison(job, ComparisonStore(tmp_path), ExperimentStore(tmp_path / "exp"), FakeSegmenter(),
                                MeanScorer(), lambda source: renderers[source.ply_path.name])
    summary = comparison.load_summary()
    assert summary["num_views"] == 4 and summary["resolution"] == [48, 32] and summary["scene"] == "test"
    assert summary["quality"]["close"]["psnr"] > summary["quality"]["far"]["psnr"]
    assert set(summary["experiments"]) == {"thing/real", "thing/close", "thing/far"}
    seg = {row["source"]: row for row in summary["segmentation"]}
    assert seg[REAL]["detection_rate"] == 1.0 and seg["far"]["mean_iou_vs_real"] == pytest.approx(1.0)
    assert len(comparison.load_table("quality")) == 8 and len(comparison.load_table("masks")) == 12
    assert comparison.frame_path("close", 3).exists() and comparison.video_path(REAL).exists()
    assert ComparisonStore(tmp_path).list_ids() == [comparison.id]


def test_real_is_reserved(tmp_path):
    with pytest.raises(ValueError):
        run_comparison(ComparisonJob(views(), {REAL: "x"}, ["p"]), ComparisonStore(tmp_path),
                       ExperimentStore(tmp_path), FakeSegmenter(), MeanScorer(), lambda s: FlatRenderer(0))


def test_mask_rows_and_summary():
    empty, full = np.zeros((2, 2), bool), np.ones((2, 2), bool)
    rows = mask_rows("p", "gs", [full, empty, empty], [full, full, empty])
    assert [r["iou_vs_real"] for r in rows] == [1.0, 0.0, None]
    assert [r["detected"] for r in rows] == [True, False, False]
    summary = summarize_masks(pd.DataFrame(rows))
    assert summary == [{"prompt": "p", "source": "gs", "detection_rate": pytest.approx(1 / 3),
                        "mean_iou_vs_real": pytest.approx(0.5)}]
    assert summarize_masks(pd.DataFrame()) == []


def test_summarize_quality():
    scores = [QualityScore(20, 0.5, 0.2), QualityScore(30, 0.7, None)]
    assert summarize_quality(scores) == {"psnr": 25, "ssim": pytest.approx(0.6), "lpips": pytest.approx(0.2)}
    assert summarize_quality([]) == {}


def test_image_quality_scorer_without_lpips():
    pytest.importorskip("torch")
    from spatialsam.evaluation.image_quality import ImageQualityScorer

    scorer = ImageQualityScorer(device="cpu", use_lpips=False)
    image = np.random.default_rng(0).integers(0, 255, (16, 16, 3)).astype(np.uint8)
    same = scorer.score(image, image)
    assert same.psnr == float("inf") and same.ssim == pytest.approx(1.0, abs=1e-5) and same.lpips is None
    with pytest.raises(ValueError):
        scorer.score(image, image[:8])
