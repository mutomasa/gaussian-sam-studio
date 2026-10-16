import numpy as np
import pytest

from spatialsam.evaluation import segmentation as seg

torch = pytest.importorskip("torch")


def random_masks(seed, n=30, shape=(24, 32)):
    rng = np.random.default_rng(seed)
    return [(rng.random(shape) > rng.random(), rng.random(shape) > rng.random()) for _ in range(n)]


def test_iou_and_dice_match_numpy_definition():
    for a, b in random_masks(0):
        assert seg.iou(a, b) == pytest.approx(seg.iou_numpy(a, b), abs=1e-6)
        assert seg.dice(a, b) == pytest.approx(seg.dice_numpy(a, b), abs=1e-6)


def test_empty_masks_are_perfect_match():
    empty = np.zeros((4, 4), bool)
    assert seg.iou(empty, empty) == 1.0 and seg.dice(empty, empty) == 1.0
    full = np.ones((4, 4), bool)
    assert seg.iou(empty, full) == 0.0 and seg.dice(empty, full) == 0.0


def test_numpy_fallback_without_torchmetrics(monkeypatch):
    monkeypatch.setattr(seg, "_classification_metrics", lambda: None)
    a, b = random_masks(1, n=1)[0]
    assert seg.iou(a, b) == seg.iou_numpy(a, b) and seg.dice(a, b) == seg.dice_numpy(a, b)


def test_image_quality_psnr_ssim():
    from spatialsam.evaluation.image_quality import psnr, ssim
    from spatialsam.training.losses import psnr as train_psnr

    rng = np.random.default_rng(2)
    a = torch.as_tensor(rng.random((32, 40, 3)), dtype=torch.float32)
    b = (a + 0.05 * torch.as_tensor(rng.standard_normal((32, 40, 3)), dtype=torch.float32)).clamp(0, 1)
    assert psnr(a, b) == pytest.approx(train_psnr(a, b), abs=1e-4)
    assert psnr(a, a) == float("inf") and ssim(a, a) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < ssim(a, b) < 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA が無い")
def test_lpips_alexnet_on_gpu():
    from spatialsam.evaluation.image_quality import ImageQualityScorer

    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
    scorer = ImageQualityScorer()
    assert scorer.score(image, image).lpips == pytest.approx(0.0, abs=1e-6)
    assert scorer.score(image, 255 - image).lpips > 0.1
