import numpy as np
import pytest

from spatialsam.spatial.camera import Camera, intrinsics, look_at
from spatialsam.spatial.mask_lifting import LiftSettings, MaskedView, backproject, lift_masks, project

UP = np.array([0.0, 0.0, 1.0])
SIZE = (60, 80)  # (H, W)


def camera_at(eye):
    return Camera(intrinsics(70, 70, SIZE[1] / 2, SIZE[0] / 2), look_at(np.asarray(eye, float), np.zeros(3), UP),
                  SIZE[1], SIZE[0])


def test_project_and_backproject_roundtrip():
    camera = camera_at([0, -5, 1])
    depth = np.full(SIZE, 4.0)
    mask = np.zeros(SIZE, bool)
    mask[30, 40] = mask[10, 5] = True
    points = backproject(mask, depth, camera)
    uv, z = project(points, camera)
    assert np.allclose(z, 4.0)
    assert sorted(map(tuple, np.floor(uv).astype(int)[:, ::-1])) == [(10, 5), (30, 40)]
    assert len(backproject(mask, depth, camera, valid=np.zeros(SIZE, bool))) == 0


def splat_view(camera, points, labels, radius=2):
    """点を小さな円として描いた「描画結果」と、ラベル 1 の点を覆うマスク（z バッファ付き）。"""
    depth = np.full(SIZE, np.inf)
    owner = np.full(SIZE, -1)
    uv, z = project(points, camera)
    for i in np.argsort(-z):  # 奥から描いて手前で上書き
        c, r = np.floor(uv[i]).astype(int)
        rr, cc = np.meshgrid(np.arange(r - radius, r + radius + 1), np.arange(c - radius, c + radius + 1), indexing="ij")
        ok = (rr >= 0) & (rr < SIZE[0]) & (cc >= 0) & (cc < SIZE[1])
        rr, cc = rr[ok], cc[ok]
        closer = z[i] < depth[rr, cc]
        depth[rr[closer], cc[closer]] = z[i]
        owner[rr[closer], cc[closer]] = i
    alpha = np.isfinite(depth).astype(np.float32)
    mask = np.isin(owner, np.nonzero(labels)[0])
    return MaskedView(camera, {7: mask}, np.where(np.isfinite(depth), depth, 0.0), alpha)


def test_lift_labels_target_and_ignores_occluded_points():
    rng = np.random.default_rng(0)
    target = rng.normal(scale=0.15, size=(30, 3))
    other = rng.normal(scale=0.15, size=(30, 3)) + [1.2, 0, 0]
    hidden = np.array([[0.0, 3.0, 0.0]])  # 正面からは target の真後ろ。背面のカメラからは見えるが対象外
    points = np.vstack([target, other, hidden])
    labels = np.r_[np.ones(30, bool), np.zeros(31, bool)]
    eyes = [[4 * np.cos(a), 4 * np.sin(a), 1.0] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    result = lift_masks(points, [splat_view(camera_at(e), points, labels) for e in eyes])
    assert result.num_views == 8 and result.object_ids == [7]
    predicted = result.labels(7)
    assert predicted[30:].sum() == 0
    assert predicted[:30].mean() > 0.8
    assert np.array_equal(result.any_labels(), predicted)


def test_min_views_excludes_rarely_seen_points():
    camera = camera_at([0, -5, 0])
    points = np.zeros((1, 3))
    view = splat_view(camera, points, np.array([True]))
    assert lift_masks(points, [view], LiftSettings(min_views=2)).labels(7).sum() == 0
    once = lift_masks(points, [view], LiftSettings(min_views=1))
    assert once.labels(7).all() and once.votes.confidence(7)[0] == pytest.approx(1.0)


def test_points_behind_camera_are_not_counted():
    camera = camera_at([0, -5, 0])
    view = MaskedView(camera, {1: np.ones(SIZE, bool)}, np.full(SIZE, 5.0), np.ones(SIZE, np.float32))
    result = lift_masks(np.array([[0.0, -10.0, 0.0]]), [view], LiftSettings(min_views=1))
    assert result.votes.visible[0] == 0 and not result.labels(1).any()
