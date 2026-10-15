import numpy as np

from spatialsam.models.types import FrameMasks
from spatialsam.video.overlay import PALETTE, blend_mask, color_for_id, overlay_masks, render_mask_only


def test_color_for_id_cycles_palette():
    assert color_for_id(0) == PALETTE[0]
    assert color_for_id(len(PALETTE)) == PALETTE[0]


def test_blend_mask_only_changes_masked_pixels():
    frame = np.zeros((4, 4, 3), np.uint8)
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    out = blend_mask(frame, mask, (200, 100, 0), alpha=0.5)
    assert out[0, 0].tolist() == [100, 50, 0]
    assert not out[1:, 1:].any()
    assert not frame.any()  # 入力は変更しない


def test_overlay_without_masks_returns_frame():
    frame = np.ones((4, 4, 3), np.uint8)
    assert overlay_masks(frame, None) is frame


def test_render_mask_only_colors_by_id():
    mask = np.zeros((4, 4), bool)
    mask[2, 2] = True
    canvas = render_mask_only((4, 4), FrameMasks(0, {3: mask}))
    assert tuple(canvas[2, 2]) == color_for_id(3)
    assert canvas.sum() == sum(color_for_id(3))
