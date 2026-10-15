import numpy as np
import pytest

from spatialsam.spatial.camera import Camera, intrinsics, intrinsics_from_fov
from spatialsam.spatial.gaussians import GaussianScene, load_ply, save_ply, sh_degree_from_coeffs
from spatialsam.spatial.renderer import GaussianRenderer, composite, expected_depth, to_rgb_uint8

SH_C0 = 0.28209479177387814


def make_scene(n=4, degree=1, seed=0):
    rng = np.random.default_rng(seed)
    k = (degree + 1) ** 2
    return GaussianScene(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        quats=rng.normal(size=(n, 4)).astype(np.float32),
        log_scales=rng.normal(size=(n, 3)).astype(np.float32),
        opacity_logits=rng.normal(size=n).astype(np.float32),
        sh=rng.normal(size=(n, k, 3)).astype(np.float32),
    )


def test_ply_roundtrip(tmp_path):
    scene = make_scene(degree=2)
    loaded = load_ply(save_ply(scene, tmp_path / "s.ply"))
    for name in ("means", "quats", "log_scales", "opacity_logits", "sh"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(scene, name))
    assert loaded.sh_degree == 2


def test_ply_rest_is_channel_major(tmp_path):
    scene = make_scene(n=1, degree=1)
    sh = np.zeros((1, 4, 3), np.float32)
    sh[0, 1:, 1] = [1, 2, 3]  # G チャンネルの高次係数
    scene = GaussianScene(scene.means, scene.quats, scene.log_scales, scene.opacity_logits, sh)
    from plyfile import PlyData
    vertex = PlyData.read(str(save_ply(scene, tmp_path / "s.ply")))["vertex"]
    assert [float(vertex[f"f_rest_{i}"][0]) for i in range(9)] == [0, 0, 0, 1, 2, 3, 0, 0, 0]


def test_scene_validates_shapes():
    scene = make_scene()
    with pytest.raises(ValueError):
        GaussianScene(scene.means, scene.quats[:, :3], scene.log_scales, scene.opacity_logits, scene.sh)
    with pytest.raises(ValueError):
        sh_degree_from_coeffs(5)


def test_camera_dict_roundtrip_and_scale():
    camera = Camera(intrinsics(100, 100, 50, 40), np.eye(4), 100, 80)
    restored = Camera.from_dict(camera.to_dict())
    assert np.array_equal(restored.K, camera.K) and np.array_equal(restored.world_to_cam, camera.world_to_cam)
    assert (restored.width, restored.height) == (100, 80)
    half = camera.scaled(0.5)
    assert (half.width, half.height) == (50, 40)
    assert half.K[0, 0] == 50 and half.K[1, 2] == 20
    assert np.allclose(intrinsics_from_fov(90, 200, 100)[1, 1], 50)


def test_conversions():
    assert to_rgb_uint8(np.array([-1.0, 0.5, 2.0])).tolist() == [0, 128, 255]
    assert composite(np.zeros((1, 1, 3)), np.array([[0.25]]), (1.0, 0.0, 0.0))[0, 0].tolist() == [0.75, 0, 0]
    assert expected_depth(np.array([3.0, 5.0]), np.array([0.0, 0.9])).tolist() == [0.0, 5.0]


def test_renderer_passes_activated_params_on_cpu():
    torch = pytest.importorskip("torch")
    calls = {}

    def rasterizer(means, quats, scales, opacities, colors, viewmats, Ks, width, height, **kw):
        calls.update(quats=quats, scales=scales, opacities=opacities, kw=kw, viewmats=viewmats)
        image = torch.zeros(1, height, width, 4)
        image[..., 0], image[..., 3] = 1.0, 2.5
        return image, torch.ones(1, height, width, 1), {}

    scene = make_scene()
    out = GaussianRenderer(scene, device="cpu", rasterizer=rasterizer).render(
        Camera(intrinsics(10, 10, 4, 3), np.eye(4), 8, 6))
    assert out.rgb.shape == (6, 8, 3) and out.rgb[0, 0].tolist() == [255, 0, 0]
    assert np.allclose(out.depth, 2.5)
    assert torch.allclose(calls["quats"].norm(dim=-1), torch.ones(len(scene)))
    assert torch.allclose(calls["scales"], torch.exp(torch.as_tensor(scene.log_scales)))
    assert calls["kw"]["sh_degree"] == 1 and calls["kw"]["render_mode"] == "RGB+ED"
    assert "backgrounds" not in calls["kw"]


def gpu_ready():
    torch = pytest.importorskip("torch")
    return torch.cuda.is_available()


@pytest.mark.skipif(not gpu_ready(), reason="CUDA が無い")
def test_gsplat_renders_single_gaussian_at_center():
    color = np.array([0.9, 0.2, 0.1])
    scene = GaussianScene(
        means=np.array([[0.0, 0.0, 4.0]], np.float32),
        quats=np.array([[1.0, 0, 0, 0]], np.float32),
        log_scales=np.log(np.full((1, 3), 0.3, np.float32)),
        opacity_logits=np.array([10.0], np.float32),
        sh=((color - 0.5) / SH_C0).astype(np.float32).reshape(1, 1, 3),
    )
    out = GaussianRenderer(scene).render(Camera(intrinsics(100, 100, 32, 32), np.eye(4), 64, 64))
    assert out.alpha[32, 32] > 0.9 and out.alpha[0, 0] < 0.01
    assert np.abs(out.rgb[32, 32].astype(int) - (color * 255).astype(int)).max() <= 3
    assert out.depth[32, 32] == pytest.approx(4.0, abs=0.05)
    assert out.depth[0, 0] == 0.0


def test_look_at_projects_target_to_center():
    from spatialsam.spatial.camera import look_at

    w2c = look_at(np.array([0.0, -5.0, 0.0]), np.zeros(3), np.array([0.0, 0.0, 1.0]))
    target = w2c @ np.array([0, 0, 0, 1.0])
    above = w2c @ np.array([0, 0, 1.0, 1.0])
    assert np.allclose(target[:3], [0, 0, 5])
    assert above[1] < 0  # world の上は画像の上（y 負）
    assert np.allclose(w2c[:3, :3] @ w2c[:3, :3].T, np.eye(3))
    with pytest.raises(ValueError):
        look_at(np.zeros(3), np.array([0, 0, 1.0]), np.array([0, 0, 1.0]))
