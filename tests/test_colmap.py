import struct

import numpy as np
import pytest

from spatialsam.spatial.colmap import load_views, quat_to_rotation, read_cameras, read_images, read_points3d


def write_sparse(root):
    root.mkdir()
    with (root / "cameras.bin").open("wb") as f:
        f.write(struct.pack("<Q", 2))
        f.write(struct.pack("<iiQQ4d", 1, 1, 800, 600, 500.0, 510.0, 400.0, 300.0))  # PINHOLE
        f.write(struct.pack("<iiQQ3d", 2, 0, 100, 50, 80.0, 50.0, 25.0))  # SIMPLE_PINHOLE
    with (root / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", 2))
        for image_id, name, camera_id, qvec in ((7, b"b.jpg", 1, (1, 0, 0, 0)), (3, b"a.jpg", 2, (0, 0, 0, 1))):
            f.write(struct.pack("<i7di", image_id, *qvec, 1.0, 2.0, 3.0, camera_id))
            f.write(name + b"\x00")
            f.write(struct.pack("<Q", 2))
            f.write(struct.pack("<ddq", 1.0, 2.0, -1) * 2)
    with (root / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", 2))
        for pid, xyz, rgb, track in ((1, (0.0, 1.0, 2.0), (255, 0, 10), 3), (2, (3.0, 4.0, 5.0), (1, 2, 3), 0)):
            f.write(struct.pack("<Q3d3BdQ", pid, *xyz, *rgb, 0.5, track))
            f.write(struct.pack("<ii", 7, 0) * track)
    return root


def test_read_cameras_and_intrinsics(tmp_path):
    cameras = read_cameras(write_sparse(tmp_path / "s") / "cameras.bin")
    assert cameras[1].model == "PINHOLE" and (cameras[1].width, cameras[1].height) == (800, 600)
    assert np.allclose(cameras[1].K(), [[500, 0, 400], [0, 510, 300], [0, 0, 1]])
    assert np.allclose(cameras[2].K(), [[80, 0, 50], [0, 80, 25], [0, 0, 1]])


def test_read_images_skips_points2d(tmp_path):
    images = read_images(write_sparse(tmp_path / "s") / "images.bin")
    assert sorted(images) == [3, 7]
    assert images[7].name == "b.jpg" and images[3].camera_id == 2
    assert np.allclose(images[7].world_to_cam()[:3, 3], [1, 2, 3])


def test_read_points3d_skips_tracks(tmp_path):
    points = read_points3d(write_sparse(tmp_path / "s") / "points3D.bin")
    assert points.xyz.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert points.rgb.tolist() == [[255, 0, 10], [1, 2, 3]]


def test_load_views_sorted_by_name(tmp_path):
    views = load_views(write_sparse(tmp_path / "s"))
    assert [v.name for v in views] == ["a.jpg", "b.jpg"]
    assert views[0].camera.width == 100
    half = views[1].camera.resized(400, 300)
    assert np.allclose(half.K, [[250, 0, 200], [0, 255, 150], [0, 0, 1]])


def test_quat_to_rotation():
    assert np.allclose(quat_to_rotation(np.array([1.0, 0, 0, 0])), np.eye(3))
    rz = quat_to_rotation(np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]))  # z 軸 90°
    assert np.allclose(rz @ [1, 0, 0], [0, 1, 0])


def test_truncated_file_raises(tmp_path):
    path = tmp_path / "cameras.bin"
    path.write_bytes(struct.pack("<Q", 1) + b"\x00" * 4)
    with pytest.raises(ValueError):
        read_cameras(path)
