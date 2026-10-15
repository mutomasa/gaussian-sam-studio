from spatialsam.ui.training_panel import guess_images_dir


def test_guess_images_dir_prefers_quarter_resolution(tmp_path):
    scene = tmp_path / "bonsai"
    (scene / "sparse" / "0").mkdir(parents=True)
    (scene / "images").mkdir()
    assert guess_images_dir(scene / "sparse" / "0") == scene / "images"
    (scene / "images_4").mkdir()
    assert guess_images_dir(scene / "sparse" / "0") == scene / "images_4"
    assert guess_images_dir(tmp_path / "other" / "sparse") is None
