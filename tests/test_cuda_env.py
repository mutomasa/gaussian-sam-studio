from pathlib import Path

from spatialsam.spatial.cuda_env import EXTENSIONS_DIR, NVCC_FLAGS, build_env, configure_cuda_build_env, ensure_cudart_link


def test_build_env_sets_toolchain():
    env = build_env(Path("/cu13"), "8.6", {"PATH": "/usr/bin"})
    assert env["CUDA_HOME"] == "/cu13"
    assert env["PATH"].startswith("/cu13/bin")
    assert env["PATH"].endswith("/usr/bin")
    assert env["NVCC_PREPEND_FLAGS"] == NVCC_FLAGS
    assert env["TORCH_CUDA_ARCH_LIST"] == "8.6"
    assert env["TORCH_EXTENSIONS_DIR"] == str(EXTENSIONS_DIR)


def test_build_env_respects_user_settings():
    current = {"PATH": "", "TORCH_CUDA_ARCH_LIST": "8.0;8.6", "TORCH_EXTENSIONS_DIR": "/x", "NVCC_PREPEND_FLAGS": "-g"}
    env = build_env(Path("/cu13"), "8.6", current)
    assert "TORCH_CUDA_ARCH_LIST" not in env
    assert "TORCH_EXTENSIONS_DIR" not in env
    assert env["NVCC_PREPEND_FLAGS"] == f"{NVCC_FLAGS} -g"


def test_build_env_without_gpu_leaves_arch_unset():
    assert "TORCH_CUDA_ARCH_LIST" not in build_env(Path("/cu13"), None, {})


def test_ensure_cudart_link_is_idempotent(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libcudart.so.13").write_bytes(b"")
    ensure_cudart_link(tmp_path)
    ensure_cudart_link(tmp_path)
    assert (tmp_path / "lib" / "libcudart.so").resolve().name == "libcudart.so.13"


def test_existing_cuda_home_is_kept():
    env = {"CUDA_HOME": "/opt/cuda"}
    assert configure_cuda_build_env(env) == Path("/opt/cuda")
    assert env == {"CUDA_HOME": "/opt/cuda"}
