"""Tab 2: 3D シーン（3DGS / 点群 / COLMAP）をカメラ軌道に沿って描画して動画を作る。3DGS の学習も起動できる。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import streamlit as st

from spatialsam.experiments.renders import RenderRun, RenderStore
from spatialsam.models.gpu import empty_cuda_cache
from spatialsam.pipeline.rendering import RenderJob, run_render
from spatialsam.spatial.colmap import ColmapView
from spatialsam.spatial.gaussians import GaussianScene
from spatialsam.spatial.renderer import GaussianRenderer
from spatialsam.spatial.scene_source import AXES, SceneSource, default_orbit
from spatialsam.spatial.trajectory import dataset_trajectory, fit_max_side, orbit_summary, orbit_trajectory
from spatialsam.ui.state import SELECTED_VIDEO_KEY, get_config, show_gpu_status
from spatialsam.ui.training_panel import PLY_PATH_KEY, SPARSE_PATH_KEY, training_panel

LAST_RENDER_KEY = "last_render_id"
ORBIT, DATASET = "オービット（注視点の周りを一周）", "データセットの撮影カメラ（COLMAP）"


@st.cache_resource(max_entries=2, show_spinner="シーンを読み込んでいます")
def load_source(source: SceneSource) -> tuple[GaussianScene, list[ColmapView]]:
    return source.load_scene(), source.load_views()


def _optional_path(label: str, help_text: str, key: str) -> Path | None:
    value = st.text_input(label, help=help_text, key=key).strip()
    return Path(value).expanduser() if value else None


def select_source() -> SceneSource | None:
    st.subheader("シーン")
    ply = _optional_path("PLY のパス（3DGS または点群）", "INRIA 形式の 3DGS PLY、または x/y/z（と red/green/blue）を持つ点群 PLY", PLY_PATH_KEY)
    sparse = _optional_path("COLMAP の sparse ディレクトリ（任意）",
                            "cameras.bin / images.bin / points3D.bin を含むディレクトリ。PLY が無ければ points3D を描く",
                            SPARSE_PATH_KEY)
    if ply is None and sparse is None:
        st.info("PLY か COLMAP のパスを入力してください")
        return None
    try:
        return SceneSource(ply, sparse)
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        return None


def orbit_frames(scene: GaussianScene, views: list[ColmapView]) -> tuple[list, dict]:
    cols = st.columns(4)
    up_axis = "+z" if views else cols[0].selectbox("上方向の軸", list(AXES), index=4)
    num_frames = cols[1].number_input("フレーム数", 2, 1000, 120)
    width = cols[2].number_input("幅 [px]", 64, 1920, 854, step=2)
    height = cols[3].number_input("高さ [px]", 64, 1080, 480, step=2)
    base = default_orbit(scene, views, up_axis, num_frames=int(num_frames), width=int(width), height=int(height))
    cols = st.columns(3)
    radius = cols[0].number_input("半径", 0.001, 1e4, float(round(base.radius, 3)), format="%.3f")
    elevation = cols[1].slider("仰角 [deg]", -80.0, 80.0, float(round(base.elevation_deg, 1)))
    fov = cols[2].slider("縦の画角 [deg]", 20.0, 100.0, base.fov_y_deg)
    spec = replace(base, radius=float(radius), elevation_deg=float(elevation), fov_y_deg=float(fov))
    return [("", c) for c in orbit_trajectory(spec)], orbit_summary(spec)


def dataset_frames(views: list[ColmapView]) -> tuple[list, dict]:
    cols = st.columns(4)
    stride = int(cols[0].number_input("間引き（stride）", 1, 100, 1))
    max_frames = int(cols[1].number_input("最大フレーム数（0 = すべて）", 0, 10000, 0)) or None
    max_side = int(cols[2].number_input("長辺 [px]", 64, 4096, 854, step=2))
    holdout = int(cols[3].number_input("評価用視点を除く間隔（0 = 除かない）", 0, 100, 0,
                                       help="8 にすると 3DGS 学習時の評価用視点（8 枚に 1 枚）を除く"))
    cameras = dataset_trajectory(views, stride, max_frames, exclude_holdout_every=holdout)
    frames = [(name, fit_max_side(camera, max_side)) for name, camera in cameras]
    return frames, {"type": "dataset", "stride": stride, "max_frames": max_frames, "max_side": max_side,
                    "exclude_holdout_every": holdout}


def select_trajectory(scene: GaussianScene, views: list[ColmapView]) -> tuple[list, dict]:
    st.subheader("カメラ軌道")
    kind = st.radio("種類", [ORBIT, DATASET] if views else [ORBIT], horizontal=True)
    return dataset_frames(views) if kind == DATASET else orbit_frames(scene, views)


def render(source: SceneSource, scene: GaussianScene, frames: list, trajectory: dict, fps: float) -> RenderRun:
    with st.status("描画中", expanded=True) as status:
        renderer = GaussianRenderer(scene)
        try:
            run = run_render(RenderJob(Path(source.label), frames, fps, trajectory, source.to_dict()), renderer,
                             RenderStore(get_config().outputs_dir), progress=status.write)
        finally:
            del renderer
            empty_cuda_cache()
        status.update(label="完了", state="complete")
    return run


def show_run(run: RenderRun) -> None:
    meta = run.load_metadata()
    perf = meta["performance"]
    cols = st.columns(4)
    cols[0].metric("フレーム数", meta["num_frames"])
    cols[1].metric("解像度", "x".join(map(str, meta["resolution"])))
    cols[2].metric("描画時間 [s]", f"{perf['elapsed_sec']:.1f}")
    cols[3].metric("Peak VRAM [MiB]", "—" if perf["peak_vram_mib"] is None else f"{perf['peak_vram_mib']:,.0f}")
    left, right = st.columns(2)
    left.video(str(run.video_path))
    left.caption(f"RGB（`{run.video_path.name}`）")
    right.video(str(run.depth_video_path))
    near, far = meta["depth_preview_range"]
    right.caption(f"深度（赤 = 近い {near:.2f} 〜 青 = 遠い {far:.2f}）")
    if st.button("この動画を Video Segmentation で使う", type="primary"):
        st.session_state[SELECTED_VIDEO_KEY] = str(run.video_path)
        st.switch_page("pages/1_video_segmentation.py")


def select_run(store: RenderStore) -> RenderRun | None:
    ids = store.list_ids()
    if not ids:
        return None
    last = st.session_state.get(LAST_RENDER_KEY)
    choice = st.selectbox("描画結果", ids, index=ids.index(last) if last in ids else 0)
    return store.open(choice)


def render_panel(gpu_available: bool) -> None:
    source = select_source()
    if source is not None:
        scene, views = load_source(source)
        bounds = np.round(np.stack(scene.bounds(0.02)), 2).tolist()
        st.caption(f"Gaussian {len(scene):,} 個 / SH 次数 {scene.sh_degree} / 撮影カメラ {len(views)} 台 / 範囲 {bounds}")
        frames, trajectory = select_trajectory(scene, views)
        fps = st.number_input("FPS", 1.0, 60.0, 24.0)
        if st.button("描画", type="primary", disabled=not gpu_available or not frames):
            st.session_state[LAST_RENDER_KEY] = render(source, scene, frames, trajectory, fps).id
    st.divider()
    run = select_run(RenderStore(get_config().outputs_dir))
    if run is not None:
        show_run(run)


def main() -> None:
    st.set_page_config(page_title="3D Rendering", layout="wide")
    st.title("3D Scene & Rendering")
    status = show_gpu_status()
    render_tab, training_tab = st.tabs(["描画", "3DGS 学習"])
    with render_tab:
        render_panel(status.available)
    with training_tab:
        training_panel(status.available)


main()
