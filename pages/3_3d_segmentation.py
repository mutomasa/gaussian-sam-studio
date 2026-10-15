"""Tab 3: SAM の 2D マスクを 3D（Gaussian）に持ち上げ、対象を強調・抽出して評価する。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from spatialsam.experiments.comparisons import REAL, ComparisonStore
from spatialsam.experiments.renders import RenderStore
from spatialsam.experiments.segmentation3d import Segmentation3D, Segmentation3DStore
from spatialsam.experiments.store import Experiment, ExperimentStore
from spatialsam.models.gpu import empty_cuda_cache
from spatialsam.pipeline.comparison import comparison_cameras
from spatialsam.pipeline.lifting import evaluate_reprojection, run_lifting
from spatialsam.spatial.gaussians import GaussianScene, load_ply
from spatialsam.spatial.mask_lifting import LiftSettings
from spatialsam.spatial.renderer import GaussianRenderer
from spatialsam.spatial.scene_highlight import dim_others
from spatialsam.spatial.scene_source import SceneSource
from spatialsam.ui.state import get_config, get_store, show_gpu_status

LAST_SEG3D_KEY = "last_seg3d_id"


@st.cache_resource(max_entries=1, show_spinner="シーンを読み込んでいます")
def load_scene(source: SceneSource) -> GaussianScene:
    return source.load_scene()


def render_experiments(store: ExperimentStore, renders: RenderStore) -> list[Experiment]:
    """入力が 3D Rendering の動画である SAM の実験だけ。"""
    render_ids = set(renders.list_ids())
    experiments = (store.open(i) for i in store.list_ids())
    return [e for e in experiments if e.load_metadata()["input"]["video"].removesuffix(".mp4") in render_ids]


def experiment_label(experiment: Experiment) -> str:
    meta = experiment.load_metadata()
    return f"{experiment.id} 「{meta['prompt']}」 {meta['input']['video']}（{len(meta['frame_indices'])} フレーム）"


def lift_settings() -> LiftSettings:
    with st.expander("持ち上げの設定"):
        cols = st.columns(3)
        ratio = cols[0].slider("信頼度のしきい値", 0.1, 0.9, LiftSettings.min_ratio, 0.05,
                               help="見えた視点のうちマスク内だった割合がこれ以上の Gaussian を対象にする")
        views = cols[1].number_input("最少の可視視点数", 1, 50, LiftSettings.min_views)
        tolerance = cols[2].slider("深度の許容誤差", 0.01, 0.3, LiftSettings.depth_tolerance, 0.01,
                                   help="描画深度よりこの割合以上奥にある Gaussian は遮蔽とみなす")
    return LiftSettings(float(tolerance), LiftSettings.min_alpha, float(ratio), int(views))


def lift_panel(experiments: ExperimentStore, renders: RenderStore, store: Segmentation3DStore) -> None:
    candidates = render_experiments(experiments, renders)
    if not candidates:
        st.info("3D Rendering で描いた動画を Video Segmentation で推論すると、ここで 3D に持ち上げられます")
        return
    experiment = st.selectbox("SAM の実験（3D 描画動画が入力）", candidates, format_func=experiment_label)
    settings = lift_settings()
    if st.button("3D に持ち上げる", type="primary"):
        with st.spinner("各視点の投影と投票をしています"):
            st.session_state[LAST_SEG3D_KEY] = run_lifting(experiment, renders, store, settings).id


def show_summary(run: Segmentation3D) -> dict:
    summary = run.load_summary()
    cols = st.columns(4)
    cols[0].metric("対象の Gaussian", f"{summary['num_labeled']:,}")
    cols[1].metric("全 Gaussian", f"{summary['num_gaussians']:,}")
    cols[2].metric("使った視点", summary["num_views"])
    cols[3].metric("一度も見えなかった", f"{summary['num_never_visible']:,}")
    st.caption(f"プロンプト「{summary['prompt']}」/ SAM 実験 `{summary['experiment']}` / 描画 `{summary['render']}`")
    st.dataframe(pd.DataFrame(summary["objects"]), hide_index=True)
    return summary


def confidence_histogram(run: Segmentation3D) -> None:
    confidence = np.maximum.reduce(list(run.load_confidence().values()))
    visible = confidence[confidence > 0]
    counts, edges = np.histogram(visible, bins=20, range=(0, 1))
    bins = [f"{low:.2f}–{high:.2f}" for low, high in zip(edges[:-1], edges[1:], strict=True)]
    st.bar_chart(pd.DataFrame({"Gaussian 数": counts}, index=bins),
                 x_label="信頼度（見えた視点のうちマスク内だった割合）")


@st.cache_resource(max_entries=1, show_spinner="描画の準備をしています")
def view_renderers(run_root: str, _scene: GaussianScene) -> dict[str, GaussianRenderer]:
    """結果ごとに一度だけ GPU に載せる（視点を動かすたびに作り直さない）。"""
    empty_cuda_cache()
    run = Segmentation3D(Path(run_root))
    return {"元のシーン": GaussianRenderer(_scene),
            "対象を強調": GaussianRenderer(dim_others(_scene, run.any_labels())),
            "対象だけ": GaussianRenderer(load_ply(run.object_ply_path))}


def show_views(run: Segmentation3D, scene: GaussianScene, renders: RenderStore) -> None:
    poses = renders.open(run.load_summary()["render"]).load_poses()
    index = st.slider("視点（描画に使ったカメラ）", 0, len(poses) - 1, 0)
    camera = poses[index][1]
    renderers = view_renderers(str(run.root), scene)
    for col, (title, renderer) in zip(st.columns(3), renderers.items(), strict=True):
        col.image(renderer.render(camera).rgb, caption=title)
    st.download_button("対象の Gaussian（object.ply）", run.object_ply_path.read_bytes(), f"{run.id}-object.ply")


def evaluation_panel(run: Segmentation3D, scene: GaussianScene, experiments: ExperimentStore) -> None:
    st.subheader("評価（評価用視点の実画像との再投影 IoU）")
    summary = run.load_summary()
    comparisons = ComparisonStore(get_config().outputs_dir)
    usable = [c for c in map(comparisons.open, comparisons.list_ids())
              if f"{summary['prompt']}/{REAL}" in c.load_summary()["experiments"]]
    if not usable:
        st.info(f"プロンプト「{summary['prompt']}」を含む「実画像 vs 3D 描画」の比較結果があると評価できます"
                "（scripts/compare_render_vs_real.py）")
        return
    comparison = st.selectbox("比較結果（評価用視点と実画像の SAM 結果）", usable, format_func=lambda c: c.id)
    if st.button("評価する"):
        reference = experiments.open(comparison.load_summary()["experiments"][f"{summary['prompt']}/{REAL}"])
        with st.spinner("評価用視点にラベルを描いています"):
            evaluate_reprojection(run, scene, comparison_cameras(comparison), reference, GaussianRenderer)
    evaluation = run.load_summary().get("reprojection")
    if evaluation:
        cols = st.columns(3)
        cols[0].metric("再投影 IoU（平均）", f"{evaluation['mean_iou']:.3f}")
        detected = evaluation["mean_iou_where_reference_detected"]
        cols[1].metric("実画像で検出された視点の平均", "—" if detected is None else f"{detected:.3f}")
        cols[2].metric("実画像の SAM が見落とした視点", len(evaluation["reference_empty_views"]))
        st.bar_chart(pd.DataFrame({"IoU": evaluation["per_view_iou"]}), x_label="評価用視点")


def main() -> None:
    st.set_page_config(page_title="3D Segmentation", layout="wide")
    st.title("3D Segmentation")
    show_gpu_status()
    config = get_config()
    experiments, renders = get_store(), RenderStore(config.outputs_dir)
    store = Segmentation3DStore(config.outputs_dir)
    lift_panel(experiments, renders, store)
    ids = store.list_ids()
    if not ids:
        return
    st.divider()
    last = st.session_state.get(LAST_SEG3D_KEY)
    run = store.open(st.selectbox("3D セグメンテーション結果", ids, index=ids.index(last) if last in ids else 0))
    summary = show_summary(run)
    scene = load_scene(SceneSource.from_dict(summary["scene_source"]))
    confidence_histogram(run)
    show_views(run, scene, renders)
    evaluation_panel(run, scene, experiments)


main()
