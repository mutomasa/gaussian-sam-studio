"""Tab 1: テキストプロンプトによる動画セグメンテーション。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from spatialsam.config import PROJECT_ROOT, AppConfig
from spatialsam.experiments.store import Experiment
from spatialsam.models.factory import SEGMENTER_NAMES, create_segmenter
from spatialsam.pipeline.segmentation import SegmentationJob, run_segmentation
from spatialsam.ui.state import LAST_EXPERIMENT_KEY, get_config, get_registry, get_store, show_gpu_status
from spatialsam.video.io import FrameSampling, frame_path
from spatialsam.video.overlay import overlay_masks, render_mask_only
from spatialsam.video.upload import save_upload, validate_video_upload

SAMPLES_DIR = PROJECT_ROOT / "samples"


def select_video(config: AppConfig) -> Path | None:
    uploaded = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi", "mkv"])
    if uploaded is not None:
        validate_video_upload(uploaded.name, uploaded.size, config.max_upload_mib)
        upload_dir = Path(tempfile.gettempdir()) / "spatialsam-uploads" / uploaded.file_id
        return save_upload(uploaded.getvalue(), uploaded.name, upload_dir)
    samples = sorted(SAMPLES_DIR.glob("*.mp4"))
    if not samples:
        st.info("`samples/` に MP4 を置くとここから選べます")
        return None
    choice = st.selectbox("またはサンプル動画", [None, *samples], format_func=lambda p: p.name if p else "—")
    return choice


def sidebar_settings(config: AppConfig) -> tuple[str, FrameSampling]:
    with st.sidebar:
        st.header("推論設定")
        default = SEGMENTER_NAMES.index(config.default_segmenter) if config.default_segmenter in SEGMENTER_NAMES else 0
        segmenter = st.selectbox("セグメンタ", SEGMENTER_NAMES, index=default)
        stride = st.number_input("フレーム間引き（stride）", 1, 30, 1)
        max_frames = st.number_input("最大フレーム数（0 = 無制限）", 0, 10000, 200)
        max_side = st.number_input("入力の長辺 [px]（0 = 元解像度）", 0, 4096, 720, step=32)
        if st.button("GPU 上のモデルを解放"):
            get_registry().release()
    sampling = FrameSampling(int(stride), int(max_frames) or None, int(max_side) or None)
    return segmenter, sampling


def run(video: Path, prompt: str, segmenter_name: str, sampling: FrameSampling, config: AppConfig) -> Experiment:
    segmenter = get_registry().get(segmenter_name, lambda: create_segmenter(segmenter_name, config))
    model_info = {"repo_id": config.sam31_source.repo_id, "revision": config.sam31_source.revision} \
        if segmenter_name == "sam3.1" else {}
    job = SegmentationJob(video, prompt, sampling, model_info)
    with st.status("処理中", expanded=True) as status:
        outcome = run_segmentation(job, segmenter, get_store(), progress=status.write)
        status.update(label="完了", state="complete")
    return outcome.experiment


def show_summary(experiment: Experiment) -> None:
    meta = experiment.load_metadata()
    perf = meta["performance"]
    cols = st.columns(4)
    cols[0].metric("フレーム数", len(meta["frame_indices"]))
    cols[1].metric("物体数", len(meta["object_ids"]))
    cols[2].metric("処理時間 [s]", f"{perf['elapsed_sec']:.1f}")
    cols[3].metric("Peak VRAM [MiB]", "—" if perf["peak_vram_mib"] is None else f"{perf['peak_vram_mib']:,.0f}")
    st.caption(f"実験 ID: `{experiment.id}` / プロンプト: `{meta['prompt']}` / モデル: `{meta['model']}`")


def show_videos(experiment: Experiment) -> None:
    left, right = st.columns(2)
    left.subheader("元動画")
    left.video(str(next(experiment.input_dir.iterdir())))
    right.subheader("マスク重畳")
    right.video(str(experiment.overlay_path))


def show_frame_viewer(experiment: Experiment) -> None:
    indices = experiment.load_metadata()["frame_indices"]
    position = st.slider("フレーム", 0, len(indices) - 1, 0)
    frame = cv2.imread(str(frame_path(experiment.frames_dir, position)))
    masks = experiment.load_prediction().by_frame().get(indices[position])
    cols = st.columns(3)
    cols[0].image(frame, channels="BGR", caption=f"元フレーム #{indices[position]}")
    cols[1].image(overlay_masks(frame, masks), channels="BGR", caption="重畳")
    cols[2].image(render_mask_only(frame.shape[:2], masks), channels="BGR", caption="マスク")
    if masks is not None:
        st.dataframe(pd.DataFrame({"object_id": list(masks.scores), "score": list(masks.scores.values())}))


def main() -> None:
    st.set_page_config(page_title="Video Segmentation", layout="wide")
    st.title("Video Segmentation")
    show_gpu_status()
    config = get_config()
    segmenter_name, sampling = sidebar_settings(config)
    try:
        video = select_video(config)
    except ValueError as error:
        st.error(str(error))
        return
    prompt = st.text_input("テキストプロンプト", placeholder="例: robot gripper")
    if st.button("推論を実行", type="primary", disabled=video is None or not prompt.strip()):
        try:
            experiment = run(video, prompt.strip(), segmenter_name, sampling, config)
        except Exception as error:  # noqa: BLE001 — UIでは失敗内容を表示して継続する
            st.exception(error)
            return
        st.session_state[LAST_EXPERIMENT_KEY] = experiment.id
    experiment_id = st.session_state.get(LAST_EXPERIMENT_KEY)
    if experiment_id:
        experiment = get_store().open(experiment_id)
        show_summary(experiment)
        show_videos(experiment)
        show_frame_viewer(experiment)


main()
