"""Evaluation ページ「実画像 vs 3D 描画」の実行フォーム #32。比較は別プロセスのジョブで動く。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from spatialsam.jobs import DONE, FAILED, active_gpu_jobs
from spatialsam.pipeline.comparison_job import (
    ComparisonJobStore,
    ComparisonRequest,
    ComparisonRunJob,
    launch_comparison,
)
from spatialsam.training.jobs import TrainingStore
from spatialsam.ui.state import get_config, get_registry
from spatialsam.ui.training_panel import guess_images_dir

LAST_COMPARISON_JOB_KEY = "last_comparison_job_id"
COMPARISON_RESULT_KEY = "comparison-result"  # 結果の選択欄（完了したら新しい結果を選ぶ）


def parse_prompts(text: str) -> list[str]:
    """カンマ・読点・改行区切りのプロンプト。空白を除き、重複は最初のものだけ残す。"""
    for separator in ("、", "\n"):
        text = text.replace(separator, ",")
    prompts = [p.strip() for p in text.split(",")]
    return list(dict.fromkeys(p for p in prompts if p))


def trained_scenes(store: TrainingStore, sparse_dir: str) -> dict[str, str]:
    """同じ COLMAP で学習し終えた 3DGS（ジョブ ID → PLY）。"""
    if not sparse_dir:
        return {}
    target = Path(sparse_dir).expanduser().resolve()
    return {job.id: str(job.ply_path) for job in store.done_jobs()
            if Path(job.load_request().sparse_dir).resolve() == target and job.ply_path.exists()}


def request_form() -> ComparisonRequest | None:
    sparse = st.text_input("COLMAP の sparse ディレクトリ", key="compare-sparse").strip()
    guessed = guess_images_dir(Path(sparse).expanduser()) if sparse else None
    images = st.text_input("実画像のディレクトリ（3DGS の学習に使ったものと同じ縮小率）", value=str(guessed or ""),
                           key=f"compare-images-{sparse}").strip()
    scenes = trained_scenes(TrainingStore(get_config().outputs_dir), sparse)
    chosen = st.multiselect("比べる 3DGS（この COLMAP で学習済みのもの）", list(scenes), default=list(scenes)[:1],
                            help="3D Rendering の「3DGS 学習」タブで学習したもの")
    cols = st.columns([3, 1, 1])
    prompts = parse_prompts(cols[0].text_input("プロンプト（カンマ区切り）", "bonsai, table", key="compare-prompts"))
    include_points = cols[1].checkbox("点群（学習なし）も比べる", value=True)
    holdout = int(cols[2].number_input("評価用に除く間隔", 1, 100, 8, help="3DGS の学習時と同じ値にする"))
    if not sparse or not images:
        return None
    if not (Path(sparse).expanduser() / "images.bin").exists() or not Path(images).expanduser().is_dir():
        st.error("sparse ディレクトリ（images.bin を含む）か実画像のディレクトリが見つかりません")
        return None
    try:
        return ComparisonRequest(str(Path(sparse).expanduser().resolve()), str(Path(images).expanduser().resolve()),
                                 prompts, {name: scenes[name] for name in chosen}, include_points, holdout)
    except ValueError as error:
        st.warning(str(error))
        return None


def start(request: ComparisonRequest) -> ComparisonRunJob:
    get_registry().release()  # このプロセスの SAM を GPU から降ろし、比較ジョブ側で読み込む
    job = ComparisonJobStore(get_config().outputs_dir).create(request)
    launch_comparison(job)
    return job


@st.fragment(run_every=2)
def live_status(job_id: str) -> None:
    job = ComparisonJobStore(get_config().outputs_dir).open(job_id)
    status = job.status()
    total = status.get("total") or job.load_request().num_steps()
    step = min(status.get("step", 0), total)
    state = status["state"]
    if state == DONE:
        st.success(f"比較が終わりました: `{job.comparison_id}`（下の「比較結果」で選べます）")
    elif state == FAILED:
        st.error(status.get("message", "失敗"))
    elif not job.is_active():
        st.warning("プロセスが見つかりません（中断された可能性があります）")
    else:
        st.progress(step / total, text=f"{step}/{total} — {status.get('message', '')}")
    with st.expander("ログ"):
        st.code(job.log_tail())


def run_panel() -> None:
    """比較の実行フォームと、直近のジョブの進捗。"""
    with st.expander("比較を実行する", expanded=not st.session_state.get(LAST_COMPARISON_JOB_KEY)):
        st.caption("評価用視点（3DGS の学習に使っていない視点）で、実画像と各シーンの描画を画質と SAM 3.1 の結果で比べます。"
                   "SAM を「(シーン数 + 1) × プロンプト数」回実行するので数分かかります（別プロセスで動き、ページを離れても続きます）。")
        request = request_form()
        busy = active_gpu_jobs(get_config().outputs_dir)
        if busy:
            st.warning(f"GPU を使うジョブが実行中です（{busy[0].id}）。終わってから実行してください")
        if request is not None:
            st.caption(f"SAM の実行回数: {len(request.prompts) * (len(request.sources) + request.include_points + 1)}")
        if st.button("比較を開始", type="primary", disabled=request is None or bool(busy)):
            st.session_state[LAST_COMPARISON_JOB_KEY] = start(request).id
            st.rerun()  # 実行中の判定をすぐ反映して二重起動を防ぐ
    job_id = st.session_state.get(LAST_COMPARISON_JOB_KEY)
    if job_id:
        live_status(job_id)
