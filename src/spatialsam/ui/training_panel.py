"""3D Rendering ページの「3DGS 学習」タブ。学習は別プロセスで動き、ここでは起動と進捗表示だけ行う。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from spatialsam.training.jobs import DONE, FAILED, TrainingJob, TrainingStore, TrainRequest, launch
from spatialsam.ui.state import get_config, get_registry

PLY_PATH_KEY, SPARSE_PATH_KEY = "ply-path", "sparse-path"  # 描画タブの入力欄


def guess_images_dir(sparse_dir: Path) -> Path | None:
    """Mip-NeRF 360 の配置（<scene>/sparse/0 と <scene>/images_4）なら 1/4 縮小画像を使う。"""
    scene_root = sparse_dir.parent.parent if sparse_dir.name.isdigit() else sparse_dir.parent
    for name in ("images_4", "images_2", "images"):
        if (scene_root / name).is_dir():
            return scene_root / name
    return None


def training_store() -> TrainingStore:
    return TrainingStore(get_config().outputs_dir)


def request_form() -> TrainRequest | None:
    sparse = st.text_input("COLMAP の sparse ディレクトリ", key="train-sparse").strip()
    sparse_dir = Path(sparse).expanduser() if sparse else None
    guessed = guess_images_dir(sparse_dir) if sparse_dir else None
    images = st.text_input("学習画像のディレクトリ", value=str(guessed or ""), key=f"train-images-{sparse}",
                           help="8GB VRAM では 1/4 縮小画像（images_4）を推奨").strip()
    cols = st.columns(3)
    steps = int(cols[0].number_input("ステップ数", 100, 100_000, 7000, step=500))
    cap_max = int(cols[1].number_input("Gaussian 数の上限", 10_000, 3_000_000, 1_000_000, step=100_000))
    holdout = int(cols[2].number_input("評価用に除く間隔（0 = 全部学習）", 0, 100, 8))
    if not sparse_dir or not images:
        return None
    if not (sparse_dir / "points3D.bin").exists() or not Path(images).expanduser().is_dir():
        st.error("sparse ディレクトリ（points3D.bin を含む）か画像ディレクトリが見つかりません")
        return None
    return TrainRequest(str(sparse_dir.resolve()), str(Path(images).expanduser().resolve()),
                        {"steps": steps, "cap_max": cap_max}, holdout)


def start(request: TrainRequest) -> TrainingJob:
    get_registry().release()  # SAM と学習を同時に GPU に載せない
    job = training_store().create(request)
    launch(job)
    return job


def open_in_render_tab(job: TrainingJob) -> None:
    """on_click で呼ぶ（ウィジェット生成前に入力欄の値を書き換えるため）。タブは切り替えられないので通知する。"""
    st.session_state[PLY_PATH_KEY] = str(job.ply_path)
    st.session_state[SPARSE_PATH_KEY] = job.load_request().sparse_dir
    st.toast("描画タブに PLY と COLMAP のパスを入力しました")


def show_job(job: TrainingJob) -> None:
    status = job.status()
    progress = status.get("progress")
    if progress:
        st.progress(progress["step"] / progress["total"],
                    text=f"{progress['step']:,}/{progress['total']:,} step / PSNR(学習画像) {progress['psnr']:.2f} / "
                         f"Gaussian {progress['num_gaussians']:,} / {progress['elapsed_sec']:.0f} 秒 / "
                         f"Peak VRAM {progress['peak_vram_mib']:,.0f} MiB")
    state = status["state"]
    if state == DONE:
        metrics = job.metrics()
        st.success(f"完了。評価用視点 {metrics.get('num_views', 0)} 枚: PSNR {metrics.get('psnr', float('nan')):.2f} / "
                   f"SSIM {metrics.get('ssim', float('nan')):.3f}")
    elif state == FAILED:
        st.error(status.get("message", "失敗"))
    elif not job.is_active():
        st.warning("プロセスが見つかりません（中断された可能性があります）")
    else:
        st.info(status.get("message", ""))
    with st.expander("ログ"):
        st.code(job.log_path.read_text(encoding="utf-8")[-4000:] if job.log_path.exists() else "")


def jobs_table(store: TrainingStore) -> pd.DataFrame:
    rows = []
    for job in map(store.open, store.list_ids()):
        status, metrics = job.status(), job.metrics()
        rows.append({"id": job.id, "state": status["state"], "steps": status.get("config", {}).get("steps"),
                     "psnr": metrics.get("psnr"), "ssim": metrics.get("ssim"),
                     "gaussians": (status.get("progress") or {}).get("num_gaussians")})
    return pd.DataFrame(rows)


@st.fragment(run_every=2)
def live_status(job_id: str) -> None:
    show_job(training_store().open(job_id))


def training_panel(gpu_available: bool) -> None:
    store = training_store()
    active = store.active_jobs()
    st.caption("学習は別プロセスで動くので、ページを離れても続きます。学習中は SAM 3.1 の推論を止めます。")
    request = request_form()
    if active:
        st.warning(f"学習中のジョブがあります: {active[0].id}")
    if st.button("学習を開始", type="primary", disabled=request is None or bool(active) or not gpu_available):
        st.session_state["last_training_id"] = start(request).id
        st.rerun()  # 実行中の判定（開始ボタンの無効化・警告）をすぐ反映して二重起動を防ぐ
    ids = store.list_ids()
    if not ids:
        return
    st.divider()
    last = st.session_state.get("last_training_id")
    job_id = st.selectbox("ジョブ", ids, index=ids.index(last) if last in ids else 0)
    live_status(job_id)
    job = store.open(job_id)
    if job.status()["state"] == DONE:
        # fragment の外に置く（fragment 内のボタンは描画タブの入力欄を再描画しない）
        st.button("この 3DGS を描画タブの入力に設定", on_click=open_in_render_tab, args=(job,))
    st.dataframe(jobs_table(store), hide_index=True)
