"""Tab 4: GT マスクとの比較による精度評価、実画像 vs 3D 描画の比較、実験一覧。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from spatialsam.config import PROJECT_ROOT
from spatialsam.evaluation.ground_truth import load_gt_dir, load_gt_zip
from spatialsam.experiments.comparisons import REAL, Comparison, ComparisonStore
from spatialsam.experiments.store import Experiment, ExperimentStore
from spatialsam.pipeline.evaluation import EvaluationReport, evaluate_experiment
from spatialsam.samples.davis import default_gt_dir
from spatialsam.ui.state import LAST_EXPERIMENT_KEY, get_config, get_store
from spatialsam.video.overlay import overlay_masks

SAMPLES_DIR = PROJECT_ROOT / "samples"


def select_experiment(store: ExperimentStore) -> Experiment | None:
    ids = store.list_ids()
    if not ids:
        st.info("実験がありません。Video Segmentation で推論してください")
        return None
    last = st.session_state.get(LAST_EXPERIMENT_KEY)
    index = ids.index(last) if last in ids else 0
    return store.open(st.selectbox("実験", ids, index=index))


def load_ground_truth(experiment: Experiment) -> dict | None:
    st.caption("GT は PNG 連番（ファイル名末尾の数字 = 元動画のフレーム番号、画素値 > 0 が前景）")
    uploaded = st.file_uploader("GT マスク（PNG を含む zip）", type=["zip"])
    if uploaded is not None:
        return load_gt_zip(uploaded.getvalue())
    default = default_gt_dir(SAMPLES_DIR, experiment.load_metadata()["input"]["video"])
    directory = st.text_input("または GT ディレクトリのパス（サンプル動画は自動入力）",
                              value=str(default or ""), key=f"gt-dir-{experiment.id}")
    if directory and Path(directory).is_dir():
        return load_gt_dir(Path(directory))
    return None


def show_report(report: EvaluationReport) -> None:
    summary = report.summary
    cols = st.columns(4)
    for col, key in zip(cols, ("J&F", "J", "F", "Dice"), strict=True):
        col.metric(key, f"{summary[key]:.3f}")
    if report.missing_gt_frames:
        st.caption(f"GT が無く評価対象外のフレーム: {len(report.missing_gt_frames)}")
    frame = pd.DataFrame([s.as_dict() for s in report.frame_scores])
    st.line_chart(frame, x="frame_index", y=["iou", "dice", "boundary_f"])
    st.download_button("per_frame.csv", frame.to_csv(index=False), "per_frame.csv", "text/csv")
    st.download_button("metrics.json", json.dumps(summary, indent=2), "metrics.json", "application/json")


def experiments_table(store: ExperimentStore) -> pd.DataFrame:
    rows = []
    for experiment_id in store.list_ids():
        experiment = store.open(experiment_id)
        meta = experiment.load_metadata()
        row = {"id": experiment_id, "prompt": meta["prompt"], "segmenter": meta["model"]["segmenter"],
               "frames": len(meta["frame_indices"]), "elapsed_sec": meta["performance"]["elapsed_sec"]}
        if experiment.has_evaluation():
            row.update(json.loads((experiment.evaluation_dir / "metrics.json").read_text()))
        rows.append(row)
    return pd.DataFrame(rows)


def gt_panel(store: ExperimentStore) -> None:
    experiment = select_experiment(store)
    if experiment is None:
        return
    gt_masks = load_ground_truth(experiment)
    if st.button("評価を実行", type="primary", disabled=not gt_masks):
        try:
            show_report(evaluate_experiment(experiment, gt_masks))
        except ValueError as error:
            st.error(str(error))


def overlaid_frame(comparison: Comparison, source: str, index: int, experiment: Experiment | None):
    frame = cv2.imread(str(comparison.frame_path(source, index)))
    if experiment is None:
        return frame
    masks = experiment.load_prediction().by_frame().get(experiment.load_metadata()["frame_indices"][index])
    return overlay_masks(frame, masks)


def show_comparison(comparison: Comparison, store: ExperimentStore) -> None:
    summary = comparison.load_summary()
    st.caption(f"評価用視点 {summary['num_views']} 枚（学習に使っていない視点）/ {summary['resolution'][0]}x"
               f"{summary['resolution'][1]} / シーン: {summary['sources']}")
    st.markdown("**画質（実画像との比較）** — PSNR・SSIM は高いほど、LPIPS は低いほど実画像に近い")
    st.dataframe(pd.DataFrame(summary["quality"]).T, use_container_width=False)
    st.markdown("**SAM 3.1（実画像での結果を基準にした一致度）**")
    st.dataframe(pd.DataFrame(summary["segmentation"]), hide_index=True)
    sources = [REAL, *summary["sources"]]
    cols = st.columns(2)
    prompt = cols[0].selectbox("プロンプト", summary["prompts"])
    index = cols[1].slider("視点", 0, summary["num_views"] - 1, 0)
    st.caption(f"視点: {summary['views'][index]}")
    for col, source in zip(st.columns(len(sources)), sources, strict=True):
        experiment_id = summary["experiments"].get(f"{prompt}/{source}")
        experiment = store.open(experiment_id) if experiment_id else None
        col.image(overlaid_frame(comparison, source, index, experiment), channels="BGR", caption=source)
    quality = comparison.load_table("quality")
    if not quality.empty:
        st.line_chart(quality.pivot(index="index", columns="source", values="psnr"), y_label="PSNR [dB]")


def comparison_panel(store: ExperimentStore) -> None:
    comparisons = ComparisonStore(get_config().outputs_dir)
    ids = comparisons.list_ids()
    if not ids:
        st.info("`scripts/compare_render_vs_real.py` で実画像と 3D 描画を比較できます")
        return
    show_comparison(comparisons.open(st.selectbox("比較結果", ids)), store)


def main() -> None:
    st.set_page_config(page_title="Evaluation", layout="wide")
    st.title("Evaluation & Experiments")
    store = get_store()
    gt_tab, comparison_tab = st.tabs(["GT マスクとの評価", "実画像 vs 3D 描画"])
    with gt_tab:
        gt_panel(store)
    with comparison_tab:
        comparison_panel(store)
    st.subheader("実験一覧")
    table = experiments_table(store)
    st.dataframe(table)
    st.download_button("experiments.csv", table.to_csv(index=False), "experiments.csv", "text/csv")


main()
