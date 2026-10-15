"""Tab 4: GT マスクとの比較による精度評価と実験一覧。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from spatialsam.evaluation.ground_truth import load_gt_dir, load_gt_zip
from spatialsam.experiments.store import Experiment, ExperimentStore
from spatialsam.pipeline.evaluation import EvaluationReport, evaluate_experiment
from spatialsam.ui.state import LAST_EXPERIMENT_KEY, get_store


def select_experiment(store: ExperimentStore) -> Experiment | None:
    ids = store.list_ids()
    if not ids:
        st.info("実験がありません。Video Segmentation で推論してください")
        return None
    last = st.session_state.get(LAST_EXPERIMENT_KEY)
    index = ids.index(last) if last in ids else 0
    return store.open(st.selectbox("実験", ids, index=index))


def load_ground_truth() -> dict | None:
    st.caption("GT は PNG 連番（ファイル名末尾の数字 = 元動画のフレーム番号、画素値 > 0 が前景）")
    uploaded = st.file_uploader("GT マスク（PNG を含む zip）", type=["zip"])
    if uploaded is not None:
        return load_gt_zip(uploaded.getvalue())
    directory = st.text_input("または GT ディレクトリのパス")
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


def main() -> None:
    st.set_page_config(page_title="Evaluation", layout="wide")
    st.title("Evaluation & Experiments")
    store = get_store()
    experiment = select_experiment(store)
    if experiment is None:
        return
    gt_masks = load_ground_truth()
    if st.button("評価を実行", type="primary", disabled=not gt_masks):
        try:
            show_report(evaluate_experiment(experiment, gt_masks))
        except ValueError as error:
            st.error(str(error))
    st.subheader("実験一覧")
    table = experiments_table(store)
    st.dataframe(table)
    st.download_button("experiments.csv", table.to_csv(index=False), "experiments.csv", "text/csv")


main()
