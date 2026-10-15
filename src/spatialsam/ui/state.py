"""Streamlit ページ間で共有するリソース（設定・ストア・モデルレジストリ）。"""

from __future__ import annotations

import streamlit as st

from spatialsam.config import AppConfig, load_config
from spatialsam.experiments.store import ExperimentStore
from spatialsam.models.gpu import GpuStatus, query_gpu_status
from spatialsam.models.registry import ModelRegistry

LAST_EXPERIMENT_KEY = "last_experiment_id"
SELECTED_VIDEO_KEY = "selected_video"  # 他ページから Video Segmentation に渡す動画のパス


@st.cache_resource
def get_config() -> AppConfig:
    return load_config()


@st.cache_resource
def get_registry() -> ModelRegistry:
    return ModelRegistry()


def get_store() -> ExperimentStore:
    return ExperimentStore(get_config().outputs_dir)


def show_gpu_status(status: GpuStatus | None = None) -> GpuStatus:
    status = status or query_gpu_status()
    if status.available:
        st.success(f"GPU: {status.name} / 空き {status.free_mib:,.0f} MiB / 全体 {status.total_mib:,.0f} MiB")
    else:
        st.warning(f"GPU 推論は利用できません: {status.reason}（fake セグメンタのみ実行可）")
    return status
