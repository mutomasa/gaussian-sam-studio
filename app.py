"""SpatialSAM Studio のトップページ。`uv run streamlit run app.py` で起動する。"""

import streamlit as st

from spatialsam.ui.state import get_config, get_registry, show_gpu_status

st.set_page_config(page_title="SpatialSAM Studio", layout="wide")
st.title("SpatialSAM Studio")
st.caption("オープンウェイトの SAM 3.1 をローカル GPU で動かす、テキスト指定の動画セグメンテーションと評価")

show_gpu_status()

config = get_config()
st.markdown(
    f"""
- **モデル**: `{config.sam31_source.repo_id}` @ `{config.sam31_source.revision[:12]}`
- **既定のセグメンタ**: `{config.default_segmenter}`
- **GPU 上のモデル**: `{get_registry().loaded_key or "なし"}`
- **実験の保存先**: `{config.outputs_dir}`

左のサイドバーからページを選んでください。

1. **Video Segmentation** — 動画とプロンプトから推論し、重畳動画を確認する
2. **3D Rendering** — 3DGS / 点群 / COLMAP のシーンをカメラ軌道に沿って描画し、RGB・深度・姿勢付きの動画を作る
3. **Evaluation** — GT マスクと比較して IoU / Dice / J&F を計算する
"""
)
