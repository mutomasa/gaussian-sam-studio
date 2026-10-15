# SpatialSAM Studio

オープンウェイトの SAM 3.1（[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1)）をローカル GPU で動かし、
テキスト指定の動画セグメンテーションと精度評価を行う Streamlit アプリ。将来は 3DGS と統合する。

- 設計: [docs/spatialsam_design.md](docs/spatialsam_design.md)
- 進捗: [docs/todo.md](docs/todo.md)

## セットアップ

```bash
uv sync                    # UI・評価・fake セグメンタ（GPU 不要）
uv sync --extra gpu        # torch 2.10 (cu128) + sam3（ローカル GPU 推論）

# SAM 3.1 の重み（HF でアクセス承認後）
hf auth login
hf download facebook/sam3.1 --revision daa63191845a41281374e725f4c9e51c7a824460
```

## 起動

```bash
uv run streamlit run app.py
```

GPU や重みが無い環境でも、セグメンタに `fake` を選べばパイプラインと UI を確認できる（結果はダミー）。

## テスト

```bash
uv run pytest
```

## 構成

```text
app.py                      トップページ（GPU 状態）
pages/                      1: Video Segmentation, 4: Evaluation
configs/model.yaml          モデル設定（repo / revision / 推論パラメータ）
src/spatialsam/
  models/                   データ型・セグメンタ（SAM 3.1 / fake）・GPU・レジストリ
  video/                    動画 I/O・重畳・アップロード検査
  evaluation/               IoU / Dice / 輪郭F / J&F、GT 読み込み
  experiments/              実験ストア（outputs/<experiment_id>/）
  pipeline/                 推論・評価の一連の処理
  ui/                       Streamlit 共有状態
tests/                      pytest
```
