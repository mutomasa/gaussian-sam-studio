# SpatialSAM Studio

オープンウェイトの SAM 3.1（[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1)）をローカル GPU で動かし、
テキスト指定の動画セグメンテーションと精度評価を行う Streamlit アプリ。将来は 3DGS と統合する。

- 設計: [docs/spatialsam_design.md](docs/spatialsam_design.md)
- 進捗: [docs/todo.md](docs/todo.md)

## セットアップ

```bash
uv sync                    # torch 2.10 (cu130) + sam3 + gsplat を含む全依存（既定で dev / gpu グループ）
uv sync --no-group gpu     # GPU 無し環境（UI・評価・fake セグメンタのみ）

# SAM 3.1 の重み（HF でアクセス承認後）
hf auth login
hf download facebook/sam3.1 --revision daa63191845a41281374e725f4c9e51c7a824460
```

重みは `configs/model.yaml` の `sam31.source.revision` で固定したものを HF キャッシュから読む
（`HF_HUB_OFFLINE=1` でも動く）。別の場所の重みを使うときは `sam31.checkpoint_path` を指定する。
fine-grained トークンの場合は「public gated repos への読み取り」権限が必要。

gsplat は torch 2.10 向けの prebuilt wheel が無いため、初回 import 時に CUDA カーネルを JIT ビルドする
（A1000 で約 2 分、`~/.cache/spatialsam/torch_extensions` にキャッシュ）。nvcc は venv 内の pip 版を
`spatialsam.spatial.cuda_env` が自動で使うので、システムへの CUDA ツールチェーンの導入は不要
（NVIDIA ドライバは CUDA 13.0 以上に対応していること）。

## サンプル動画（GT 付き）

```bash
uv run python scripts/prepare_samples.py   # DAVIS 2017 から samples/ に生成（CC BY-NC 4.0）
```

詳細は [samples/README.md](samples/README.md)。全サンプルを一括で推論・評価するには:

```bash
uv run python scripts/benchmark_samples.py                 # → outputs/benchmarks/<日時>-sam3.1.csv
uv run python scripts/benchmark_samples.py --fp32-backbone  # bf16 化の精度影響の比較用
```

## 起動

```bash
uv run streamlit run app.py
```

GPU や重みが無い環境でも、セグメンタに `fake` を選べばパイプラインと UI を確認できる（結果はダミー）。

**3D Rendering** ページでは、3DGS / 点群の PLY や COLMAP の sparse ディレクトリを指定し、オービットまたは
撮影カメラの軌道に沿って描画する。結果（RGB / 深度 / 姿勢 / MP4）は `outputs/renders/<render_id>/` に保存され、
「この動画を Video Segmentation で使う」で SAM 3.1 の入力にできる。

同じページの「3DGS 学習」タブ、または CLI で、COLMAP（画像 + sparse）から 3DGS を学習できる
（gsplat の MCMC 戦略。学習は別プロセスで動き、結果は `outputs/training/<job_id>/point_cloud.ply`）。

```bash
uv run python scripts/train_3dgs.py --sparse <scene>/sparse/0 --images <scene>/images_4 [--steps 7000] [--background]
```

学習に使っていない視点で、実画像と 3D 描画を画質（PSNR / SSIM / LPIPS）と SAM 3.1 の結果で比べられる
（結果は Evaluation ページの「実画像 vs 3D 描画」タブ。bonsai での結果は [docs/render_vs_real.md](docs/render_vs_real.md)）。

```bash
uv run python scripts/compare_render_vs_real.py --sparse <scene>/sparse/0 --images <scene>/images_4 \
    --source 3dgs=outputs/training/<job_id>/point_cloud.ply --points --prompt bonsai --prompt table
```

## テスト

```bash
uv run pytest
```

## 構成

```text
app.py                      トップページ（GPU 状態）
pages/                      1: Video Segmentation, 2: 3D Rendering, 3: 3D Segmentation, 4: Evaluation
configs/model.yaml          モデル設定（repo / revision / 推論パラメータ）
src/spatialsam/
  models/                   データ型・セグメンタ（SAM 3.1 / fake）・GPU・レジストリ
  video/                    動画 I/O・重畳・アップロード検査
  evaluation/               IoU / Dice / 輪郭F / J&F、GT 読み込み
  experiments/              実験ストア（outputs/<experiment_id>/）
  pipeline/                 推論・評価の一連の処理
  ui/                       Streamlit 共有状態
  samples/                  サンプル動画の生成（DAVIS 2017）
  spatial/                  3DGS / 点群 / COLMAP の読み込み、gsplat 描画、カメラ軌道
  training/                 gsplat による 3DGS 学習とジョブ管理（別プロセス）
scripts/prepare_samples.py  サンプル動画と GT を samples/ に生成
scripts/benchmark_samples.py  サンプルの一括推論・評価（CSV）
scripts/train_3dgs.py       COLMAP から 3DGS を学習
scripts/compare_render_vs_real.py  実画像 vs 3D 描画（画質と SAM）の比較
scripts/lift_masks.py       SAM の 2D マスクを Gaussian に持ち上げる（3D セグメンテーション）
scripts/record_demo.py      bonsai の全体の流れをデモ動画に録る（Playwright、待ち時間は早送り）
configs/samples.yaml        サンプルのシーケンスとプロンプト例
tests/                      pytest
```
