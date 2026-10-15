# TODO

設計は [spatialsam_design.md](spatialsam_design.md) を参照。完了したら `[x]` にする。

## Phase 0 — 環境・前提

- [x] git リポジトリ初期化、`.gitignore`
- [x] `uv` プロジェクト作成（Python 3.12、core / `gpu` extra / dev group）
- [ ] `facebook/sam3.1` のアクセス申請と承認（2026-10-15時点で403、未承認）
- [ ] `uv sync --extra gpu` で torch 2.10 (cu128) + sam3 を導入
- [ ] `hf download facebook/sam3.1 --revision daa63191845a41281374e725f4c9e51c7a824460`
- [ ] RTX A1000 (8GB) で SAM 3.1 の推論検証（Peak VRAM・速度、`use_fa3=False`）
- [ ] 8GB に載らない場合は `facebook/sam3` → Grounding DINO + SAM 2.1 アダプタを実装

## Phase 1 — Video Segmentation & Evaluation Studio（MVP）

### ドメイン・基盤
- [x] セグメンテーション結果のデータ型（`FrameMasks` / `SegmentationResult`）
- [x] 設定ファイル `configs/model.yaml` と読み込み
- [x] GPU状態取得（デバイス名・空きVRAM）と Peak VRAM 計測
- [x] Model Registry（GPU上に同時に1モデルだけ保持・解放）

### 動画
- [x] 動画メタ情報の取得、フレーム読み出し（間引き・最大フレーム数・リサイズ）
- [x] フレーム列の MP4 書き出し
- [x] マスク重畳（物体IDごとの色、alphaブレンド）

### モデルアダプタ
- [x] `VideoSegmenter` インターフェース
- [x] `FakeSegmenter`（GPU・重み無しでUI/パイプラインを動かすためのダミー）
- [x] `Sam31Segmenter`（`build_sam3_multiplex_video_predictor` を遅延import）
- [ ] `Sam31Segmenter` を実機で動作確認（重み承認待ち）
- [ ] bf16 autocast・チャンク分割・OOM時の自動リトライ

### 評価
- [x] IoU / Dice
- [x] 輪郭 F-measure、J&F
- [x] GTマスク（PNG連番）の読み込みとフレーム対応付け
- [x] フレーム別評価と集約（CSV/JSON）

### 実験ストア
- [x] 実験ディレクトリ作成（`outputs/<experiment_id>/`）
- [x] `metadata.json`（モデル・revision・プロンプト・設定・性能）
- [x] 予測マスク保存（フレームごとの `.npz`）、`tracks.json`、`overlay.mp4`
- [x] `evaluation/metrics.json`, `per_frame.csv`

### パイプライン・UI
- [x] 推論パイプライン（入力 → 推論 → 重畳 → 保存）
- [x] 評価パイプライン（実験 + GT → メトリクス保存）
- [x] Streamlit トップページ（GPU状態表示）
- [x] Tab 1: Video Segmentation（アップロード・プロンプト・推論設定・結果表示）
- [x] Tab 4: Evaluation（GT取り込み・メトリクス表示・エクスポート）
- [ ] GT付きサンプル動画の用意（DAVIS 2017 等）と `samples/README.md`
- [ ] 実機SAM 3.1で入力→推論→可視化の一連動作確認

### テスト
- [x] pytest で各モジュールの単体テスト

## Phase 2 — Point Cloud / 3DGS Rendering

- [ ] gsplat 導入（PyTorch/CUDAとの整合確認）
- [ ] PLY/PCD・既存3DGSシーンのインポート
- [ ] カメラ軌道設定と RGB/Depth/Pose 付きフレーム生成、MP4化
- [ ] 生成動画を Phase 1 パイプラインに投入
- [ ] 実画像 vs レンダリング画像の比較（PSNR/SSIM/LPIPS、IoU）

## Phase 3 — Text-to-3D Segmentation

- [ ] 2D Mask Lifting（深度・カメラ姿勢による逆投影）
- [ ] 複数視点の統合と整合性チェック
- [ ] 3D点群／Gaussian のハイライト表示
- [ ] 3D IoU 等の評価
