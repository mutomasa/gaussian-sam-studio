# TODO

設計は [spatialsam_design.md](spatialsam_design.md) を参照。完了したら `[x]` にする。

運用：作業ごとにブランチを切って PR を出す。問題点・未完了の作業は GitHub issue（`#N`）で管理し、PR 本文の `Closes #N` で閉じる。

## Phase 0 — 環境・前提

- [x] git リポジトリ初期化、`.gitignore`
- [x] `uv` プロジェクト作成（Python 3.12、core / `gpu` extra / dev group）
- [x] `facebook/sam3.1` のアクセス申請と承認（2026-10-16 承認。fine-grained トークンは gated repo 読み取り権限が必要） #1
- [x] `uv sync` で torch 2.10 + sam3 を導入 #2（不足依存 #6、上流バグ回避 #7）。gsplat の JIT ビルドのため cu128 → cu130 に変更 #13
- [x] `hf download facebook/sam3.1 --revision daa63191845a41281374e725f4c9e51c7a824460` #1
- [x] RTX A1000 (8GB) で SAM 3.1 の推論検証（Peak VRAM・速度、`use_fa3=False`）→ [docs/benchmark_sam31.md](benchmark_sam31.md) #3
- [x] ~~8GB に載らない場合のフォールバック実装~~ → bf16 backbone で 100 フレーム 4.7GB に収まるため不要 #3

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
- [x] `Sam31Segmenter` を実機で動作確認 #3
- [x] 画像エンコーダのバッチサイズを 16 → 1 に変更（8GB で OOM するため）#2
- [x] backbone の bf16 化・フレームの CPU オフロード・OOM時の自動リトライ #4
- [ ] 長尺動画のチャンク分割（チャンク間の物体ID継続）#9

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
- [x] GT付きサンプル動画の用意（DAVIS 2017 val の 7 シーケンス）と `samples/README.md` #5
- [x] 実機SAM 3.1で入力→推論→可視化の一連動作確認（`scripts/benchmark_samples.py`） #3

### テスト
- [x] pytest で各モジュールの単体テスト

## Phase 2 — Point Cloud / 3DGS Rendering

- [x] gsplat 導入（PyTorch/CUDAとの整合確認）#13
- [x] PLY・既存3DGSシーン・COLMAP のインポート #14（PCD は未対応）
- [x] カメラ軌道設定と RGB/Depth/Pose 付きフレーム生成、MP4化 #15
- [x] 生成動画を Phase 1 パイプラインに投入 #15
- [x] gsplat による 3DGS 学習ジョブ（8GB 向け）#18
- [x] 実画像 vs レンダリング画像の比較（PSNR/SSIM/LPIPS、IoU）#16 → [render_vs_real.md](render_vs_real.md)

## Phase 3 — Text-to-3D Segmentation

- [ ] 2D Mask Lifting（深度・カメラ姿勢による逆投影）
- [ ] 複数視点の統合と整合性チェック
- [ ] 3D点群／Gaussian のハイライト表示
- [ ] 3D IoU 等の評価
