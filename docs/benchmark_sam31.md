# SAM 3.1 実機ベンチマーク（RTX A1000 8GB）

- 日付: 2026-10-16
- モデル: `facebook/sam3.1` @ `daa63191845a41281374e725f4c9e51c7a824460`（`sam3.1_multiplex.pt`）
- コード: `facebookresearch/sam3` @ `7f473cfd983d5eb43040ff608f66bff0d7bdcb53`、torch 2.10.0+cu128
- 設定: `configs/model.yaml` の既定（`use_fa3=false`、画像エンコーダのバッチ 1、フレームの CPU オフロード）
- 入力: DAVIS 2017 val 480p（854x480）の全フレーム、テキストプロンプトは `configs/samples.yaml`
- 評価: 全予測マスクの和集合 vs GT 前景（J = IoU、F = 輪郭 F、J&F = 平均）
- 再現: `HF_HUB_OFFLINE=1 uv run python scripts/benchmark_samples.py [--fp32-backbone]`

## 結果（既定: backbone bf16）

| sequence | prompt | frames | J | F | J&F | Peak VRAM [MiB] | 時間 [s] |
|---|---|---:|---:|---:|---:|---:|---:|
| blackswan | swan | 50 | 0.958 | 0.987 | 0.972 | 4,220 | 55.1* |
| cows | cow | 104 | 0.954 | 0.950 | 0.952 | 4,672 | 82.3 |
| dog | dog | 60 | 0.961 | 0.988 | 0.974 | 4,302 | 48.1 |
| goat | goat | 90 | 0.925 | 0.970 | 0.947 | 4,551 | 71.7 |
| libby | dog | 49 | 0.906 | 0.987 | 0.946 | 4,211 | 39.6 |
| parkour | person | 100 | 0.947 | 0.988 | 0.967 | 4,634 | 79.5 |
| camel | camel | 90 | 0.845 | 0.873 | 0.859 | 4,612 | 72.0 |

\* 最初のサンプルはモデル構築（約 15 秒）を含む。推論は約 0.8 秒/フレーム。

- 単一物体の 6 本は J&F 0.95〜0.97。
- camel は GT 対象外の 2 頭目を 10 フレーム目から検出して追跡するため低下する（狙いどおり、テキスト指定では
  「その種類の物体すべて」が対象になる）。
- Peak VRAM はフレーム数にほぼ比例して増え、100 フレームで約 4.7GB。

## backbone bf16 vs fp32

fp32 では 90〜104 フレームの 4 本が OOM し、自動リトライで半分のフレーム数に縮小された（Peak 約 7.0GB）。
同じフレームで両者の予測マスクを比べると：

| sequence | 比較フレーム | 予測マスク IoU（平均） | 最小 |
|---|---:|---:|---:|
| blackswan | 50 | 0.9997 | 0.9995 |
| cows | 52 | 0.9996 | 0.9993 |
| dog | 60 | 0.9996 | 0.9990 |
| goat | 45 | 0.9993 | 0.9988 |
| libby | 49 | 0.9988 | 0.9824 |
| parkour | 50 | 0.9983 | 0.9793 |
| camel | 45 | 0.9996 | 0.9984 |

bf16 化による精度差は実質ゼロで、VRAM は約 2.3〜2.8GB 減る。**8GB GPU では bf16 backbone を既定とする**。
