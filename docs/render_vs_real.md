# 実画像 vs 3D 描画画像の比較（#16）

- 日付: 2026-10-16
- シーン: Mip-NeRF 360 bonsai（images_4 = 780x520）
- 評価用視点: 画像名順で 8 枚に 1 枚の 37 視点（3DGS の学習に使っていない新規視点）
- 比べたもの
  - `real`: 実画像
  - `3dgs`: gsplat（MCMC、7000 step、Gaussian 100 万個）で学習した 3DGS の描画
  - `points`: COLMAP 点群（20.6 万点）を学習せずに Gaussian にした描画
- SAM 3.1: 各ソースを同じ順の動画（8fps）にして、同じテキストプロンプトで推論した。bonsai には正解マスクが無いので、実画像での結果を基準にした。
- 再現:

  ```bash
  uv run python scripts/train_3dgs.py --sparse <bonsai>/sparse/0 --images <bonsai>/images_4 --steps 7000
  uv run python scripts/compare_render_vs_real.py --sparse <bonsai>/sparse/0 --images <bonsai>/images_4 \
      --source 3dgs=outputs/training/<job_id>/point_cloud.ply --points \
      --prompt bonsai --prompt table --prompt bicycle
  ```

  結果は Evaluation ページの「実画像 vs 3D 描画」タブで、視点ごとに 3 ソースのマスクを並べて見られる。

## 画質（実画像との比較）

> 2026-10-16 以降は TorchMetrics で計算している（#41）。PSNR と LPIPS は下の表と同じ値で、SSIM は画像の端の扱いの違いで 3DGS が 0.943 → 0.942、点群が 0.142 → 0.141 になる。

| ソース | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| 3dgs | 30.73 | 0.943 | 0.074 |
| points | 10.49 | 0.142 | 0.727 |

## SAM 3.1 の結果

- 検出率: マスクが出た視点の割合
- IoU: 実画像のマスクとの IoU の平均。どちらにもマスクが無い視点は除く

| プロンプト | ソース | 検出率 | IoU（vs 実画像） |
|---|---|---:|---:|
| bonsai | real | 0.97 | — |
| bonsai | 3dgs | 0.92 | 0.92 |
| bonsai | points | 0.00 | 0.00 |
| table | real | 1.00 | — |
| table | 3dgs | 1.00 | 0.98 |
| table | points | 0.76 | 0.57 |
| bicycle | real | 0.46 | — |
| bicycle | 3dgs | 0.46 | 0.97 |
| bicycle | points | 0.00 | 0.00 |

## わかったこと

- **学習した 3DGS なら、新規視点の描画でも SAM 3.1 は実画像とほぼ同じ結果になる。**
  - table は IoU 0.98、bicycle は 0.97 で、検出率も実画像と同じだった。
  - bonsai の IoU は 0.92 だが、差は視点 1・2 の 2 枚だけ。この 2 枚はマスクが無く IoU 0 になっている。残りの視点は 0.96 前後だった。
- **bonsai の差は、画質ではなく追跡が始まるタイミングの違い。**
  - 実画像も 3DGS も、最初の視点では検出しなかった。
  - 実画像は視点 1 から、3DGS は視点 3 から物体を追跡し始めた。
  - 検出された物体のスコアは 0.72 と 0.71 でほぼ同じ。
  - 評価用視点は 8 枚おきなので、視点が大きく飛ぶ。このため、追跡が始まるタイミングの差がそのまま検出率に出る。
- **学習しない点群の描画では、SAM はほとんど使えない。**
  - 細かい構造の bonsai と bicycle は一度も検出されなかった。
  - 面積の大きい table だけは 76% の視点で検出されたが、IoU は 0.57 にとどまった。
  - 画質指標（PSNR 10.5、LPIPS 0.73）の低さと合っている。

## 注意

- 基準は「実画像での SAM の結果」で、正解マスクではない。実画像でも見落とす物体（bicycle は 46%）については、一致度しか測れない。
- 視点の順序と間隔で追跡の結果が変わる。連続した撮影順（全画像）で比べると、検出率の差はもっと小さくなると考えられる。ただし、その場合は学習に使った視点を含むことになる。
