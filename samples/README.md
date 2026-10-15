# samples

GT 付きサンプル動画の置き場所。動画と GT はリポジトリに含めない（`.gitignore`）。

## 生成

```bash
uv run python scripts/prepare_samples.py
```

[DAVIS 2017](https://davischallenge.org/) TrainVal 480p（約 830MB、sha256 検証あり）を
`~/.cache/spatialsam/davis/` に取得し、`configs/samples.yaml` のシーケンスだけを取り出す。

- **ライセンス**：CC BY-NC 4.0（非商用のみ）。論文等で使う場合は DAVIS 2017 を引用する
- **動画**：`samples/<sequence>.mp4`（24fps、854x480、全フレーム）
- **GT**：`samples/gt/<sequence>/00000.png ...`（DAVIS のパレット PNG。画素値 = 物体ID、0 = 背景）。
  ファイル名の番号 = 動画のフレーム番号

## 収録シーケンス

| sequence | プロンプト例 | フレーム | 狙い |
|---|---|---|---|
| blackswan | swan | 50 | 単一物体・静的背景の基本ケース |
| cows | cow | 104 | 単一物体・大きな対象 |
| dog | dog | 60 | 姿勢変化 |
| goat | goat | 90 | 背景と色が近い |
| libby | dog | 49 | 木や柵による部分遮蔽 |
| parkour | person | 100 | 大きなカメラ移動・視点変化 |
| camel | camel | 90 | GT 対象外のラクダが背後に映る（比較用） |

評価は「全予測マスクの和集合 vs GT 前景」なので、GT 対象以外に同じ種類の物体が映ると
その分 IoU が下がる。camel 以外はそうした物体が映らないものを選んでいる。

## 使い方

1. Video Segmentation で「サンプル動画」から選ぶ（プロンプト例が自動入力される）
2. Evaluation で実験を選ぶと GT ディレクトリが自動入力される
