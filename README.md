# 2026_yoshida: 発火予算付きLIF画像復元

共有設計 https://chatgpt.com/s/t_6afbc84a52f08191963cd37f387e2724 に対応。
元のLISTA・以前のLIFも選択可能。GitHubの元コードに対する主な変更は
`src/model.py`, `src/utils.py`, 学習2ファイル, `src/test.py`, 設定2ファイル。
追加実装は `src/snn.py`, `src/evaluate_events.py`。検証を `tests/test_budget_lif.py` に収録。

## 実行

既存のPyTorch環境があればその仮想環境を利用する。プロジェクト直下で実行。

```powershell
python src/train_ML_CSC.py --config config_ml_csr.json
```

最後に表示される保存先を `$model` に指定する（以下の日時は置き換える）。

```powershell
$model = "models/学習後に表示されたフォルダ名_CSR"
python src/test.py --model-dir $model --mode dense --scan-limit 3 --samples 3
python src/evaluate_events.py --checkpoint "$model/model_csr.pth" --config "$model/config.json" --task csr --batch-size 1 --batches 1
python src/evaluate_events.py --checkpoint "$model/model_csr.pth" --config "$model/config.json" --task csr --batch-size 1 --batches 100 --output "$model/event_evaluation.json"
python src/test.py --model-dir $model --mode event --scan-limit 3 --samples 3
```

分類学習は `python src/train_ML_CSC_Image_Classifier.py --config config.json`。
分類の比較には保存先の `model.pth` と `--task classification` を指定する。
CSRはSGD、分類はAdam。どちらも代理勾配+BPTT。

## 計算と設定

`model: "lif_budget"` を選ぶ。3層、各層 `T: 3` 時刻を既定にした。
`lif.event_budget: 1000` は **1画像・1層・全時刻合計** の採用発火数。
同じ場所の再発火と負の発火もそれぞれ1回として数える。
`beta=0.9, theta=1.0, delta=0.1` は固定設定。学習対象は各層の
`Wy, Wz, dictionary, input_gain`。分類では分類器も学習する。
これらの設定値は出発点であり、復元精度を保証する調整済みの値ではない。

```
b = conv2d(y, Wy)
r = b + conv2d(z, Wz) - z
u = beta*v + input_gain*r
candidate = H(u-theta) - H(-u-theta)
s = 候補から画像ごとの残り予算内で選択
z = z + delta*s
v = u - theta*s
```

選択優先度は `abs(u)-theta`。同点はコードの平坦化インデックス順。
残り予算が尽きた画像は以後発火しない。バッチ全体が尽きると早期終了する。
初期値は `Wy=D` のコピー、`Wz=0`, `input_gain=0.8`。
`Wz` は学習する3×3畳み込みで、以前の `I-D0^T D0+B` とは別物。
画像ごとにコード、膜電位、信号、発火予算をリセットする。

学習では通常の畳み込みでrを再計算する。推論のeventモードでは
`r += event_conv2d(delta_z, Wz) - delta_z` と差分更新する。
発火しなかった場所のコードは保持する。採用した発火だけ膜電位を減算リセット。
最終復元は通常の転置畳み込み。入力変換も通常の畳み込み。

逆伝播は±theta周辺の三角形代理勾配（幅2.0）。ゼロ膜電位でも勾配が通る幅に設定する。予算に余裕がある画像は閾値未満の
場所にも代理勾配を通す。候補数が予算を超えた場合は選択位置だけに通し、
予算ゼロでは通さない。ランキング自体は微分しない。時間方向の状態はdetachしない。

CSRの既定損失は最終層から入力画像へ復元したMSEだけ。
lambda_sparse/group/spike/dictは0、BudgetLIFLayerにはeta/lambdaや近接写像を使わない。
分類は元のCE + lambda_recon×画像要素数×復元MSEを維持（正則化係数は0）。
入力画像への人工ノイズ追加は行わない。

## 互換性と評価

新方式は**再学習が必要**。旧LIFのcheckpointへ新configを組み合わせないこと。
`test.py` は保存configとcheckpointから `lista`, `lif`, `lif_budget` を区別する。
元のLISTAは設定のmodel省略または `lista`、旧LIFは `lif`。
既存の辞書図、段階ごとの復元図、分類への寄与の可視化を残している。
LIFの辞書削減APIは対応外として明示的にエラーにする。

比較評価は同一checkpoint・同一画像にdense/eventを適用し、MSE、PSNR、分類精度、
発火率、実行時間、コード差、復元差、再帰畳み込みのMAC推定をJSON出力する。
既定は1画像/バッチ、ウォームアップなし。`--warmup 1` で両方式を1回準備実行できる。
処理開始とバッチごとの進捗を表示する。実行時間はウォームアップ回数とともに解釈する。

イベント実装は非ゼロの差分コードから有効な出力位置へ加算するPyTorch参照実装。
Pythonループ・探索・散布加算のため**MACが減っても実時間が増える場合がある**。
比較MACは再帰Wz畳み込みだけ。入力変換、復元、LIF全マップ計算、予算選択、
メモリ転送は含まない。dense側はpaddingを含む標準推定、event側は有効寄与数。
浮動小数点の加算順で閾値・順位付近の発火が違うことがある。
電力削減や元のLISTAより高精度・高速であることを実測したものではない。

## 検証

```powershell
python -m unittest discover -s tests -v
```

合成入力による数値検証であり、MNIST等での学習精度検証は別途必要。

## 発火予算の超過記録

`budget_exceeded_steps` は、発火候補数が残り予算を超えた画像×層×時刻の件数。
`budget_rejected_candidates` は、その際に不採用になった候補の総数。
予算と候補数が等しい場合は超過に数えない。予算切れで停止した画像の
その後の仮想的な発火は計算・計上しない。実際の採用発火は予算以下に制限される。

学習時は `training_log.json` の `train_budget_counts` にエポック別・層別の
合計を保存。評価時は `event_evaluation.json` のdense/event各metricsと、
`test.py` の `evaluation_summary.json` に画像・層を合計した件数を保存する。
各層の `last_stats` には直近バッチの画像別件数も保持する。
