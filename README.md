# 2026_yoshida-main
2026年度の卒業研究での使用コード

# CSC-LISTAとSNN近似の比較

```text
.venv/                Python仮想環境
models/               CSC-LISTA・SNN更新器の学習済み重み
dataset/              CIFAR-10とデータセット情報
teacher_training/     元のCSC-LISTAの学習コード
  json/               教師学習の設定と旧設定の控え
snn_approximation/    SNN更新器の学習・評価コード
  json/               教師の指定、SNN学習、評価の設定
result/               Matplotlibの描画コードと生成したPNG
  json/               比較の測定値
README.md
requirements.txt      すべてのPythonコードの依存パッケージ
```

以下はプロジェクト直下から実行します。

## 準備

このPCの `.venv` は、すでにインストール済みの親フォルダーの環境へのジャンクションです。この場所で引き続き利用できます。

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

別のPCでは、プロジェクトのファイルをコピーしてから Python 3.14 で新しい `.venv` を作り、同じコマンドで依存パッケージを入れてください。

## 1. 元のCSC-LISTAを学習

`teacher_training/json/config.json` で学習条件を設定します。既定はCIFAR-10、30エポックです。

```powershell
& '.\.venv\Scripts\python.exe' .\teacher_training\train_ML_CSC_Image_Classifier.py
```

重み、設定、学習ログ、精度と損失のグラフは `models/teacher_training/<実行日時>/` に保存されます。

## 2. SNN更新器の学習と比較

`snn_approximation/json/model_config.json` の `teacher_directory` に、教師の `model.pth` と `config.json` が入ったフォルダーを指定します。既定は既存の `models/CIFAR10_GROUP` です。既定の `approximate_layers: [3]` と `skipped_updates: 1` では、第3層の3回目だけをSNNで近似します。

- `snn_training_config.json` · SNN更新器の学習条件
- `evaluation_config.json` · 評価枚数とバッチサイズ。既定は1,024枚。全件評価は `--test-images 10000`

教師を変更した場合は、新しい教師に合わせてSNN更新器を学習します。

```powershell
& '.\.venv\Scripts\python.exe' .\snn_approximation\snn_lista_accelerator.py
```

既存の教師と対応する第3層専用の学習済みSNNで評価だけを行う場合は、重みを指定します。評価する層はチェックポイントから読み込みます。

```powershell
& '.\.venv\Scripts\python.exe' .\snn_approximation\snn_lista_accelerator.py --checkpoint .\models\SNN\snn_update.pth --skipped-updates 1
```

測定結果は `result/json/metrics.json`、`metrics2.json`、`metrics3.json` のように連番で保存します。新規学習したSNNの重みは、同じ番号の `models/SNN/snn_update.pth`、`snn_update2.pth`、`snn_update3.pth` に保存します。学習済み重みで評価だけ行う場合は測定結果を追加します。別の場所に保存するときは `--output <フォルダー>` を指定します。

## 3. 比較グラフの保存と再描画

評価が完了したら、保存されたJSONからグラフを作成するために次を実行します。

```powershell
& '.\.venv\Scripts\python.exe' .\result\plot_csc_lista_vs_snn.py
```

引数なしでは `result/json/metrics.json` を読み込みます。別の測定結果を描くときは、`--metrics` で使うJSONを指定します。PNGの保存先は `--output` で指定できます。同名のPNGがある場合は、`csc_lista_vs_snn2.png`、`csc_lista_vs_snn3.png` のように連番で保存します。指定した名前が数字で終わる場合は、`csc_lista_vs_snn2_2.png` のように連番を追加します。

```powershell
& '.\.venv\Scripts\python.exe' .\result\plot_csc_lista_vs_snn.py --metrics .\result\json\metrics2.json --output .\result\csc_lista_vs_snn2.png
```
