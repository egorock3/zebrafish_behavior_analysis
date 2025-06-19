├── LICENSE
├── Makefile                 <- makeコマンドで操作するためのファイル（例: make train, make data）
├── README.md                <- プロジェクトのトップレベル README
├── data
│   ├── external             <- 外部提供の生データ
│   ├── interim              <- 中間データ（前処理途中）
│   ├── processed            <- モデルに使う最終的なデータ
│   └── raw                  <- 変換前の元データ（変更しない）
│
├── docs                     <- Sphinx ドキュメントプロジェクト
│
├── models                   <- 学習済みモデル・予測結果・モデルのサマリーなど
│
├── notebooks                <- Jupyter Notebookファイル（命名例: `1.0-jqp-initial-data-exploration`）
│
├── references               <- 仕様書、マニュアル、データ辞書など
│
├── reports
│   └── figures              <- グラフ・図などの出力物
│
├── requirements.txt         <- pipで環境を再現するためのパッケージリスト
├── setup.py                 <- `pip install -e .` によって src をモジュールとして読み込ませるための設定
│
├── src
│   ├── __init__.py
│   ├── train.py             <- モデル切り替え可能なメインの学習スクリプト
│   │
│   ├── data
│   │   └── make_dataset.py  <- データ取得・前処理など
│   │
│   ├── models               <- モデル定義・切り替え可能な構造
│   │   ├── base             <- 抽象基底クラスと共通前処理
│   │   │   ├── __init__.py
│   │   │   ├── model.py     <- `BaseModel`クラスなど
│   │   │   └── preprocess.py
│   │   │
│   │   ├── lightgbm         <- LightGBM用モデル・前処理ロジック
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── preprocess.py
│   │   │
│   │   └── cnn              <- CNN（例: Keras/TensorFlow）用ロジック
│   │       ├── __init__.py
│   │       ├── model.py
│   │       └── preprocess.py
│   │
│   └── visualization
│       └── visualize.py     <- 可視化・分析用スクリプト
│
└── tox.ini                  <- テストやLintの設定（tox用）
