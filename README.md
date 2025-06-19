## 📁 ディレクトリ構成


```
├── LICENSE
├── Makefile                 # make train, make data などのコマンドを定義
├── README.md                # プロジェクトのトップレベル説明書
├── data
│   ├── external             # 外部提供の生データ
│   ├── interim              # 前処理途中の中間データ
│   ├── processed            # モデル学習・予測に使う最終データ
│   └── raw                  # 元の生データ（編集しない）
├── docs                     # Sphinxドキュメント
├── models                   # 学習済みモデル・結果など
├── notebooks                # Jupyter Notebookファイル
├── references               # 仕様書、マニュアル、データ辞書など
├── reports
│   └── figures              # 図表・グラフなど
├── requirements.txt         # pipで使う依存パッケージリスト
├── setup.py                 # `pip install -e .` のための設定ファイル
├── src                      # ソースコード本体
│   ├── __init__.py
│   ├── train.py             # モデル学習スクリプト
│   ├── data
│   │   └── make_dataset.py  # データ取得・前処理スクリプト
│   ├── models               # モデル定義と切り替え可能な構造
│   │   ├── base             # 基底モデル・共通前処理
│   │   ├── lightgbm         # LightGBMモデル関連
│   │   └── cnn              # CNNモデル関連（Kerasなど）
│   └── visualization
│       └── visualize.py     # 可視化スクリプト
└── tox.ini                  # テストやLintの設定
```