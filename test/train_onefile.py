import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from itertools import product
import os

# ========== 設定 ========== #
DATA_PATH = "C:\zebrafish_behavior_analysis\data\processed\zebrafish_features_processed_sample.csv"  # CSVファイルのパス
WINDOW_SIZE = 30
STEP_SIZE = 5
N_SPLITS = 5
REPEAT_NUM = 3

# ハイパーパラメータグリッド（必要に応じて調整）
param_grid = {
    'n_estimators': [100],
    'max_depth': [5, 10],
    'gamma': [0.0, 0.2],
    'objective': ['multi:softmax'],
    'eval_metric': ['mlogloss']
}

# ========== データ読み込みと前処理 ========== #
df = pd.read_csv(DATA_PATH)
# データの確認
X_raw = df.drop(columns=['label'])  # 特徴量（例：重心角度、腹部角度など）
y_raw = df['label'].astype(int)     # ラベル列

def slice_windows(X, y, window_size, step_size):
    Xs, ys = [], []
    for i in range(0, len(X) - window_size + 1, step_size):
        X_window = X.iloc[i:i + window_size].values
        y_window = y.iloc[i:i + window_size].values
        if len(np.unique(y_window)) == 1:
            Xs.append(X_window.flatten())
            ys.append(y_window[0])
    return np.array(Xs), np.array(ys)

X_windowed, y_windowed = slice_windows(X_raw, y_raw, WINDOW_SIZE, STEP_SIZE)
unique_labels = np.unique(y_windowed)
print("ウィンドウ後のラベル:", unique_labels)


# ========== train/test 分割 ========== #
X_train, X_test, y_train, y_test = train_test_split(
    X_windowed, y_windowed, test_size=0.3, stratify=y_windowed, random_state=42
)

# ========== 標準化（XGBoost不要なら省略可能） ========== #
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========== ハイパーパラメータ探索 ========== #
param_combinations = list(product(*param_grid.values()))
param_keys = list(param_grid.keys())

for trial_num, values in enumerate(param_combinations, 1):
    params = dict(zip(param_keys, values))
    print(f"\n=== Trial {trial_num}: {params} ===")

    f1_scores = []
    acc_scores = []

    for repeat in range(REPEAT_NUM):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            model = xgb.XGBClassifier(**params)
            model.fit(X_train[train_idx], y_train[train_idx])
            y_pred = model.predict(X_train[val_idx])
            f1 = f1_score(y_train[val_idx], y_pred, average='macro')
            acc = accuracy_score(y_train[val_idx], y_pred)
            f1_scores.append(f1)
            acc_scores.append(acc)
            print(f"Repeat {repeat+1}, Fold {fold_idx} — F1: {f1:.4f}, Acc: {acc:.4f}")

    print(f"[CV Avg] F1: {np.mean(f1_scores):.4f}, Acc: {np.mean(acc_scores):.4f}")

    # ========== テスト評価 ========== #
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    f1_test = f1_score(y_test, y_pred_test, average='macro')
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"[Test] F1: {f1_test:.4f}, Acc: {acc_test:.4f}")
