# C:\zebrafish_behavior_analysis\src\trainers\XGBoostOptimizer.py
import os
import sys
import time
import logging
import datetime
import argparse
import numpy as np
import pandas as pd

from itertools import product

# プロジェクトルートをsys.pathに追加して、他のモジュールをインポートできるようにする
# XGBoostOptimizer.py が src/trainers/ にある場合、プロジェクトルートはその2つ上の階層
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# kfold_trainer.py と result_saver.py を正しいパスからインポート
from src.trainers.kfold_trainer import KFoldTrainer # 同じsrc/trainers内なので相対インポートも可能だが、統一のため絶対パス
from src.utils.result_saver import SaveSummary
from src.utils.config_loader import load_config, get_project_root # config_loaderをインポート

class XGBoostOptimizer:
    def __init__(self):
        self.model_name = 'XGBoost'
        
        self.project_root = get_project_root()
        # experiment_config.yaml から設定を読み込む
        self.exp_config = load_config(os.path.join(self.project_root, 'config', 'experiment_config.yaml'))
        
        if not self.exp_config:
            raise ValueError("Failed to load experiment_config.yaml. Please check the path and content.")

        # 設定ファイルからパスや共通パラメータを取得
        self.base_data_path = os.path.join(self.project_root, self.exp_config['paths']['base_data_path'])
        self.base_results_path = os.path.join(self.project_root, self.exp_config['paths']['base_results_path'], self.model_name)
        self.k_fold_num = self.exp_config['general']['k_fold_num']
        self.repeat_num = self.exp_config['general']['repeat_num']
        self.target_label_column = self.exp_config['general']['target_label_column']

        # ロギング設定
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'xgboost_optimizer_{timestamp}.log'
        self.log_filepath = os.path.join(self.base_results_path, log_filename)
        os.makedirs(self.base_results_path, exist_ok=True) 

        # ロガーの初期化（ハンドラーをクリアして重複ログを防ぐ）
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=self.log_filepath, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
        self.log_print = logging.info
        self.log_print(f"Logging to: {self.log_filepath}")
        self.log_print(f"Loaded Experiment Config: {self.exp_config}")


    def run_hyperparameter_search(self):
        # 今回は、experiment_config.yamlのXGBoostセクションから直接ハイパーパラメータを取得し、固定値で動かす
        xgboost_hyperparams = {
            'n_estimators': self.exp_config['XGBoost']['n_estimators'],
            'max_depth': self.exp_config['XGBoost']['max_depth'],
            'max_leaves': self.exp_config['XGBoost']['max_leaves'],
            'gamma': self.exp_config['XGBoost']['gamma'],
            'objective': self.exp_config['XGBoost']['objective'],
            'eval_metric': self.exp_config['XGBoost']['eval_metric'],
            'seed': self.exp_config['XGBoost']['seed']
        }
        
        # training_settingsも設定から取得
        training_settings = {
            'early_stopping_rounds': self.exp_config['XGBoost']['early_stopping_rounds'],
            'verbose': self.exp_config['XGBoost']['verbose']
        }

        # 訓練するデータセットのリストをconfigから取得
        datasets_to_train = self.exp_config['datasets_to_train']

        total_trials_counter = 0 # 複数データセットの場合の試行数カウンター
        for dataset_config in datasets_to_train:
            total_trials_counter += 1 # 各データセットで1試行とカウント
            
            # データセットのファイル名とパラメータを取得
            dataset_filename = dataset_config['filename']
            window_size = dataset_config['time_window']
            step_size = dataset_config['step_size']
            
            use_dataset_path = os.path.join(self.base_data_path, dataset_filename)
            
            if not os.path.exists(use_dataset_path):
                self.log_print(f"Error: Dataset not found at {use_dataset_path}. Skipping.")
                continue

            self.log_print(f"\n--- Processing dataset: {dataset_filename} (TW:{window_size}, SS:{step_size}) ---")

            # 今回は単一のハイパーパラメータ組み合わせなので、trial_numberも1とする
            trial_number = 1 
            hyper = xgboost_hyperparams.copy() # ハイパーパラメータは固定値

            self.log_print("Generated hyper params for trial: %s", hyper)

            trial_results = {
                'trial_number': trial_number,
                'hyperparameters': hyper,
                'data_tw': window_size,
                'data_ss': step_size,
                'repeats': []
            }

            for repeat_idx in range(self.repeat_num):
                current_repeat_num = repeat_idx + 1
                self.log_print(f"Running repeat {current_repeat_num}/{self.repeat_num}...")
                
                kfold_trainer = KFoldTrainer(
                    model_name=self.model_name,
                    hyper_params=hyper,
                    window_size=window_size,
                    step_size=step_size,
                    k_cv_num=self.k_fold_num,
                    dataset_path=use_dataset_path,
                    save_folder=self.base_results_path,
                    trial_number=trial_number,
                    target_label_column=self.target_label_column,
                    log_print=self.log_print
                )

                start_time = time.time()
                f1_cv, f1_ts, accuracy_cv, accuracy_ts, current_trial_folder, best_epochs = kfold_trainer.train_and_evaluate(
                    param_name=f"{self.model_name}_{os.path.splitext(dataset_filename)[0]}", # 結果フォルダ名にファイル名を使用
                    repeat=current_repeat_num,
                    training_settings=training_settings
                )

                elapsed_time = time.time() - start_time
                elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
                self.log_print(f"Repeat {current_repeat_num} finished in {elapsed_time_str}, CV F1: {f1_cv:.4f}, Test F1: {f1_ts:.4f}")
                
                repeat_result = {
                    'repeat_number': current_repeat_num,
                    'accuracy_cv': accuracy_cv,
                    'accuracy_ts': accuracy_ts,
                    'f1_score_cv': f1_cv,
                    'f1_score_ts': f1_ts,
                    'best_epochs': best_epochs,
                    'elapsed_time': elapsed_time
                }
                trial_results['repeats'].append(repeat_result)

            # 全リピート終了後、この試行の結果を保存
            sv = SaveSummary(trial_results, current_trial_folder, self.base_results_path, self.model_name)
            sv.save_results()

            avg_f1_cv_for_trial = np.mean([r['f1_score_cv'] for r in trial_results['repeats']])
            self.log_print(f"\nTrial {trial_number} Summary (Data: {dataset_filename}):")
            self.log_print(f"Average CV F1 Score: {avg_f1_cv_for_trial:.4f}")
            self.log_print(f"Best CV F1 Score (across repeats): {max(r['f1_score_cv'] for r in trial_results['repeats']):.4f}")
            self.log_print(f"Current hyperparameters: {hyper}")


def main():
    optimizer = XGBoostOptimizer()
    optimizer.run_hyperparameter_search()

if __name__ == "__main__":
    main()