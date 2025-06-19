import os
import sys
import time
import yaml
import logging
import argparse
import datetime
import pandas as pd
import numpy as np

from itertools import product
from src.trainers.kfold_trainer import KFoldTrainer
from src.utils.result_saver import SaveSummary


class XGBoostOptimizer:
    def __init__(self):
        self.trial_results_list = []
        self.result_directory = None
        self.trial_folder = None
        self.result_name = None
        self.params = None

    def read_params(self, file_path, model_name='XGBoost'):
        with open(file_path, 'r',encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get(model_name, {})

    def run_hyperparameter_search(self, params_file):
        model_name = 'XGBoost'
        self.params = self.read_params(params_file, model_name)

        base_result_path = self.params['base_result_path']
        result_path = os.path.join(base_result_path)
        k_fold_num = self.params['k_fold_num']
        repeat_num = self.params['repeat']
        dataset_paths = self.params['univariate_data_path']

        dataset_files = [
            name for name in os.listdir(dataset_paths)
            if os.path.isfile(os.path.join(dataset_paths, name)) and name.endswith('.csv')
        ]

        for dataset_name in dataset_files:
            dataset_path = os.path.join(dataset_paths, dataset_name)
            tw_number = ss_number = None
            parts = dataset_name.split('_')
            for part in parts:
                if part.startswith('tw'):
                    tw_number = ''.join(filter(str.isdigit, part))
                elif part.startswith('ss'):
                    ss_number = ''.join(filter(str.isdigit, part))
            window_size = int(tw_number or 5)
            step_size = int(ss_number or 3)

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            log_name = f'xgboost_optimizer_tw{window_size}_ss{step_size}.log'
            os.makedirs(result_path, exist_ok=True)
            logging.basicConfig(filename=os.path.join(result_path, log_name), level=logging.INFO, filemode='w')
            log_print = logging.info

            self.train_model(
                dataset_path, window_size, step_size,
                k_fold_num, result_path, model_name, log_print,
                repeat_num, timestamp
            )

    def train_model(self, dataset_path, window_size, step_size,
                    k_fold_num, result_path, model_name, log_print,
                    repeat_num, timestamp):

        self.result_name = f"{model_name}_tw{window_size}_ss{step_size}_{timestamp}"
        self.result_directory = os.path.join(result_path, f"{self.result_name}_results")
        os.makedirs(self.result_directory, exist_ok=True)

        config = {
            'timestamp': timestamp,
            'window_size': window_size,
            'step_size': step_size,
            'k_fold_num': k_fold_num,
            'dataset_path': dataset_path
        }
        pd.DataFrame([config]).to_csv(os.path.join(self.result_directory, f'{model_name}_experiment_config.csv'), index=False)

        param_grid_config = {
            'n_estimators': [1000],
            'max_depth': [7, 11, 15],
            'max_leaves': [127, 2047, 32767],
            'gamma': [0.0, 0.3, 0.5],
            'objective': ['multi:softmax'],
            'eval_metric': ['mlogloss']
        }
        pd.DataFrame([param_grid_config]).to_csv(
            os.path.join(self.result_directory, f'{model_name}_parameter_grid_config.csv'), index=False
        )

        param_grid = product(*param_grid_config.values())
        total_trials = np.prod([len(v) for v in param_grid_config.values()])

        for trial_number, params in enumerate(param_grid, start=1):
            log_print(f"\nRunning trial {trial_number}/{total_trials}...")
            self.trial_results_list = []
            hyper = dict(zip(param_grid_config.keys(), params))
            trial_results = {
                'trial_number': trial_number,
                'hyperparameters': hyper,
                'repeats': []
            }

            for repeat in range(1, repeat_num + 1):
                log_print(f"Running repeat {repeat}/{repeat_num}...")
                kfold_trainer = KFoldTrainer(
                    model_name=model_name,
                    hyper_params=hyper,
                    window_size=window_size,
                    step_size=step_size,
                    k_cv_num=k_fold_num,
                    dataset_path=dataset_path,
                    save_folder=result_path,
                    trial_number=trial_number,
                    target_label_column='label',  # ← ← CSV内のラベル列名に合わせて変更可
                    log_print=log_print
                )

                start_time = time.time()
                f1_cv, f1_ts, acc_cv, acc_ts, self.trial_folder, best_epochs = kfold_trainer.train_and_evaluate(
                    param_name=self.result_name, repeat=repeat,
                    #epoch=None,
                    #batch_size=None
                )
                elapsed = time.time() - start_time
                log_print(f"Repeat {repeat} finished in {datetime.timedelta(seconds=elapsed)}")
                trial_results['repeats'].append({
                    'repeat_number': repeat,
                    'accuracy_cv': acc_cv,
                    'accuracy_ts': acc_ts,
                    'f1_score_cv': f1_cv,
                    'f1_score_ts': f1_ts,
                    'best_epochs': best_epochs,
                    'elapsed_time': elapsed
                })

            trial_results_list = [trial_results]
            sv = SaveSummary(trial_number, trial_results_list, self.trial_folder, self.result_directory)
            sv.save_results()
            avg_score = sum(r['accuracy_cv'] for r in trial_results['repeats']) / len(trial_results['repeats'])
            log_print(f"Trial {trial_number} Summary:")
            log_print(f"Average CV Accuracy: {avg_score:.4f}")
            log_print(f"Best CV Accuracy: {max(r['accuracy_cv'] for r in trial_results['repeats']):.4f}")
            log_print(f"Hyperparameters: {hyper}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_file', type=str, required=True)
    args = parser.parse_args()

    optimizer = XGBoostOptimizer()
    optimizer.run_hyperparameter_search(args.params_file)


if __name__ == '__main__':
    main()
