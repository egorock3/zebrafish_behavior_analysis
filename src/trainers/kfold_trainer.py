# C:\zebrafish_behavior_analysis\src\trainers\kfold_trainer.py
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import xgboost as xgb
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder # LabelEncoderをインポート

class KFoldTrainer:
    def __init__(self, model_name, hyper_params, window_size, step_size, k_cv_num, dataset_path, save_folder, trial_number, target_label_column, log_print):
        self.model_name = model_name
        self.hyper_params = hyper_params
        self.window_size = window_size
        self.step_size = step_size
        self.k_cv_num = k_cv_num
        self.dataset_path = dataset_path
        self.save_folder = save_folder
        self.trial_number = trial_number
        self.target_label_column = target_label_column
        self.log_print = log_print
        self.num_classes = 0
        self.class_labels = []
        self.label_encoder = LabelEncoder() # LabelEncoderのインスタンス化

    def train_and_evaluate(self, param_name, repeat, training_settings):
        # ... (データロードと分割のtry-exceptブロック) ...

        kf = KFold(n_splits=self.k_cv_num, shuffle=True, random_state=self.hyper_params.get('seed', 42) + repeat)
        
        cv_f1_scores = []
        cv_accuracies = []
        best_epochs_list = []

        self.log_print(f"Starting {self.k_cv_num}-Fold Cross Validation...")
        for fold, (train_index, val_index) in enumerate(kf.split(X_train_full, y_train_full)):
            self.log_print(f"--- Fold {fold+1}/{self.k_cv_num} ---")
            X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
            y_train, y_val = y_train_full[train_index], y_train_full[val_index]

            if self.model_name == 'XGBoost':
                model_params = self.hyper_params.copy()
                model_params['num_class'] = self.num_classes

                model = xgb.XGBClassifier(**model_params)
                
                verbose = training_settings.get('verbose', False) 

                model.fit(X_train, y_train,
                          verbose=verbose)
                
                best_epoch = model_params.get('n_estimators', 100) 
                best_epochs_list.append(best_epoch)

                y_pred_val = model.predict(X_val)
            else:
                self.log_print(f"Error: Model '{self.model_name}' is not implemented in KFoldTrainer.")
                # ここでNoneを返して、後続の処理に進まないようにする
                return None, None, None, None, None, None

            cv_f1_scores.append(f1_score(y_val, y_pred_val, average='macro', zero_division=0))
            cv_accuracies.append(accuracy_score(y_val, y_pred_val))
            
        avg_f1_cv = np.mean(cv_f1_scores)
        avg_accuracy_cv = np.mean(cv_accuracies)
        avg_best_epoch = int(np.mean(best_epochs_list)) if best_epochs_list else None
        
        self.log_print(f"CV finished. Avg F1: {avg_f1_cv:.4f}, Avg Accuracy: {avg_accuracy_cv:.4f}, Avg Best Epoch: {avg_best_epoch}")

        # ★★★ここから変更★★★
        # final_model をここで初期化し、スコープをtrain_and_evaluateメソッド全体に広げる
        final_model = None 
        y_pred_test = None # y_pred_test もここで初期化しておく

        # テストセットでの最終評価
        self.log_print("Training final model on full training data and evaluating on test set...")
        if self.model_name == 'XGBoost':
            model_params = self.hyper_params.copy()
            model_params['num_class'] = self.num_classes
            
            final_model = xgb.XGBClassifier(**model_params) # ここでfinal_modelにインスタンスを代入
            
            verbose = training_settings.get('verbose', False) 

            final_model.fit(X_train_full, y_train_full,
                            verbose=verbose)
            
            y_pred_test = final_model.predict(X_test) # ここでy_pred_testに代入
            
            # モデルの保存
            model_save_dir = os.path.join(self.save_folder, f"{param_name}_results", f"trial_{self.trial_number}_repeat_{repeat}")
            os.makedirs(model_save_dir, exist_ok=True)
            final_model.save_model(os.path.join(model_save_dir, f"{self.model_name}_final_model.json"))
            self.log_print(f"Final model saved to {os.path.join(model_save_dir, f'{self.model_name}_final_model.json')}")
        else: # XGBoost以外のモデルの場合も考慮
            self.log_print(f"Error: Final model training for '{self.model_name}' is not implemented.")
            return None, None, None, None, None, None
        # ★★★変更ここまで★★★


        f1_ts = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        accuracy_ts = accuracy_score(y_test, y_pred_test)
        
        self.log_print(f"Test evaluation: F1: {f1_ts:.4f}, Accuracy: {accuracy_ts:.4f}")


        # 結果保存用の個別フォルダ
        current_trial_folder = os.path.join(self.save_folder, f"{param_name}_results", f"trial_{self.trial_number}_repeat_{repeat}")
        os.makedirs(current_trial_folder, exist_ok=True)

        # 混同行列の可視化と保存
        self._plot_confusion_matrix(y_test, y_pred_test, current_trial_folder, f"{self.model_name}_confusion_matrix_test.png")

        return avg_f1_cv, f1_ts, avg_accuracy_cv, accuracy_ts, current_trial_folder, avg_best_epoch

    def _plot_confusion_matrix(self, y_true, y_pred, save_path, filename):
        """混同行列をプロットして保存するヘルパー関数"""
        cm = confusion_matrix(y_true, y_pred, labels=self.class_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        full_save_path = os.path.join(save_path, filename)
        plt.savefig(full_save_path)
        plt.close() # プロットを閉じてメモリを解放
        self.log_print(f"Confusion matrix saved to {full_save_path}")