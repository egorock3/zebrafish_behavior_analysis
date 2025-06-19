# C:\zebrafish_behavior_analysis\src\utils\result_saver.py
import os
import pandas as pd
import json
import logging

class SaveSummary: # ★★★ この行が重要！
    def __init__(self, trial_results, current_trial_folder, base_results_path, model_name):
        self.trial_results = trial_results
        self.current_trial_folder = current_trial_folder
        self.base_results_path = base_results_path
        self.model_name = model_name
        self.log_print = logging.info

    def save_results(self):
        summary_data = []
        for repeat_result in self.trial_results['repeats']:
            row = {
                'trial_number': self.trial_results['trial_number'],
                'data_tw': self.trial_results['data_tw'],
                'data_ss': self.trial_results['data_ss'],
                'repeat_number': repeat_result['repeat_number'],
                'hyperparameters': json.dumps(self.trial_results['hyperparameters']),
                'accuracy_cv': repeat_result['accuracy_cv'],
                'accuracy_ts': repeat_result['accuracy_ts'],
                'f1_score_cv': repeat_result['f1_score_cv'],
                'f1_score_ts': repeat_result['f1_score_ts'],
                'best_epochs': repeat_result['best_epochs'],
                'elapsed_time_seconds': repeat_result['elapsed_time']
            }
            summary_data.append(row)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            summary_filename = f"{self.model_name}_tw{self.trial_results['data_tw']:02d}_ss{self.trial_results['data_ss']:02d}_summary.csv"
            summary_filepath = os.path.join(self.base_results_path, summary_filename)

            if os.path.exists(summary_filepath):
                summary_df.to_csv(summary_filepath, mode='a', header=False, index=False)
                self.log_print(f"Results appended to {summary_filepath}")
            else:
                summary_df.to_csv(summary_filepath, index=False)
                self.log_print(f"Results saved to {summary_filepath}")
        else:
            self.log_print("No summary data to save for this trial.")