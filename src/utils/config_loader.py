# C:\zebrafish_behavior_analysis\src\utils\config_loader.py
import yaml
import os

def load_config(config_path): # ★★★ この行が重要！
    """指定されたYAML設定ファイルを読み込む"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return None

def get_project_root(): # ★★★ この行も重要！
    """プロジェクトのルートディレクトリのパスを取得する"""
    # config_loader.py が src/utils/ にあるので、2つ上のディレクトリがルート
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 例: 使い方 (このブロックは実行時に必須ではないが、テスト用に残しておく)
if __name__ == "__main__":
    project_root = get_project_root()
    exp_config_path = os.path.join(project_root, 'config', 'experiment_config.yaml')

    exp_config = load_config(exp_config_path)

    if exp_config:
        print("\nExperiment Config Loaded:")
        print(exp_config)