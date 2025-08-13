import os
from src.evaluate import backtest_random_policy

if __name__ == "__main__":
    config_path = os.environ.get("CONFIG", "config.example.yaml")
    paths = backtest_random_policy(config_path, episodes=100)
    print(f"Smoke backtest done. Logs at: {paths.log_dir}")