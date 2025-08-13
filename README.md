# Autonomous RL Trading Engine (Safe-by-default)

Quickstart
- Install Python 3.11+
- Create venv and install requirements:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Copy config and run backtest:
  ```bash
  cp config.example.yaml config.yaml
  python main.py --mode backtest --config config.yaml
  ```
- Paper mode (signals only, no real orders):
  ```bash
  python main.py --mode paper --config config.yaml
  ```

Safety Defaults
- Defaults to paper mode.
- auto_topup.enabled: false (human approval required to change)
- Risk limits enforced in env: max_position_pct, max_daily_loss, max_drawdown_pct

Outputs
- logs/trades.csv: trade log
- logs/account_snapshots.csv: account snapshots
- logs/equity.png: equity curve
- logs/signals.csv: paper/live signals

CLI
```bash
python main.py --mode {backtest,train,paper,live} --config config.yaml
```

Colab
- See `notebooks/colab_quickstart.ipynb` for one-click backtest + paper run.
