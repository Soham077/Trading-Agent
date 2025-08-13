import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


@dataclass
class LogPaths:
    log_dir: str
    trade_log: str
    account_snapshot: str
    signals_file: str
    equity_png: str


def setup_log_paths(base_dir: str, trade_log_filename: str, account_snapshot_filename: str, signals_filename: str, equity_curve_filename: str) -> LogPaths:
    ensure_dir(base_dir)
    trade_log = os.path.join(base_dir, trade_log_filename)
    account_snapshot = os.path.join(base_dir, account_snapshot_filename)
    signals_file = os.path.join(base_dir, signals_filename)
    equity_png = os.path.join(base_dir, equity_curve_filename)
    return LogPaths(base_dir, trade_log, account_snapshot, signals_file, equity_png)


class TradeLogger:
    HEADER = [
        "timestamp","symbol","timeframe","action","price","qty","commission","slippage","pnl","position_id","balance_before","balance_after","model_version"
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADER)

    def log(self, row: Dict[str, object]) -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                row.get("timestamp"),
                row.get("symbol"),
                row.get("timeframe"),
                row.get("action"),
                row.get("price"),
                row.get("qty"),
                row.get("commission"),
                row.get("slippage"),
                row.get("pnl"),
                row.get("position_id"),
                row.get("balance_before"),
                row.get("balance_after"),
                row.get("model_version"),
            ])


class AccountSnapshotLogger:
    HEADER = [
        "timestamp","total_balance","unrealized_pnl","realized_pnl","margin_used","free_margin","exposure"
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADER)

    def log(self, total_balance: float, unrealized_pnl: float, realized_pnl: float, margin_used: float, free_margin: float, exposure: float, ts: Optional[datetime] = None) -> None:
        ts = ts or datetime.now(timezone.utc)
        iso = ts.isoformat()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([iso, total_balance, unrealized_pnl, realized_pnl, margin_used, free_margin, exposure])


class SignalsLogger:
    HEADER = ["timestamp","symbol","action","price","qty","model_version","notes"]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADER)

    def log(self, timestamp: datetime, symbol: str, action: str, price: float, qty: float, model_version: str, notes: str = "") -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp.isoformat(), symbol, action, qty, price, model_version, notes])