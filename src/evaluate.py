from __future__ import annotations

import os
from typing import Any, Dict

import yaml

from .data.loader import DataSpec, load_dataset
from .envs.market_env import EnvConfig, RiskLimits, make_env_from_df
from .utils.logging import setup_log_paths, TradeLogger, AccountSnapshotLogger
from .utils.metrics import plot_equity_curve


def read_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_env_config(cfg: Dict[str, Any], model_version: str) -> EnvConfig:
    risk = RiskLimits(
        max_position_pct=float(cfg["risk_limits"]["max_position_pct"]),
        max_daily_loss=float(cfg["risk_limits"]["max_daily_loss"]),
        max_drawdown_pct=float(cfg["risk_limits"]["max_drawdown_pct"]),
    )
    env_cfg = EnvConfig(
        window_size=int(cfg["env"]["window_size"]),
        commission=float(cfg["env"]["commission"]),
        spread=float(cfg["env"]["spread"]),
        slippage=float(cfg["env"]["slippage"]),
        leverage=float(cfg["env"]["leverage"]),
        account_size=float(cfg["env"]["account_size"]),
        margin_mode=str(cfg["env"]["margin_mode"]),
        action_space=str(cfg["env"]["action_space"]),
        risk_limits=risk,
        symbols=list(cfg["symbols"]),
        timeframes=list(cfg["timeframes"]),
        timeframe=str(cfg["timeframes"][0]),
        model_version=model_version,
    )
    return env_cfg


def backtest_random_policy(config_path: str, episodes: int = 100):
    cfg = read_config(config_path)

    log_dir = cfg["logging"]["log_dir"]
    models_dir = cfg["logging"]["models_dir"]
    log_paths = setup_log_paths(
        log_dir,
        cfg["logging"]["trade_log_filename"],
        cfg["logging"]["account_snapshot_filename"],
        cfg["logging"]["signals_filename"],
        cfg["logging"]["equity_curve_filename"],
    )

    data_spec = DataSpec(
        source=str(cfg["data"]["source"]),
        csv_path=cfg["data"].get("csv_path"),
        datetime_column=str(cfg["data"]["datetime_column"]),
        tz=str(cfg["data"]["tz"]),
        start=cfg["data"].get("start"),
        end=cfg["data"].get("end"),
    )

    symbols = list(cfg["symbols"]) or ["AAPL"]
    timeframes = list(cfg["timeframes"]) or ["1h"]
    df = load_dataset(data_spec, symbols, timeframes)

    env_cfg = build_env_config(cfg, model_version=str(cfg["model"]["version"]))
    env = make_env_from_df(df, symbols[0], timeframes[0], env_cfg)

    trade_logger = TradeLogger(log_paths.trade_log)
    acct_logger = AccountSnapshotLogger(log_paths.account_snapshot)

    import numpy as np

    equity_all = []
    position_id = 0
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step = 0
        prev_equity = info.get("equity", env.cash_balance)
        while not done:
            if env.action_space.__class__.__name__ == "Discrete":
                action = env.action_space.sample()
            else:
                action = env.action_space.sample()

            before_pos = float(env.position_size)
            before_cash = float(env.cash_balance)
            before_equity = float(prev_equity)

            obs, reward, done, truncated, step_info = env.step(action)

            equity = float(step_info.get("equity", 0.0))
            cash = float(step_info.get("cash_balance", 0.0))
            position = float(step_info.get("position_size", 0.0))
            acct_logger.log(total_balance=equity, unrealized_pnl=equity - cash, realized_pnl=env.realized_pnl, margin_used=abs(position) * 0.0, free_margin=cash, exposure=abs(position))
            equity_all.append(equity)

            # Trade logging if position changed
            delta_pos = position - before_pos
            if abs(delta_pos) > 1e-9:
                # Approximate trade record
                idx = int(env.current_step)
                ts = env.df.loc[idx, "timestamp"] if "timestamp" in env.df.columns else None
                price = float(env.df.loc[idx, "close"]) if "close" in env.df.columns else 0.0
                commission = abs(price * delta_pos) * env.config.commission
                pnl = equity - before_equity - 0.0  # reward already equals equity delta
                act_name = "BUY" if delta_pos > 0 else "SELL"
                position_id += 1
                trade_logger.log({
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "symbol": env.config.symbols[0] if env.config.symbols else "SYM",
                    "timeframe": env.config.timeframe,
                    "action": act_name,
                    "price": price,
                    "qty": abs(delta_pos),
                    "commission": commission,
                    "slippage": env.config.slippage,
                    "pnl": pnl,
                    "position_id": position_id,
                    "balance_before": before_equity,
                    "balance_after": equity,
                    "model_version": env.config.model_version,
                })

            prev_equity = equity
            step += 1
        # end episode

    plot_equity_curve(equity_all, log_paths.equity_png, title="Backtest Equity")
    return log_paths