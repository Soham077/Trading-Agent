from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import yaml

from .evaluate import backtest_random_policy
from .train import train_entrypoint
from .data.loader import DataSpec, load_dataset, load_yf
from .envs.market_env import EnvConfig, RiskLimits, make_env_from_df
from .utils.logging import setup_log_paths, SignalsLogger
from .data.live_adapters import LiveConfig, make_ccxt_exchange, fetch_ticker_price, paper_signal_writer
from .utils.config import expand_env_vars


def load_cfg(path: str):
	with open(path, "r") as f:
		raw = yaml.safe_load(f)
	return expand_env_vars(raw)


def require_human_approval() -> None:
	# In non-interactive mode, we do not allow auto top-ups
	raise PermissionError("Human approval required for any account funding change. 'auto_topup.enabled' must remain false.")


def _yf_symbol(sym: str) -> str:
	# crude mapping for common crypto
	if "/" in sym:
		base, quote = sym.split("/", 1)
		if quote in ("USDT", "USD"):
			return f"{base}-USD"
	return sym


def run_paper(cfg_path: str):
	cfg = load_cfg(cfg_path)
	if cfg.get("auto_topup", {}).get("enabled", False):
		require_human_approval()

	log_paths = setup_log_paths(
		cfg["logging"]["log_dir"],
		cfg["logging"]["trade_log_filename"],
		cfg["logging"]["account_snapshot_filename"],
		cfg["logging"]["signals_filename"],
		cfg["logging"]["equity_curve_filename"],
	)
	signals_logger = SignalsLogger(log_paths.signals_file)

	live_cfg = LiveConfig(
		exchange=str(cfg["live"]["exchange"]),
		market_type=str(cfg["live"].get("market_type", "spot")),
		testnet=bool(cfg["live"].get("testnet", True)),
		api_key=cfg["live"].get("api_key"),
		api_secret=cfg["live"].get("api_secret"),
		poll_interval_seconds=float(cfg["live"].get("poll_interval_seconds", 1.0)),
	)

	# Try to initialize exchange, but tolerate failure
	exchange = None
	try:
		exchange = make_ccxt_exchange(live_cfg)
	except Exception:
		exchange = None

	symbols = cfg.get("symbols", ["BTC/USDT"]) or ["BTC/USDT"]

	# Fallback dataset for last known price if exchange not available
	last_prices = {}
	try:
		data_spec = DataSpec(
			source=str(cfg["data"].get("source", "csv")),
			csv_path=cfg["data"].get("csv_path"),
			datetime_column=str(cfg["data"].get("datetime_column", "timestamp")),
			tz=str(cfg["data"].get("tz", "UTC")),
			start=cfg["data"].get("start"),
			end=cfg["data"].get("end"),
		)
		if data_spec.source == "csv" and data_spec.csv_path and os.path.exists(data_spec.csv_path):
			import pandas as pd
			df = load_dataset(data_spec, symbols, cfg.get("timeframes", ["1h"]))
			if "close" in df.columns:
				last_prices[symbols[0]] = float(df["close"].iloc[-1])
	except Exception:
		pass

	import time, random
	for _ in range(5):
		for s in symbols:
			price = None
			# Try live price first
			if exchange is not None:
				try:
					price = fetch_ticker_price(exchange, s)
				except Exception:
					price = None
			# Try yfinance
			if price is None:
				try:
					yf_sym = _yf_symbol(s)
					df = load_yf(yf_sym, cfg.get("timeframes", ["1h"])[0], cfg["data"].get("start"), cfg["data"].get("end"))
					if len(df) > 0:
						price = float(df["close"].iloc[-1])
				except Exception:
					price = None
			# Try last known
			if price is None:
				price = float(last_prices.get(s, 0.0))

			side = random.choice(["BUY", "SELL", "HOLD"]) if price else "HOLD"
			qty = 0.0 if side == "HOLD" else 0.001
			paper_signal_writer(s, side, qty, float(price or 0.0), cfg["model"].get("version", "untrained"), signals_logger)
		time.sleep(live_cfg.poll_interval_seconds)


def run_live(cfg_path: str):
	# By default, we do not place real orders; would mirror paper but behind feature flag
	print("Live mode is disabled by default for safety. Use paper mode.")


def main():
	parser = argparse.ArgumentParser(description="Autonomous RL Trading Engine")
	parser.add_argument("--mode", choices=["backtest", "train", "paper", "live"], default="paper")
	parser.add_argument("--config", required=True)
	args = parser.parse_args()

	if args.mode == "backtest":
		paths = backtest_random_policy(args.config, episodes=100)
		print(f"Backtest complete. Logs at: {paths.log_dir}")
	elif args.mode == "train":
		model_path = train_entrypoint(args.config)
		print(f"Training complete. Model saved to: {model_path}")
	elif args.mode == "paper":
		run_paper(args.config)
		print("Paper run complete. See signals.csv in log_dir.")
	elif args.mode == "live":
		run_live(args.config)


if __name__ == "__main__":
	main()