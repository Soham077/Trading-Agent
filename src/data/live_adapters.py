from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import ccxt

from ..utils.logging import SignalsLogger


@dataclass
class LiveConfig:
    exchange: str
    market_type: str
    testnet: bool
    api_key: Optional[str]
    api_secret: Optional[str]
    poll_interval_seconds: float


def make_ccxt_exchange(cfg: LiveConfig) -> ccxt.Exchange:
    ex_class = getattr(ccxt, cfg.exchange)
    params = {}
    if cfg.exchange == "binance" and cfg.testnet:
        params["options"] = {"defaultType": cfg.market_type}
        ex = ex_class({
            "apiKey": cfg.api_key or os.getenv("BINANCE_API_KEY", ""),
            "secret": cfg.api_secret or os.getenv("BINANCE_API_SECRET", ""),
            "enableRateLimit": True,
            "urls": {"api": {"public": "https://testnet.binance.vision/api", "private": "https://testnet.binance.vision/api"}},
            **params,
        })
    else:
        ex = ex_class({
            "apiKey": cfg.api_key or "",
            "secret": cfg.api_secret or "",
            "enableRateLimit": True,
            **params,
        })
    return ex


def paper_signal_writer(symbol: str, side: str, qty: float, price: float, model_version: str, logger: SignalsLogger) -> None:
    ts = datetime.now(timezone.utc)
    logger.log(ts, symbol, side, price, qty, model_version)


def fetch_ticker_price(exchange: ccxt.Exchange, symbol: str) -> float:
    ticker = exchange.fetch_ticker(symbol)
    # attempt to use last or close
    price = ticker.get("last") or ticker.get("close")
    if price is None:
        raise RuntimeError(f"Could not fetch price for {symbol}")
    return float(price)