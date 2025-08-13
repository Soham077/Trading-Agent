from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


@dataclass
class DataSpec:
    source: str  # csv|yfinance|ccxt
    csv_path: Optional[str]
    datetime_column: str
    tz: str
    start: Optional[str]
    end: Optional[str]


SUPPORTED_TIMEFRAME_TO_YF = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d",
}


def load_csv(csv_path: str, datetime_column: str, tz: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV path not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if datetime_column not in df.columns:
        raise ValueError(f"CSV missing datetime column: {datetime_column}")
    df[datetime_column] = pd.to_datetime(df[datetime_column], utc=True)
    if tz and tz.upper() != "UTC":
        df[datetime_column] = df[datetime_column].dt.tz_convert(tz)
    df = df.sort_values(datetime_column).reset_index(drop=True)
    return df


def load_yf(symbol: str, timeframe: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    interval = SUPPORTED_TIMEFRAME_TO_YF.get(timeframe)
    if interval is None:
        raise ValueError(f"Unsupported timeframe for yfinance: {timeframe}")
    ticker = yf.Ticker(symbol)
    hist = ticker.history(interval=interval, start=start, end=end)
    hist = hist.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    hist = hist.reset_index().rename(columns={"Date": "timestamp"})
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)
    hist = hist.dropna().reset_index(drop=True)
    return hist[["timestamp", "open", "high", "low", "close", "volume"]]


def align_multi(df_by_symbol: Dict[Tuple[str, str], pd.DataFrame], datetime_col: str) -> pd.DataFrame:
    frames = []
    for (symbol, timeframe), df in df_by_symbol.items():
        d = df.copy()
        d = d[[datetime_col, "open", "high", "low", "close", "volume"]]
        d = d.rename(columns={
            "open": f"{symbol}_{timeframe}_open",
            "high": f"{symbol}_{timeframe}_high",
            "low": f"{symbol}_{timeframe}_low",
            "close": f"{symbol}_{timeframe}_close",
            "volume": f"{symbol}_{timeframe}_volume",
        })
        frames.append(d)
    out = frames[0]
    for f in frames[1:]:
        out = pd.merge_asof(out.sort_values(datetime_col), f.sort_values(datetime_col), on=datetime_col, direction="nearest")
    return out.dropna().reset_index(drop=True)


def load_dataset(spec: DataSpec, symbols: List[str], timeframes: List[str]) -> pd.DataFrame:
    if spec.source == "csv":
        df = load_csv(spec.csv_path or "", spec.datetime_column, spec.tz)
        return df
    elif spec.source == "yfinance":
        by = {}
        for s in symbols:
            for tf in timeframes:
                by[(s, tf)] = load_yf(s, tf, spec.start, spec.end)
        aligned = align_multi(by, "timestamp")
        return aligned
    else:
        raise NotImplementedError("CCXT and live adapters are implemented in live_adapters.py for runtime use")