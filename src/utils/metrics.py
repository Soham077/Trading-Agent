from __future__ import annotations

import math
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_returns(equity: Iterable[float]) -> np.ndarray:
    equity = np.asarray(list(equity), dtype=float)
    returns = np.diff(equity) / equity[:-1]
    return returns


def rolling_sharpe_ratio(equity: Iterable[float], window: int = 30, risk_free_rate: float = 0.0) -> pd.Series:
    returns = pd.Series(compute_returns(equity))
    if len(returns) == 0:
        return pd.Series(dtype=float)
    excess = returns - (risk_free_rate / max(len(returns), 1))
    sharpe = excess.rolling(window).mean() / (excess.rolling(window).std(ddof=1) + 1e-8)
    return sharpe


def max_drawdown(equity: Iterable[float]) -> Tuple[float, int, int]:
    equity = np.asarray(list(equity), dtype=float)
    if equity.size == 0:
        return 0.0, -1, -1
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / (peaks + 1e-8)
    min_dd_idx = int(np.argmin(drawdowns))
    min_dd = float(drawdowns[min_dd_idx])
    peak_idx = int(np.argmax(equity[: min_dd_idx + 1])) if min_dd_idx >= 0 else -1
    return min_dd, peak_idx, min_dd_idx


def plot_equity_curve(equity: Iterable[float], out_path: str, title: str = "Equity Curve") -> None:
    eq = list(map(float, equity))
    plt.figure(figsize=(10, 5))
    plt.plot(eq, label="Equity")
    mdd, _, _ = max_drawdown(eq)
    plt.title(f"{title} | Max DD: {mdd:.2%}")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()