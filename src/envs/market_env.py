from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class RiskLimits:
    max_position_pct: float
    max_daily_loss: float
    max_drawdown_pct: float


@dataclass
class EnvConfig:
    window_size: int
    commission: float
    spread: float
    slippage: float
    leverage: float
    account_size: float
    margin_mode: str
    action_space: str  # discrete|continuous
    risk_limits: RiskLimits
    symbols: List[str]
    timeframes: List[str]
    timeframe: str
    model_version: str


class MarketEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, price_col: str, config: EnvConfig):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.price_col = price_col
        self.config = config
        self.window_size = config.window_size

        # State: window of OHLCV + position/balance
        num_features = 5  # ohlcv: open, high, low, close, volume
        self.observation_shape = (self.window_size, num_features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)

        if config.action_space == "discrete":
            # 0: hold, 1: long, 2: short, 3: close
            self.action_space = spaces.Discrete(4)
        else:
            # continuous action: target position in [-1, 1]
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Internal state
        self.current_step = 0
        self.position_size = 0.0  # positive long, negative short
        self.entry_price = None
        self.cash_balance = float(self.config.account_size)
        self.equity_curve: List[float] = []
        self.realized_pnl = 0.0
        self.max_equity = self.cash_balance
        self.episode_start_equity = self.cash_balance

    def _get_price(self, idx: int) -> float:
        return float(self.df.loc[idx, self.price_col])

    def _get_features(self, idx: int) -> np.ndarray:
        start = max(0, idx - self.window_size + 1)
        window = self.df.iloc[start: idx + 1]
        if len(window) < self.window_size:
            pad = np.zeros((self.window_size - len(window), 5), dtype=np.float32)
        else:
            pad = np.zeros((0, 5), dtype=np.float32)
        feats = window[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)
        obs = np.vstack([pad, feats])
        return obs

    def _update_equity(self, price: float) -> float:
        position_value = self.position_size * price
        equity = self.cash_balance + position_value
        self.max_equity = max(self.max_equity, equity)
        self.equity_curve.append(equity)
        return equity

    def _apply_risk_limits(self, next_price: float) -> None:
        # Max position size
        max_pos_value = self.config.account_size * self.config.risk_limits.max_position_pct * self.config.leverage
        if abs(self.position_size * next_price) > max_pos_value:
            # scale down to cap
            desired_qty = math.copysign(max_pos_value / max(next_price, 1e-8), self.position_size)
            self._close_partial(self.position_size - desired_qty, next_price)

        # Daily loss
        daily_loss = (self.equity_curve[-1] - self.episode_start_equity) / max(self.episode_start_equity, 1e-8)
        if daily_loss <= -abs(self.config.risk_limits.max_daily_loss):
            # force flat
            if self.position_size != 0.0:
                self._close_position(next_price)

        # Max drawdown
        if self.max_equity > 0:
            dd = (self.equity_curve[-1] - self.max_equity) / self.max_equity
            if dd <= -abs(self.config.risk_limits.max_drawdown_pct):
                if self.position_size != 0.0:
                    self._close_position(next_price)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.position_size = 0.0
        self.entry_price = None
        self.cash_balance = float(self.config.account_size)
        self.realized_pnl = 0.0
        self.equity_curve = [self.cash_balance]
        self.max_equity = self.cash_balance
        self.episode_start_equity = self.cash_balance
        obs = self._get_features(self.current_step - 1)
        info = {"equity": self.cash_balance}
        return obs, info

    def _close_position(self, price: float) -> float:
        if self.position_size == 0.0:
            return 0.0
        pnl = (price - self.entry_price) * self.position_size
        commission = abs(price * self.position_size) * self.config.commission
        pnl -= commission
        self.cash_balance += pnl
        self.realized_pnl += pnl
        self.position_size = 0.0
        self.entry_price = None
        return pnl

    def _close_partial(self, qty_to_close: float, price: float) -> float:
        if qty_to_close == 0.0:
            return 0.0
        if self.position_size == 0.0:
            return 0.0
        close_qty = np.clip(qty_to_close, -abs(self.position_size), abs(self.position_size))
        sign = math.copysign(1.0, self.position_size)
        if sign > 0:
            # reducing long
            pnl = (price - self.entry_price) * close_qty
        else:
            # reducing short
            pnl = (self.entry_price - price) * close_qty
        commission = abs(price * close_qty) * self.config.commission
        pnl -= commission
        self.cash_balance += pnl
        self.realized_pnl += pnl
        self.position_size -= close_qty
        if self.position_size == 0.0:
            self.entry_price = None
        return pnl

    def step(self, action):
        done = False
        info: Dict[str, float] = {}

        if self.current_step >= len(self.df) - 1:
            done = True
            obs = self._get_features(self.current_step - 1)
            return obs, 0.0, done, False, info

        price = self._get_price(self.current_step)
        next_price = self._get_price(self.current_step + 1)

        if isinstance(action, (list, tuple, np.ndarray)) and self.config.action_space == "continuous":
            target_pos = float(np.clip(action[0], -1.0, 1.0))
            target_value = target_pos * self.cash_balance * self.config.leverage
            target_qty = target_value / max(price, 1e-8)
            delta_qty = target_qty - self.position_size
            if abs(delta_qty) > 1e-8:
                # apply slippage and spread at entry
                trade_price = price * (1 + self.config.spread + math.copysign(self.config.slippage, delta_qty))
                commission = abs(trade_price * delta_qty) * self.config.commission
                self.cash_balance -= commission
                if self.position_size == 0.0:
                    self.entry_price = trade_price
                else:
                    # average price for simplicity
                    total_qty = self.position_size + delta_qty
                    if total_qty != 0:
                        self.entry_price = (self.entry_price * self.position_size + trade_price * delta_qty) / total_qty
                self.position_size += delta_qty
        else:
            # discrete
            act = int(action)
            if act == 0:
                pass
            elif act == 1:  # go long 1x leverage on full allowed capital
                max_pos_value = self.config.account_size * self.config.risk_limits.max_position_pct * self.config.leverage
                qty = max_pos_value / max(price, 1e-8)
                trade_price = price * (1 + self.config.spread + self.config.slippage)
                commission = abs(trade_price * qty) * self.config.commission
                self.cash_balance -= commission
                if self.position_size == 0.0:
                    self.entry_price = trade_price
                else:
                    total_qty = self.position_size + qty
                    if total_qty != 0:
                        self.entry_price = (self.entry_price * self.position_size + trade_price * qty) / total_qty
                self.position_size += qty
            elif act == 2:  # go short
                max_pos_value = self.config.account_size * self.config.risk_limits.max_position_pct * self.config.leverage
                qty = max_pos_value / max(price, 1e-8)
                trade_price = price * (1 - self.config.spread - self.config.slippage)
                commission = abs(trade_price * qty) * self.config.commission
                self.cash_balance -= commission
                if self.position_size == 0.0:
                    self.entry_price = trade_price
                else:
                    total_qty = self.position_size - qty
                    if total_qty != 0:
                        self.entry_price = (self.entry_price * self.position_size - trade_price * qty) / total_qty
                self.position_size -= qty
            elif act == 3:
                self._close_position(price)

        equity = self._update_equity(next_price)

        # Safety caps
        self._apply_risk_limits(next_price)

        reward = equity - self.equity_curve[-2] if len(self.equity_curve) >= 2 else 0.0

        self.current_step += 1
        obs = self._get_features(self.current_step - 1)
        truncated = False
        info.update({
            "equity": equity,
            "cash_balance": self.cash_balance,
            "position_size": self.position_size,
            "entry_price": self.entry_price if self.entry_price is not None else 0.0,
        })
        if self.current_step >= len(self.df) - 1:
            done = True
        return obs, float(reward), bool(done), bool(truncated), info

    def render(self):
        pass


def make_env_from_df(df: pd.DataFrame, symbol: str, timeframe: str, config: EnvConfig) -> MarketEnv:
    # Expect columns: timestamp, open, high, low, close, volume
    price_col = "close"
    env = MarketEnv(df, price_col, config)
    return env