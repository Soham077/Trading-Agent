import pandas as pd
import numpy as np

from src.envs.market_env import MarketEnv, EnvConfig, RiskLimits, make_env_from_df


def make_dummy_df(n=200):
    ts = pd.date_range("2020-01-01", periods=n, freq="H", tz="UTC")
    data = pd.DataFrame({
        "timestamp": ts,
        "open": np.linspace(100, 120, n),
        "high": np.linspace(101, 121, n),
        "low": np.linspace(99, 119, n),
        "close": np.linspace(100, 120, n) + np.sin(np.arange(n)),
        "volume": np.random.randint(1000, 2000, size=n)
    })
    return data


def make_cfg():
    return EnvConfig(
        window_size=30,
        commission=0.0005,
        spread=0.0,
        slippage=0.0,
        leverage=1,
        account_size=10000.0,
        margin_mode="cross",
        action_space="discrete",
        risk_limits=RiskLimits(0.02, 0.05, 0.25),
        symbols=["AAPL"],
        timeframes=["1h"],
        timeframe="1h",
        model_version="test",
    )


def test_reset_and_step():
    df = make_dummy_df()
    env = make_env_from_df(df, "AAPL", "1h", make_cfg())
    obs, info = env.reset()
    assert obs.shape[0] == 30
    done = False
    steps = 0
    while not done and steps < 50:
        action = 0
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    assert steps > 0