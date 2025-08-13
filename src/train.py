from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import yaml

from .data.loader import DataSpec, load_dataset
from .envs.market_env import EnvConfig, RiskLimits, make_env_from_df
from .agents.ppo_agent import train_ppo
from .agents.sac_agent import train_sac
from .utils.checkpoint import make_checkpoint_callback


@dataclass
class TrainConfig:
    algorithm: str
    total_timesteps: int
    learning_rate: float
    batch_size: int
    n_steps: int
    policy_kwargs: Dict[str, Any]


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


def train_entrypoint(config_path: str):
    cfg = read_config(config_path)

    # Ensure safety defaults
    assert cfg["auto_topup"]["enabled"] is False, "auto_topup must be disabled by default"

    log_dir = cfg["logging"]["log_dir"]
    models_dir = cfg["logging"]["models_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    data_spec = DataSpec(
        source=str(cfg["data"]["source"]),
        csv_path=cfg["data"].get("csv_path"),
        datetime_column=str(cfg["data"]["datetime_column"]),
        tz=str(cfg["data"]["tz"]),
        start=cfg["data"].get("start"),
        end=cfg["data"].get("end"),
    )

    symbols: List[str] = list(cfg["symbols"]) or ["AAPL"]
    timeframes: List[str] = list(cfg["timeframes"]) or ["1h"]
    df = load_dataset(data_spec, symbols, timeframes)

    env_cfg = build_env_config(cfg, model_version=str(cfg["model"]["version"]))
    env_fn = lambda: make_env_from_df(df, symbols[0], timeframes[0], env_cfg)

    algo = str(cfg["train"]["algorithm"]).lower()
    total_timesteps = int(cfg["train"]["total_timesteps"]) or 10000
    learning_rate = float(cfg["train"]["learning_rate"]) or 3e-4
    n_steps = int(cfg["train"].get("n_steps", 2048))
    batch_size = int(cfg["train"]["batch_size"]) or 64
    policy_kwargs = cfg["train"].get("policy_kwargs", {}) or {}

    num_envs = int(cfg["parallel"]["num_envs"]) or 1
    vec_type = str(cfg["parallel"]["vec_env"]) or "dummy"

    tensorboard_log = os.path.join(log_dir, "tb")
    os.makedirs(tensorboard_log, exist_ok=True)

    checkpoint_cb = make_checkpoint_callback(int(cfg["logging"].get("checkpoint_interval_steps", 0)), models_dir)

    if algo == "ppo":
        model = train_ppo(env_fn, num_envs, vec_type, total_timesteps, learning_rate, n_steps, batch_size, policy_kwargs, tensorboard_log, checkpoint_cb)
    elif algo == "sac":
        model = train_sac(env_fn, num_envs, vec_type, total_timesteps, learning_rate, batch_size, policy_kwargs, tensorboard_log, checkpoint_cb)
    else:
        raise ValueError("Unsupported algorithm: choose 'ppo' or 'sac'")

    # Save final model
    final_path = os.path.join(models_dir, f"{algo}_final.zip")
    model.save(final_path)
    return final_path