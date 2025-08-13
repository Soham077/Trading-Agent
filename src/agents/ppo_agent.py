from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_vec_env(env_fn, num_envs: int, vec_type: str = "dummy"):
    if num_envs <= 1:
        return DummyVecEnv([env_fn])
    if vec_type == "subproc":
        return SubprocVecEnv([env_fn for _ in range(num_envs)])
    return DummyVecEnv([env_fn for _ in range(num_envs)])


def train_ppo(env_fn, num_envs: int, vec_type: str, total_timesteps: int, learning_rate: float, n_steps: int, batch_size: int, policy_kwargs: Optional[Dict[str, Any]] = None, tensorboard_log: Optional[str] = None, callback=None):
    vec_env = make_vec_env(env_fn, num_envs, vec_type)
    model = PPO("MlpPolicy", vec_env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, policy_kwargs=policy_kwargs or {}, verbose=1, tensorboard_log=tensorboard_log)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return model


def evaluate_ppo(model: PPO, env: gym.Env, episodes: int = 10):
    rewards = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
            if done or truncated:
                break
        rewards.append(ep_reward)
    return {
        "mean_reward": sum(rewards) / max(len(rewards), 1),
        "rewards": rewards,
    }