from __future__ import annotations

import os
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class StepCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"model_step_{self.n_calls}.zip")
            try:
                self.model.save(model_path)
                if self.verbose:
                    print(f"Saved checkpoint: {model_path}")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to save checkpoint: {e}")
        return True


def make_checkpoint_callback(checkpoint_interval_steps: int, models_dir: str) -> Optional[StepCheckpointCallback]:
    if checkpoint_interval_steps and checkpoint_interval_steps > 0:
        return StepCheckpointCallback(save_freq=checkpoint_interval_steps, save_path=models_dir)
    return None