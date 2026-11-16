"""Wraps a trained Stable-Baselines3 model so it looks like any other Agent."""
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.base import Agent
from scsim.environment import SupplyChainEnv

ALGOS = {"ppo": PPO, "a2c": A2C}


class RLAgent(Agent):
    def __init__(self, run_dir, name="ppo", checkpoint="best"):
        run_dir = Path(run_dir)
        cfg = yaml.safe_load(open(run_dir / "train_config.yaml"))
        self.name = name
        self.model = ALGOS[cfg["algo"]].load(run_dir / f"{checkpoint}_model.zip", device="cpu")
        # only the observation statistics are needed, the dummy env is just there to load them
        self.vecnorm = VecNormalize.load(str(run_dir / f"{checkpoint}_vecnormalize.pkl"),
                                         DummyVecEnv([SupplyChainEnv]))
        self.vecnorm.training = False
        self.vecnorm.norm_reward = False

    def act(self, obs):
        o = self.vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32)[None])
        action, _ = self.model.predict(o, deterministic=True)
        return action[0]
