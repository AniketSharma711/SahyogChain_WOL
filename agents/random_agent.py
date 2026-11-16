import numpy as np

from agents.base import Agent


class RandomAgent(Agent):
    """Uniformly random action every day. Lower bound for everything else."""

    name = "random"

    def reset(self, env, seed=None):
        self.nvec = env.action_space.nvec
        self.rng = np.random.default_rng(seed)  # own rng so results don't depend on the env's

    def act(self, obs):
        return self.rng.integers(0, self.nvec)
