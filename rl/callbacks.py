import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from evals.scenarios import make_env


class ValidationCallback(BaseCallback):
    """Every `every` steps: run the deterministic policy on fixed (scenario, seed) episodes and
    keep the best checkpoint. Saves the VecNormalize statistics together with the model -
    otherwise the best model would be paired with the obs statistics from the end of training."""

    def __init__(self, scenarios, seeds, every, run_dir):
        super().__init__()
        self.envs = {s: make_env(s) for s in scenarios}
        self.seeds = list(seeds)
        self.every = every
        self.run_dir = Path(run_dir)
        self.best = -np.inf
        self.next_eval = every
        self.t0 = time.time()
        self.csv_path = self.run_dir / "validation.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "minutes", "mean_score"] + list(self.envs))

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            self.validate()
        return True

    def validate(self):
        vecnorm = self.model.get_vec_normalize_env()
        scores = {}
        for name, env in self.envs.items():
            totals = []
            for seed in self.seeds:
                obs, _ = env.reset(seed=seed)
                done, total = False, 0.0
                while not done:
                    action, _ = self.model.predict(vecnorm.normalize_obs(obs[None]), deterministic=True)
                    obs, r, term, trunc, _ = env.step(action[0])
                    total += r
                    done = term or trunc
                totals.append(total * env.params.reward_scale)
            scores[name] = float(np.mean(totals))
        mean = float(np.mean(list(scores.values())))
        minutes = (time.time() - self.t0) / 60
        self.logger.record("validation/mean_score", mean)
        for name, s in scores.items():
            self.logger.record(f"validation/{name}", s)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, round(minutes, 2), round(mean, 1)]
                                   + [round(scores[n], 1) for n in self.envs])
        tag = ""
        if mean > self.best:
            self.best = mean
            self.model.save(self.run_dir / "best_model")
            vecnorm.save(str(self.run_dir / "best_vecnormalize.pkl"))
            tag = "  <- new best"
        print(f"[{self.num_timesteps:>9,d} steps | {minutes:5.1f} min] validation score {mean:>10,.0f}{tag}", flush=True)
