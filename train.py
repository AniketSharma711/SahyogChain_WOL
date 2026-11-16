"""
Train an RL planner with Stable-Baselines3 on a mix of scenarios.

    python train.py --config configs/train_ppo.yaml --run-name ppo_seed0
    python train.py --config configs/train_ppo.yaml --seed 1 --run-name ppo_seed1
    python train.py --config configs/train_a2c.yaml --run-name a2c_seed0

Everything for a run ends up in runs/<run-name>/:
    train_config.yaml      the exact training config used
    env_configs.yaml       the scenario configs the agent was trained on
    validation.csv         validation score over time (fixed seeds, deterministic policy)
    best_model.zip + best_vecnormalize.pkl    checkpoint with the best validation score
    final_model.zip + final_vecnormalize.pkl
    tb/                    tensorboard logs  (tensorboard --logdir runs)
"""
# NOTE: torch / SB3 are imported inside main(). The env workers are spawned processes that
# re-import this file, and they shouldn't load torch just to run the simulator.
import os

# numpy's OpenBLAS reserves a buffer per CPU thread in every process (~650 MB of committed memory
# per process on my 20-thread laptop). 16 env workers x 650 MB is how the first runs crashed with
# out-of-memory. The simulator only does tiny matrix ops, so 1 BLAS thread per process is plenty.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import functools  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from evals.scenarios import make_env, scenario_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_ppo.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--timesteps", type=int, default=None, help="override total_timesteps (for quick tests)")
    ap.add_argument("--seed", type=int, default=None, help="override the training seed")
    args = ap.parse_args()

    import torch
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    from rl.callbacks import ValidationCallback
    from rl.vec_env import LightSubprocVecEnv

    cfg = yaml.safe_load(open(args.config))
    if args.timesteps:
        cfg["total_timesteps"] = args.timesteps
    if args.seed is not None:
        cfg["seed"] = args.seed
    run_dir = Path("runs") / (args.run_name or f"{cfg['algo']}_seed{cfg['seed']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "train_config.yaml", "w"), sort_keys=False)
    yaml.safe_dump({s: scenario_config(s) for s in cfg["scenarios"]}, open(run_dir / "env_configs.yaml", "w"))

    seed = int(cfg["seed"])
    # small MLPs on CPU: a few threads is plenty
    torch.set_num_threads(int(cfg.get("torch_threads", 4)))
    # the MultiDiscrete head builds 19 Categorical distributions per forward pass, argument
    # validation on each of them was ~30% of its time (actions are always valid anyway)
    torch.distributions.Distribution.set_default_validate_args(False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env_fn = functools.partial(make_env, cfg["scenarios"])
    venv = LightSubprocVecEnv([env_fn] * cfg["n_envs"])
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, gamma=cfg["hyperparams"]["gamma"], **cfg["vec_normalize"])

    hp = dict(cfg["hyperparams"])
    if cfg.get("linear_lr"):
        lr0 = hp["learning_rate"]
        hp["learning_rate"] = lambda progress_remaining: lr0 * progress_remaining
    algo = {"ppo": PPO, "a2c": A2C}[cfg["algo"]]
    model = algo("MlpPolicy", venv, seed=seed, device="cpu", verbose=0,
                 tensorboard_log=str(run_dir / "tb"), policy_kwargs=cfg["policy_kwargs"], **hp)

    val = cfg["validation"]
    callback = ValidationCallback(cfg["scenarios"], val["seeds"], val["every"], run_dir)
    print(f"training {cfg['algo']} (seed {seed}) for {cfg['total_timesteps']:,} steps on {cfg['scenarios']} "
          f"-> {run_dir}", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=cfg["total_timesteps"], callback=callback, tb_log_name=cfg["algo"])
    callback.validate()  # one last time at the very end
    model.save(run_dir / "final_model")
    venv.save(str(run_dir / "final_vecnormalize.pkl"))
    venv.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min, best validation score {callback.best:,.0f}")


if __name__ == "__main__":
    main()
