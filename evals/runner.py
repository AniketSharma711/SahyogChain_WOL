"""Runs agents on scenarios and collects per-episode metrics (+ optional trajectories)."""
import time
from pathlib import Path

import pandas as pd

from evals.metrics import compute_metrics
from evals.scenarios import make_env

TEST_SEED_OFFSET = 10_000  # test episodes use seeds 10000, 10001, ... (training never resets with these)


def run_episode(env, agent, seed, agent_seed=None):
    obs, info = env.reset(seed=seed)
    agent.reset(env, seed=seed if agent_seed is None else agent_seed)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs))
        done = terminated or truncated
    return env.unwrapped.tracker


def evaluate(agents, scenarios, n_episodes=10, seed_offset=TEST_SEED_OFFSET, save_dir=None,
             save_trajectories=2, verbose=True):
    """agents: dict name -> Agent. Every agent sees the exact same episodes (same seeds).

    Returns a DataFrame with one row per (scenario, agent, episode). If save_dir is given,
    the daily logs / orders / shipments of the first `save_trajectories` episodes are saved
    as CSV for the dashboard.
    """
    rows = []
    save_dir = Path(save_dir) if save_dir else None
    for scen in scenarios:
        env = make_env(scen, record_shipments=save_dir is not None)
        for name, agent in agents.items():
            t0 = time.time()
            for ep in range(n_episodes):
                seed = seed_offset + ep
                tracker = run_episode(env, agent, seed)
                row = {"scenario": scen, "agent": name, "episode": ep, "seed": seed}
                row.update(compute_metrics(tracker))
                rows.append(row)
                if save_dir is not None and ep < save_trajectories:
                    save_trajectory(tracker, save_dir / "trajectories" / scen / name / f"ep{ep}")
            if verbose:
                sub = pd.DataFrame([r for r in rows if r["scenario"] == scen and r["agent"] == name])
                print(f"  {scen:<22} {name:<12} score {sub['score'].mean():>11,.0f}  "
                      f"fill {sub['fill_rate'].mean():.3f}  otif {sub['otif_rate'].mean():.3f}  "
                      f"({time.time() - t0:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def save_trajectory(tracker, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tracker.daily_frame().to_csv(folder / "daily.csv", index=False)
    tracker.orders_frame().to_csv(folder / "orders.csv", index=False)
    if tracker.shipments is not None:
        tracker.shipments_frame().to_csv(folder / "shipments.csv", index=False)
    pd.DataFrame(tracker.events_log, columns=["day", "event"]).to_csv(folder / "events.csv", index=False)
