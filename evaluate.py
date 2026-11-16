"""
Benchmark agents on the test scenarios.

    python evaluate.py                                    # all baselines, all scenarios
    python evaluate.py --rl ppo=runs/ppo_main a2c=runs/a2c_main --episodes 20

Writes to results/<name>/:
    episodes.csv     one row per (scenario, agent, episode) with every metric
    summary.csv      means per (scenario, agent)
    leaderboard.md   human readable tables
    trajectories/    daily logs of the first few episodes (for the dashboard)
"""
import argparse
import json
import platform
import time
from pathlib import Path

from agents import BASELINES, make_agent
from evals.leaderboard import summarize, write_report
from evals.runner import TEST_SEED_OFFSET, evaluate
from evals.scenarios import ALL_SCENARIOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="*", default=list(BASELINES), help="baseline names")
    ap.add_argument("--rl", nargs="*", default=[], help="name=run_dir pairs, e.g. ppo=runs/ppo_main")
    ap.add_argument("--checkpoint", default="best", choices=["best", "final"])
    ap.add_argument("--scenarios", nargs="*", default=ALL_SCENARIOS)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--trajectories", type=int, default=2, help="episodes per (scenario, agent) to save in full")
    ap.add_argument("--out", default="results/benchmark")
    args = ap.parse_args()

    agents = {name: make_agent(name) for name in args.agents}
    rl_names = []
    for spec in args.rl:
        name, path = spec.split("=", 1)
        from agents.rl_agent import RLAgent
        agents[name] = RLAgent(path, name=name, checkpoint=args.checkpoint)
        rl_names.append(name)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"evaluating {list(agents)} on {args.scenarios}, {args.episodes} episodes each")
    df = evaluate(agents, args.scenarios, n_episodes=args.episodes, save_dir=out,
                  save_trajectories=args.trajectories)

    df.to_csv(out / "episodes.csv", index=False)
    summarize(df).to_csv(out / "summary.csv", index=False)
    write_report(df, out / "leaderboard.md", rl_agents=rl_names)
    meta = {"agents": list(agents), "rl_runs": dict(s.split("=", 1) for s in args.rl),
            "checkpoint": args.checkpoint, "scenarios": args.scenarios, "episodes": args.episodes,
            "seeds": [TEST_SEED_OFFSET, TEST_SEED_OFFSET + args.episodes - 1],
            "python": platform.python_version(), "minutes": round((time.time() - t0) / 60, 2)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}/leaderboard.md ({meta['minutes']} min)")
    print((out / "leaderboard.md").read_text(encoding="utf-8").split("\n## ")[0])


if __name__ == "__main__":
    main()
