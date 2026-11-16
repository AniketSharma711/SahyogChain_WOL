"""
Watch one episode in the terminal.

    python scripts/demo.py --agent forecast --scenario warehouse_failure
    python scripts/demo.py --agent ppo --run runs/ppo_main --scenario peak_season --every 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import BASELINES  # noqa: E402
from evals.metrics import compute_metrics  # noqa: E402
from evals.scenarios import ALL_SCENARIOS, make_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="forecast", help=f"one of {list(BASELINES)} or an RL name with --run")
    ap.add_argument("--run", default=None, help="training run folder for RL agents")
    ap.add_argument("--scenario", default="normal", choices=ALL_SCENARIOS)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--every", type=int, default=1, help="print every n-th day")
    args = ap.parse_args()

    if args.run:
        from agents.rl_agent import RLAgent
        agent = RLAgent(args.run, name=args.agent)
    else:
        agent = BASELINES[args.agent]()

    env = make_env(args.scenario, render_mode="ansi")
    obs, _ = env.reset(seed=args.seed)
    agent.reset(env, seed=args.seed)
    events = dict()
    for day, text in env.events.log:
        events.setdefault(day, []).append(text)

    done = False
    while not done:
        day = env.t
        obs, reward, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
        if day % args.every == 0 or done or day in events:
            for text in events.get(day, []):
                print(f"  !! {text}")
            print(env.render(), "\n")

    m = compute_metrics(env.tracker)
    print(f"score ${m['score']:,.0f} | profit ${m['net_profit']:,.0f} | fill rate {m['fill_rate']:.1%} | "
          f"on time {m['otif_rate']:.1%} | waste {m['waste_units']:.0f} units")


if __name__ == "__main__":
    main()
