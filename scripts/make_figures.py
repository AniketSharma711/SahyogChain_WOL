"""
Figures for the README (saved to docs/figures/).

    python scripts/make_figures.py --results results/benchmark --runs runs/ppo_seed0 runs/a2c_seed0

Several training seeds of one algorithm (agents named ppo_s0, ppo_s1, ...) are shown as one bar
(mean over seeds, error bars = min / max).
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import BASELINES  # noqa: E402
from evals.leaderboard import family, normalized_scores  # noqa: E402
from evals.metrics import compute_metrics  # noqa: E402
from evals.runner import run_episode  # noqa: E402
from evals.scenarios import make_env  # noqa: E402

OUT = Path("docs/figures")
COLORS = {"random": "#9e9e9e", "greedy": "#bcaaa4", "fcfs": "#90caf9", "rule_based": "#1e88e5",
          "forecast": "#43a047", "ppo": "#e53935", "a2c": "#fb8c00"}


def heuristic_validation_score(name, run):
    """Score of a baseline on exactly the validation episodes the RL run used."""
    cfg = yaml.safe_load(open(Path(run) / "train_config.yaml"))
    scores = []
    for scen in cfg["scenarios"]:
        env = make_env(scen)
        for seed in cfg["validation"]["seeds"]:
            scores.append(compute_metrics(run_episode(env, BASELINES[name](), seed))["score"])
    return float(np.mean(scores))


def learning_curves(runs):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ["forecast", "rule_based"]:
        ref = heuristic_validation_score(name, runs[0])
        ax.axhline(ref / 1000, ls="--", lw=1, color=COLORS[name], label=f"{name} heuristic")
    styles = ["-", "--", ":", "-."]
    seen = {}
    for run in runs:
        v = pd.read_csv(Path(run) / "validation.csv")
        name = Path(run).name
        k = seen[family(name)] = seen.get(family(name), -1) + 1
        ax.plot(v.timesteps / 1e6, v.mean_score / 1000, label=name, color=COLORS.get(family(name)),
                ls=styles[k % len(styles)])
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel("validation score ($1000 / episode)")
    ax.set_title("Validation score during training (6 scenarios x 2 fixed seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=130)


def normalized_bar(df):
    norm = normalized_scores(df)
    fams = {}
    for a in norm.columns:
        fams.setdefault(family(a), []).append(a)
    order = [f for f in COLORS if f in fams and f != "random"]
    scen = list(dict.fromkeys(df.scenario))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.8 / len(order)
    x = np.arange(len(scen))
    lo, hi = 0, 100
    for i, f in enumerate(order):
        vals = norm.loc[scen, fams[f]]
        mean = vals.mean(axis=1)
        yerr, label = None, f
        if vals.shape[1] > 1:
            yerr = [mean - vals.min(axis=1), vals.max(axis=1) - mean]
            label = f"{f} ({vals.shape[1]} seeds)"
        ax.bar(x + i * width - 0.4 + width / 2, mean, width, yerr=yerr, capsize=2, label=label, color=COLORS[f])
        lo, hi = min(lo, np.nanmin(vals.values)), max(hi, np.nanmax(vals.values))
    ax.axhline(100, color="k", lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [s.replace("_", "\n") for s in scen])
    ax.set_ylabel("normalised score")
    ax.set_title("Normalised score per scenario (0 = random, 100 = forecast heuristic)")
    ax.set_ylim(max(-100, lo - 10), hi + 10)
    ax.legend(ncol=len(order), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "normalized_scores.png", dpi=130)


def trajectory(results, scenario, agents, ep="ep0"):
    """Inventory (all products, in volume) per warehouse for a few agents on one episode."""
    base = Path(results) / "trajectories" / scenario
    daily = {a: pd.read_csv(base / a / ep / "daily.csv") for a in agents if (base / a / ep).exists()}
    if not daily:
        return
    any_d = next(iter(daily.values()))
    whs = [c.split("|")[1] for c in any_d.columns if c.startswith("utilization|")]
    fig, axes = plt.subplots(1, len(whs) + 1, figsize=(15, 3.6))
    for ax, w in zip(axes, whs):
        for a, d in daily.items():
            ax.plot(d.day, d[f"utilization|{w}"] * 100, label=a, color=COLORS.get(family(a)))
        for i, day in enumerate(any_d.day[any_d[f"wh_up|{w}"] == 0]):
            ax.axvspan(day - 0.5, day + 0.5, color="red", alpha=0.1, lw=0, label="closed" if i == 0 else None)
        ax.set_title(f"{w} warehouse")
        ax.set_xlabel("day")
        ax.set_ylabel("space used (%)")
        ax.grid(alpha=0.3)
    ax = axes[-1]
    for a, d in daily.items():
        backlog = d[[c for c in d.columns if c.startswith("backlog|")]].sum(axis=1)
        ax.plot(d.day, backlog, label=a, color=COLORS.get(family(a)))
    ax.set_title("open customer backlog")
    ax.set_xlabel("day")
    ax.set_ylabel("units")
    ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{scenario.replace('_', ' ')} - one test episode", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / f"trajectory_{scenario}.png", dpi=130, bbox_inches="tight")


def cost_breakdown(df, scenario):
    """The costs a planner can actually influence (the purchase cost of what's sold is ~ the same for everyone)."""
    sub = df[df.scenario == scenario].assign(family=df.agent.map(family))
    sub = sub.groupby("family").mean(numeric_only=True)  # averages over training seeds too
    agents = [a for a in COLORS if a in sub.index]
    tab = pd.DataFrame({a: {"production shifts": sub.loc[a, "shift_cost"],
                            "holding + overflow": sub.loc[a, "holding_cost"],
                            "outbound shipping": sub.loc[a, "shipping_cost"],
                            "inbound freight": sub.loc[a, "inbound_freight"],
                            "waste": sub.loc[a, "waste_cost"],
                            "late / lost penalties": sub.loc[a, "service_penalty"]} for a in agents}).T / 1000
    ax = tab.plot(kind="bar", stacked=True, figsize=(9.5, 4), colormap="tab20c")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_ylabel("$1000 per episode")
    ax.set_title(f"Controllable costs per episode - {scenario.replace('_', ' ')}")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / f"costs_{scenario}.png", dpi=130)
    plt.close("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/benchmark")
    ap.add_argument("--runs", nargs="*", default=[])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(args.results) / "episodes.csv", keep_default_na=False, na_values=[""])
    if args.runs:
        learning_curves(args.runs)
    normalized_bar(df)
    first_rl = next((a for a in df.agent.unique() if family(a) == "ppo"), None)
    agents = [a for a in ["forecast", first_rl, "rule_based"] if a in set(df.agent)]
    for scen in ["warehouse_failure", "peak_season", "supply_shortage"]:
        trajectory(args.results, scen, agents)
    for scen in ["normal", "peak_season"]:
        cost_breakdown(df, scen)
    print(f"figures written to {OUT}/")


if __name__ == "__main__":
    main()
