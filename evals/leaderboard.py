"""
Turns the per-episode results into leaderboard tables.

Normalised score per scenario (Atari style):  100 * (agent - random) / (forecast - random)
so 0 = random policy, 100 = the forecast heuristic, >100 = better than the best hand-written baseline.

Because every agent is evaluated on the same seeds (= same customers and disruptions), we can
also do paired comparisons, which have much tighter confidence intervals than comparing means.
"""
import re

import numpy as np
import pandas as pd

SUMMARY_METRICS = ["score", "net_profit", "service_penalty", "revenue", "fill_rate", "otif_rate",
                   "high_priority_otif", "avg_delay_days", "csat", "inventory_turnover", "stockout_rate",
                   "overstock_rate", "waste_units", "holding_cost", "shipping_cost", "shift_cost", "production_cost",
                   "warehouse_utilization", "vehicle_utilization", "express_share", "co2_kg"]

# 97.5% quantiles of the t distribution, for 95% confidence intervals with few episodes
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
         2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
         2.052, 2.048, 2.045, 2.042]


def t975(dof):
    if dof < 1:
        return float("nan")
    return _T975[dof - 1] if dof <= len(_T975) else (2.00 if dof <= 60 else 1.98 if dof <= 120 else 1.96)


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return t975(len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def summarize(df, metrics=SUMMARY_METRICS):
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby(["scenario", "agent"], sort=False)
    out = g[metrics].mean()
    out["score_ci95"] = g["score"].apply(ci95)
    out["episodes"] = g.size()
    out["early_terminations"] = g["terminated"].apply(lambda s: int((s != "").sum()))
    return out.reset_index()


def normalized_scores(df, low="random", high="forecast"):
    means = df.groupby(["scenario", "agent"])["score"].mean().unstack("agent")
    return 100 * means.sub(means[low], axis=0).div(means[high] - means[low], axis=0)


def paired_comparison(df, a, b):
    """Per scenario: mean of (score_a - score_b) over the same seeds, its 95% CI, and how often a wins."""
    piv = df.pivot_table(index=["scenario", "seed"], columns="agent", values="score")
    # a can also be a list of agents (training seeds of one algorithm) -> their mean score per episode
    left = piv[list(a)].mean(axis=1) if isinstance(a, (list, tuple)) else piv[a]
    diff = (left - piv[b]).rename("diff").reset_index()
    rows = []
    for scen, d in diff.groupby("scenario", sort=False):
        rows.append({"scenario": scen, "mean_diff": d["diff"].mean(), "ci95": ci95(d["diff"]),
                     "win_rate": (d["diff"] > 0).mean(), "episodes": len(d)})
    return pd.DataFrame(rows)


def family(agent):
    """'ppo_s1' / 'ppo_seed1' -> 'ppo': training seeds of the same algorithm."""
    m = re.fullmatch(r"(.+?)_s(?:eed)?\d+", agent)
    return m.group(1) if m else agent


def overall_table(df, low="random", high="forecast"):
    """agents x scenarios: mean score ($k), plus mean normalised score and average rank.
    If an algorithm was trained with several seeds (ppo_s0, ppo_s1, ...) there's an extra
    row with the mean over the seeds."""
    scen_order = list(dict.fromkeys(df["scenario"]))
    means = df.groupby(["agent", "scenario"])["score"].mean().unstack("scenario")[scen_order]
    table = means / 1000.0
    norm = normalized_scores(df, low, high)
    table["norm_score"] = norm.T.mean(axis=1)
    table["avg_rank"] = means.rank(ascending=False).mean(axis=1)
    groups = {}
    for a in table.index:
        groups.setdefault(family(a), []).append(a)
    for fam, members in groups.items():
        if len(members) > 1:
            row = table.loc[members].mean()
            row["avg_rank"] = np.nan  # ranks of the individual seeds don't average into anything meaningful
            table.loc[f"{fam} (mean of {len(members)} seeds)"] = row
    return table.sort_values("norm_score", ascending=False)


def to_markdown(df, floatfmt="{:,.1f}", index=True):
    """Tiny markdown table writer (didn't want to add tabulate as a dependency)."""
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for idx, *values in df.itertuples(index=True, name=None):  # itertuples keeps ints as ints
        cells = [str(idx)] if index else []
        for v in values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(df, path, rl_agents=(), low="random", high="forecast"):
    """Markdown leaderboard with the overall table, per-scenario KPIs and paired comparisons."""
    out = ["# Leaderboard", "",
           f"{df['episode'].nunique()} test episodes per (scenario, agent), same seeds for every agent.",
           f"Scores in $1000 (= profit - service penalties). norm_score: 0 = {low}, 100 = {high}.", ""]
    out.append(to_markdown(overall_table(df, low, high)))

    summ = summarize(df)
    cols = ["agent", "score", "score_ci95", "net_profit", "fill_rate", "otif_rate", "avg_delay_days",
            "csat", "inventory_turnover", "stockout_rate", "waste_units", "shipping_cost", "co2_kg"]
    fmt = summ.copy()
    for c in ["score", "score_ci95", "net_profit", "shipping_cost"]:
        fmt[c] = fmt[c] / 1000.0
    for scen, part in fmt.groupby("scenario", sort=False):
        out += ["", f"## {scen}", "", "(money columns in $1000)", ""]
        out.append(to_markdown(part[cols].set_index("agent").sort_values("score", ascending=False),
                               floatfmt="{:,.3f}"))

    behaviour = [c for c in df.columns if c.startswith("share_")] + ["ordered_units_per_day"]
    if all(c in df.columns for c in behaviour):
        beh = df.groupby("agent", sort=False)[behaviour].mean()
        beh.columns = [c.replace("share_", "") for c in beh.columns]
        out += ["", "## What the policies actually do (average over all scenarios)", "",
                "Share of days (or production-line-days) where the policy picked each option.", ""]
        out.append(to_markdown(beh, floatfmt="{:,.2f}"))

    groups = {}
    for a in rl_agents:
        groups.setdefault(family(a), []).append(a)
    comparisons = [(a, a) for a in rl_agents]
    comparisons += [(f"{fam} (mean of {len(m)} seeds)", m) for fam, m in groups.items() if len(m) > 1]
    for title, a in comparisons:
        comp = paired_comparison(df, a, high).set_index("scenario")
        comp[["mean_diff", "ci95"]] /= 1000.0
        out += ["", f"## Paired comparison: {title} vs {high}", "",
                "mean_diff and ci95 in $1000 per episode, win_rate = share of test episodes where "
                f"{title} scored higher.", ""]
        out.append(to_markdown(comp, floatfmt="{:,.2f}"))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
