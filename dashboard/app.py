"""
Results dashboard.

    streamlit run dashboard/app.py

Reads the folders written by evaluate.py (results/<name>/episodes.csv + trajectories/).
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals.leaderboard import family, normalized_scores, overall_table, summarize  # noqa: E402
from scsim.config import load_config  # noqa: E402

RESULTS = ROOT / "results"
TRUCK_CAPACITY = load_config()["logistics"]["truck"]["capacity"]  # same in every scenario
MONEY = ["cogs", "waste_value", "disposal", "shift", "inbound_freight", "transfer", "truck", "express",
         "handling", "holding", "overflow", "late_penalty", "lost_penalty", "termination"]

st.set_page_config(page_title="Supply chain RL - results", layout="wide")


@st.cache_data
def load_episodes(folder):
    return pd.read_csv(Path(folder) / "episodes.csv", keep_default_na=False, na_values=[""])


@st.cache_data
def load_traj(folder, scenario, agent, ep, kind):
    path = Path(folder) / "trajectories" / scenario / agent / ep / f"{kind}.csv"
    return pd.read_csv(path) if path.exists() else None


def columns(df, prefix):
    """'on_hand|North|fresh_milk' style columns -> {('North', 'fresh_milk'): column}"""
    out = {}
    for c in df.columns:
        parts = c.split("|")
        if parts[0] == prefix and len(parts) > 1:
            out[tuple(parts[1:])] = c
    return out


folders = sorted(p.parent for p in RESULTS.glob("*/episodes.csv"))
if not folders:
    st.error("No results yet - run `python evaluate.py` first.")
    st.stop()

st.sidebar.title("Supply chain RL")
folder = st.sidebar.selectbox("Results", folders, format_func=lambda p: p.name)
df = load_episodes(str(folder))
scenarios = list(dict.fromkeys(df["scenario"]))
agents = list(dict.fromkeys(df["agent"]))
scenario = st.sidebar.selectbox("Scenario", scenarios)
first_ppo = next((a for a in agents if family(a) == "ppo"), None)
default = [a for a in agents if a in ("forecast", "rule_based", first_ppo)] or agents[:2]
chosen = st.sidebar.multiselect("Agents", agents, default=default)
traj_dir = folder / "trajectories" / scenario
episodes = sorted({p.name for a in chosen for p in (traj_dir / a).glob("ep*")}) if traj_dir.exists() else []
episode = st.sidebar.selectbox("Episode (saved trajectories)", episodes) if episodes else None

tabs = st.tabs(["Leaderboard", "Inventory", "Demand", "Service", "Costs", "Logistics"])

# ---------------------------------------------------------------- leaderboard
with tabs[0]:
    st.subheader("Overall")
    st.caption("Mean score in $1000 (profit minus service penalties). norm_score: 0 = random, 100 = forecast heuristic.")
    if {"random", "forecast"} <= set(agents):
        st.dataframe(overall_table(df).style.format("{:,.1f}"), width="stretch")
        norm = normalized_scores(df).reset_index().melt(id_vars="scenario", var_name="agent", value_name="norm")
        fig = px.bar(norm, x="scenario", y="norm", color="agent", barmode="group",
                     labels={"norm": "normalised score"}, category_orders={"scenario": scenarios})
        fig.add_hline(y=100, line_dash="dot", annotation_text="forecast heuristic")
        st.plotly_chart(fig, width="stretch")
    st.subheader(f"KPIs - {scenario}")
    summ = summarize(df[df.scenario == scenario]).drop(columns="scenario").set_index("agent")
    st.dataframe(summ.style.format("{:,.3f}"), width="stretch")
    fig = px.box(df[df.scenario == scenario], x="agent", y="score", points="all",
                 title="Score per episode (same seeds for every agent)")
    st.plotly_chart(fig, width="stretch")

if episode is None:
    for t in tabs[1:]:
        with t:
            st.info("No saved trajectories for this selection.")
    st.stop()

daily = {a: load_traj(str(folder), scenario, a, episode, "daily") for a in chosen}
daily = {a: d for a, d in daily.items() if d is not None}
events = next((load_traj(str(folder), scenario, a, episode, "events") for a in chosen), None)


def shade_events(fig):
    if events is not None:
        for _, e in events[events.event.str.startswith("[scheduled")].iterrows():
            fig.add_vline(x=e.day, line_dash="dot", line_color="gray")
    return fig


# ------------------------------------------------------------------ inventory
with tabs[1]:
    any_daily = next(iter(daily.values()))
    whs = [k[0] for k in columns(any_daily, "utilization")]
    prods = list(dict.fromkeys(k[1] for k in columns(any_daily, "on_hand")))
    prod = st.selectbox("Product", ["all (volume-weighted utilisation)"] + prods)
    for w in whs:
        fig = go.Figure()
        for a, d in daily.items():
            if prod in prods:
                fig.add_scatter(x=d.day, y=d[f"on_hand|{w}|{prod}"], name=f"{a} on hand", mode="lines")
                fig.add_scatter(x=d.day, y=d[f"inbound|{w}|{prod}"], name=f"{a} in transit",
                                mode="lines", line=dict(dash="dot"))
            else:
                fig.add_scatter(x=d.day, y=d[f"utilization|{w}"], name=a, mode="lines")
        if prod not in prods:
            fig.add_hline(y=1.0, line_color="red", annotation_text="capacity")
        down = any_daily.day[any_daily[f"wh_up|{w}"] == 0]
        for x in down:
            fig.add_vrect(x0=x - 0.5, x1=x + 0.5, fillcolor="red", opacity=0.08, line_width=0)
        fig.update_layout(title=f"{w}" + (" (red = warehouse closed)" if len(down) else ""), height=300,
                          margin=dict(t=40, b=20))
        st.plotly_chart(shade_events(fig), width="stretch")

# --------------------------------------------------------------------- demand
with tabs[2]:
    d0 = next(iter(daily.values()))  # demand is identical for every agent (same seed)
    for prod in prods:
        dem = sum(d0[c] for (r, p), c in columns(d0, "demand").items() if p == prod)
        exp = sum(d0[c] for (r, p), c in columns(d0, "expected_demand").items() if p == prod)
        fc = sum(d0[c] for (r, p), c in columns(d0, "forecast").items() if p == prod)
        fig = go.Figure()
        fig.add_bar(x=d0.day, y=dem, name="orders (units)", opacity=0.5)
        fig.add_scatter(x=d0.day, y=exp, name="expected", mode="lines")
        fig.add_scatter(x=d0.day, y=fc, name="7-day forecast", mode="lines", line=dict(dash="dash"))
        fig.update_layout(title=f"{prod} - network demand", height=280, margin=dict(t=40, b=20))
        st.plotly_chart(shade_events(fig), width="stretch")
    if events is not None and len(events):
        st.subheader("Events this episode")
        st.dataframe(events, width="stretch", hide_index=True)

# -------------------------------------------------------------------- service
with tabs[3]:
    fig = go.Figure()
    for a, d in daily.items():
        backlog = sum(d[c] for c in columns(d, "backlog").values())
        fig.add_scatter(x=d.day, y=backlog, name=a, mode="lines")
    fig.update_layout(title="Open backlog (units)", height=300)
    st.plotly_chart(shade_events(fig), width="stretch")
    fig = go.Figure()
    for a, d in daily.items():
        fig.add_scatter(x=d.day, y=sum(d[c] for c in columns(d, "lost").values()).cumsum(), name=a)
    fig.update_layout(title="Cumulative lost sales (units)", height=300)
    st.plotly_chart(shade_events(fig), width="stretch")
    rows = []
    for a in daily:
        o = load_traj(str(folder), scenario, a, episode, "orders")
        f = o[o.status == "fulfilled"]
        late = (f.last_delivery - f.due).clip(lower=0)
        rows.append(pd.DataFrame({"agent": a, "days_late": late}))
    lat = pd.concat(rows)
    fig = px.histogram(lat, x="days_late", color="agent", barmode="group", title="Delivery delay of fulfilled orders")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------- costs
with tabs[4]:
    tot = pd.DataFrame({a: d[MONEY].sum() for a, d in daily.items()}).T / 1000
    tot = tot.drop(columns="cogs")  # cost of goods sold is ~ the same for everyone and dwarfs the rest
    tot = tot.loc[:, tot.abs().sum() > 0]
    fig = px.bar(tot, barmode="stack", labels={"value": "$1000", "index": "agent", "variable": "cost"},
                 title="Where the money goes this episode (excluding cost of goods sold)")
    st.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    for a, d in daily.items():
        fig.add_scatter(x=d.day, y=d.reward.cumsum() / 1000, name=a)
    fig.update_layout(title="Cumulative reward ($1000)", height=320)
    st.plotly_chart(shade_events(fig), width="stretch")

# ------------------------------------------------------------------ logistics
with tabs[5]:
    a = st.selectbox("Agent", list(daily))
    ships = load_traj(str(folder), scenario, a, episode, "shipments")
    if ships is not None and len(ships):
        names = [w for w in whs]
        ships["lane"] = ships.warehouse.map(dict(enumerate(names))) + " -> " + ships.region.map(dict(enumerate(names)))
        lanes = ships.groupby(["day", "lane", "carrier"], as_index=False).qty.sum()
        fig = px.scatter(lanes, x="day", y="lane", size="qty", color="carrier",
                         title="Shipment timeline (bubble = units shipped that day)")
        st.plotly_chart(shade_events(fig), width="stretch")
    d = daily[a]
    c1, c2 = st.columns(2)
    with c1:
        util = d.truck_volume / (d.trucks.where(d.trucks > 0) * TRUCK_CAPACITY)
        fig = px.line(x=d.day, y=util, title="Truck utilisation (own fleet)", labels={"x": "day", "y": "load / capacity"})
        st.plotly_chart(fig, width="stretch")
    with c2:
        prod_cols = columns(d, "levels")
        lv = pd.DataFrame({p[0]: d[c] for p, c in prod_cols.items()})
        fig = px.line(lv, title="Production shift level (0 off, 1 single, 2 double)")
        st.plotly_chart(fig, width="stretch")
