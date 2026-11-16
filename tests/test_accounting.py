"""The money and the units have to add up, whatever the policy does."""
import numpy as np
import pytest

from agents import BASELINES
from evals.metrics import compute_metrics
from evals.scenarios import ALL_SCENARIOS, make_env


def run(scenario, agent_name, seed):
    env = make_env(scenario, record_shipments=True)
    obs, _ = env.reset(seed=seed)
    begin_units = (env.inv.total + env.pipeline.qty).copy()
    agent = BASELINES[agent_name]()
    agent.reset(env, seed=seed)
    rewards, done = [], False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs))
        rewards.append(r)
        assert (env.inv.qty >= 0).all() and (env.inv.total >= 0).all()
        np.testing.assert_array_equal(env.inv.total, env.inv.qty.sum(axis=2))
        assert env.inv.qty[:, :, :min(env.t, env.params.horizon)].sum() == 0  # no expired stock on shelves
        done = term or trunc
    return env, begin_units, np.array(rewards)


CASES = [(s, a) for s in ALL_SCENARIOS for a in ["random", "forecast"]] + [("normal", "greedy"), ("peak_season", "fcfs")]


@pytest.mark.parametrize("scenario,agent", CASES)
def test_books_balance(scenario, agent):
    env, begin_units, rewards = run(scenario, agent, seed=11)
    tr, p = env.tracker, env.params
    days = tr.days

    def total(key):
        return sum(np.sum(d[key]) for d in days)

    def per_product(key):
        return sum(np.asarray(d[key]).reshape(-1, p.P).sum(axis=0) for d in days)

    # units: start + produced = sold + wasted + end, for every product
    end_units = (env.inv.total + env.pipeline.qty).sum(axis=0)
    produced = sum(np.asarray(d["produced"]) for d in days)
    sold = per_product("shipped")
    wasted = per_product("expired") + per_product("damaged") + per_product("spoiled_in_transit")
    np.testing.assert_array_equal(begin_units.sum(axis=0) + produced, sold + wasted + end_units)

    # value version of the same thing (inventory at unit cost)
    lhs = tr.begin_inventory_value + total("procurement")
    rhs = total("cogs") + total("waste_value") + tr.end_inventory_value
    assert lhs == pytest.approx(rhs, rel=1e-9)

    # reward = score, and score = accrual profit - penalties
    m = compute_metrics(tr)
    assert rewards.sum() * p.reward_scale == pytest.approx(m["score"], rel=1e-9)
    assert m["score"] == pytest.approx(m["net_profit"] - m["service_penalty"] - total("termination"), rel=1e-9)
    # accrual vs cash profit only differ by the change in inventory value
    assert m["net_profit"] - m["cash_profit"] == pytest.approx(
        tr.end_inventory_value - tr.begin_inventory_value, abs=1e-6)
    assert env.cash - p.initial_cash == pytest.approx(m["cash_profit"], abs=1e-6)

    # customer orders
    orders = tr.orders_frame()
    assert orders["oid"].is_unique
    assert orders["qty"].sum() == total("demand")
    assert orders["shipped"].sum() == total("units_shipped") == sold.sum()
    cancelled = orders[orders.status == "cancelled"]
    assert (cancelled.qty - cancelled.shipped).sum() == total("lost")
    assert ((orders.status == "fulfilled") == (orders.shipped == orders.qty)).all()
    assert (orders.shipped <= orders.qty).all()
    open_orders = orders[orders.status == "open"]
    assert len(open_orders) == len(env.book.open)

    # shipment log agrees with the order book
    ships = tr.shipments_frame()
    assert ships["qty"].sum() == sold.sum()
    late = ships.groupby("oid").delivery_day.max()
    fulfilled = orders[orders.status == "fulfilled"].set_index("oid")
    assert (late.loc[fulfilled.index] == fulfilled.last_delivery).all()


def test_metric_ranges():
    env, _, _ = run("peak_season", "forecast", seed=3)
    m = compute_metrics(env.tracker)
    for k in ["fill_rate", "otif_rate", "high_priority_otif", "stockout_rate", "overstock_rate",
              "waste_rate", "warehouse_utilization", "vehicle_utilization", "production_efficiency",
              "express_share", "split_shipment_rate"]:
        assert 0.0 <= m[k] <= 1.0, k
    assert 0 <= m["csat"] <= 100
    assert m["avg_delay_days"] >= 0
    assert m["co2_kg"] > 0 and m["fuel_l"] > 0 and m["packaging_kg"] > 0
