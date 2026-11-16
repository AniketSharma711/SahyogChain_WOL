"""Timeline of a day: lead times, spoilage, damage, supplier capacity, termination."""
import numpy as np
import pytest

from scsim.supplier import DOUBLE, OFF, SINGLE
from tests.helpers import NO_DEMAND, empty_env, idle_action

MILK, CEREAL, BLENDER = 0, 1, 2
NORTH, CENTRAL, SOUTH = 0, 1, 2


def order_action(env, w, p, lots, **kw):
    lots_arr = np.zeros((env.params.W, env.params.P), dtype=int)
    lots_arr[w, p] = lots
    prod = np.zeros(env.params.P, dtype=int)
    prod[p] = SINGLE
    return idle_action(env, order_lots=lots_arr, production=prod, **kw)


@pytest.mark.parametrize("w", [NORTH, CENTRAL, SOUTH])
@pytest.mark.parametrize("expedite", [False, True])
def test_order_arrives_after_lead_time(w, expedite):
    env = empty_env(NO_DEMAND)
    L = env.params.exp_lead_time[w] if expedite else env.params.lead_time[w]
    env.step(order_action(env, w, CEREAL, 1, expedite=expedite))  # day 0: ordered, produced, shipped
    for day in range(1, 10):
        # after step (day - 1) the env holds the stock of the morning of `day`
        assert env.inv.total[w, CEREAL] == (25 if day >= L else 0), day
        env.step(idle_action(env))


def test_daily_log_is_end_of_day_before_deliveries():
    env = empty_env(NO_DEMAND)
    env.step(order_action(env, NORTH, CEREAL, 1))  # arrives on the morning of day 2
    env.step(idle_action(env))                     # day 1
    assert env.tracker.days[1]["on_hand"][NORTH, CEREAL] == 0   # empty shelf at the end of day 1
    assert env.tracker.days[1]["inbound"][NORTH, CEREAL] == 25
    assert env.inv.total[NORTH, CEREAL] == 25                   # ...unloaded overnight


def test_milk_spoils_after_shelf_life():
    env = empty_env(NO_DEMAND)
    env.step(order_action(env, NORTH, MILK, 1))  # produced day 0 -> sellable up to day 11
    S = env.params.shelf_life[MILK]
    while env.t < S:
        env.step(idle_action(env))
    expired = [int(np.sum(d["expired"])) for d in env.tracker.days]
    assert expired[S - 1] == 40 and sum(expired) == 40
    assert env.inv.total.sum() == 0


def test_spoiled_in_transit_when_warehouse_closed_too_long():
    env = empty_env(NO_DEMAND)
    env.events.wh_up[1:20, NORTH] = False   # closed until day 19, milk expires on day 11
    env.step(order_action(env, NORTH, MILK, 1))
    while env.t < 20:
        env.step(idle_action(env))
    assert sum(int(np.sum(d["spoiled_in_transit"])) for d in env.tracker.days) == 40
    assert env.inv.total.sum() == 0 and env.pipeline.qty.sum() == 0


def test_damaged_delivery():
    env = empty_env(NO_DEMAND)
    env.events.damage_frac[2, NORTH] = 0.25
    env.step(order_action(env, NORTH, CEREAL, 2))  # 50 units, arrive on day 2
    env.step(idle_action(env))
    assert env.inv.total[NORTH, CEREAL] == 50 - int(50 * 0.25)
    assert int(np.sum(env.tracker.days[-1]["damaged"])) == int(50 * 0.25)


def test_supplier_capacity_and_fifo():
    env = empty_env(NO_DEMAND)
    sup, p = env.supplier, env.params
    a = sup.place_order(NORTH, MILK, 80, expedite=False, day=0)
    b = sup.place_order(SOUTH, MILK, 80, expedite=False, day=0)
    assert a == b == 80
    levels = np.array([SINGLE, OFF, OFF])
    dispatches, produced, paid = sup.produce(levels, np.ones(p.P))
    cap = int(p.sup_capacity[MILK])
    assert produced[MILK] == cap
    assert dispatches == [(NORTH, MILK, 80, False), (SOUTH, MILK, cap - 80, False)]
    assert sup.queued[SOUTH, MILK] == 160 - cap
    # disruption: 40% capacity on a double shift
    _, produced, paid = sup.produce(np.array([DOUBLE, OFF, OFF]), np.array([0.4, 1, 1]))
    assert produced[MILK] == min(160 - cap, int(np.floor(cap * 1.6 * 0.4 + 1e-9)))
    assert paid[MILK] == pytest.approx(cap * 1.6)


def test_supplier_rejects_beyond_order_book():
    env = empty_env(NO_DEMAND)
    sup = env.supplier
    limit = int(sup.book_limit[CEREAL])
    assert sup.place_order(NORTH, CEREAL, limit - 10, False, 0) == limit - 10
    assert sup.place_order(SOUTH, CEREAL, 25, False, 0) == 10
    assert sup.place_order(SOUTH, CEREAL, 25, False, 0) == 0


def test_shift_cost_charged_even_without_orders():
    env = empty_env(NO_DEMAND)
    p = env.params
    env.step(idle_action(env, production=np.array([DOUBLE, SINGLE, OFF])))
    assert env.tracker.days[0]["shift"] == pytest.approx(p.shift_cost[0, 2] + p.shift_cost[1, 1])


def test_transfer_moves_stock_fefo():
    env = empty_env(NO_DEMAND)
    env.inv.add(NORTH, MILK, 30, expiry=5)
    env.inv.add(NORTH, MILK, 30, expiry=9)
    code = env.action_layout.transfer_code(NORTH, CENTRAL)
    env.step(idle_action(env, transfers=np.array([code, 0, 0])))
    assert env.inv.total[NORTH, MILK] == 20       # one lot of 40 left, oldest first
    assert env.inv.qty[NORTH, MILK, 9] == 20
    days = env.logistics.transfer_days[NORTH, CENTRAL]
    while env.t < days:
        env.step(idle_action(env))
    assert env.inv.qty[CENTRAL, MILK, 5] == 30 and env.inv.qty[CENTRAL, MILK, 9] == 10


def test_insolvency_terminates_with_penalty():
    env = empty_env(NO_DEMAND, {"finance": {"initial_cash": 1000, "credit_limit": 0}})
    p = env.params
    # 20 blenders ordered, 12 made on day 0 (single shift) -> ~$1250 paid out with $1000 in the bank
    _, r, term, trunc, info = env.step(order_action(env, SOUTH, BLENDER, 4))
    assert term and not trunc and info["termination_reason"] == "insolvent"
    assert env.tracker.days[0]["termination"] == pytest.approx(p.term_penalty_per_day * (p.horizon - 1))
    assert r * p.reward_scale < -p.term_penalty_per_day * (p.horizon - 2)


def test_inventory_collapse():
    env = empty_env()  # normal demand but nothing in stock and we never order
    done, info = False, {}
    while not done:
        _, _, term, trunc, info = env.step(idle_action(env))
        done = term or trunc
    assert term and info["termination_reason"] == "inventory_collapse"
    assert env.t < env.params.horizon


def test_termination_penalty_beats_limping_on():
    """Terminating early must never be better than the worst plausible day of continuing."""
    env = empty_env()
    p = env.params
    worst_day = p.lost_penalty * (p.price * p.base_demand.sum(axis=0)).sum() * 3  # 3x demand peak
    assert p.term_penalty_per_day > worst_day
