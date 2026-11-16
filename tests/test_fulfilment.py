"""Hand-built situations for the fulfilment rules (allocation, sourcing, carriers, cancellations)."""
import numpy as np
import pytest

from scsim.spaces import (ALLOC_FCFS, ALLOC_PRIORITY, CARRIER_ECONOMY, CARRIER_EXPRESS, CARRIER_SMART,
                          SOURCE_FLEXIBLE, SOURCE_LOCAL)
from tests.helpers import NO_DEMAND, empty_env

MILK, CEREAL, BLENDER = 0, 1, 2
NORTH, CENTRAL, SOUTH = 0, 1, 2


def add_order(env, day, region, product, qty, high):
    env.book.add_day(day, np.array([region]), np.array([product]), np.array([qty]), np.array([high]))
    return env.book.open[-1]


def fulfil(env, t=0, allocation=ALLOC_PRIORITY, sourcing=SOURCE_LOCAL, carrier=CARRIER_ECONOMY):
    return env.book.fulfil(t, env.inv, env.logistics, env.events, allocation, sourcing, carrier)


@pytest.mark.parametrize("allocation,winner", [(ALLOC_FCFS, "old"), (ALLOC_PRIORITY, "vip")])
def test_allocation_rule_decides_who_gets_scarce_stock(allocation, winner):
    env = empty_env(NO_DEMAND)
    env.inv.add(NORTH, MILK, 5, expiry=10)
    old = add_order(env, 0, NORTH, MILK, 5, high=False)
    vip = add_order(env, 0, NORTH, MILK, 5, high=True)
    res = fulfil(env, allocation=allocation)
    assert res.units == 5
    got = {"old": old, "vip": vip}[winner]
    assert got.shipped == 5 and [o.oid for o in res.closed] == [got.oid]


def test_local_sourcing_never_cross_ships():
    env = empty_env(NO_DEMAND)
    env.inv.add(CENTRAL, CEREAL, 50, expiry=40)
    o = add_order(env, 0, NORTH, CEREAL, 10, high=False)
    res = fulfil(env, sourcing=SOURCE_LOCAL)
    assert res.units == 0 and o.remaining == 10


def test_flexible_uses_nearest_and_splits():
    env = empty_env(NO_DEMAND)
    env.inv.add(NORTH, CEREAL, 4, expiry=40)
    env.inv.add(CENTRAL, CEREAL, 3, expiry=40)   # 300 km from North
    env.inv.add(SOUTH, CEREAL, 50, expiry=40)    # 620 km from North
    o = add_order(env, 0, NORTH, CEREAL, 10, high=False)
    res = fulfil(env, sourcing=SOURCE_FLEXIBLE)
    assert o.remaining == 0 and o.n_shipments == 3
    assert res.shipped[:, CEREAL].tolist() == [4, 3, 3]
    assert o.wh_mask == 0b111


def test_flexible_serves_local_customers_first():
    """Pass 1 serves every order from its home warehouse before anything is cross-shipped."""
    env = empty_env(NO_DEMAND)
    env.inv.add(CENTRAL, MILK, 10, expiry=10)
    north_vip = add_order(env, 0, NORTH, MILK, 10, high=True)   # sorted first, but North is empty
    central = add_order(env, 0, CENTRAL, MILK, 10, high=False)
    fulfil(env, sourcing=SOURCE_FLEXIBLE)
    assert central.shipped == 10 and north_vip.shipped == 0


def test_closed_warehouse_does_not_ship():
    env = empty_env(NO_DEMAND)
    env.inv.add(NORTH, MILK, 20, expiry=10)
    env.inv.add(CENTRAL, MILK, 20, expiry=10)
    env.events.wh_up[0, NORTH] = False
    o = add_order(env, 0, NORTH, MILK, 8, high=False)
    assert fulfil(env, sourcing=SOURCE_LOCAL).units == 0
    res = fulfil(env, sourcing=SOURCE_FLEXIBLE)
    assert res.shipped[CENTRAL, MILK] == 8 and o.remaining == 0


@pytest.mark.parametrize("carrier,high,expect_express", [
    (CARRIER_ECONOMY, True, False),
    (CARRIER_SMART, True, True),     # truck 2 days > due 1 day, express makes it in 1
    (CARRIER_SMART, False, False),   # standard order due in 3 days, truck is fine
    (CARRIER_EXPRESS, False, True),
])
def test_carrier_choice(carrier, high, expect_express):
    env = empty_env(NO_DEMAND)
    env.inv.add(NORTH, CEREAL, 10, expiry=40)
    o = add_order(env, 0, CENTRAL, CEREAL, 10, high=high)  # North -> Central is 300 km
    res = fulfil(env, sourcing=SOURCE_FLEXIBLE, carrier=carrier)
    assert (res.express_units == 10) == expect_express
    lg = env.logistics
    expected_days = lg.express_days[NORTH, CENTRAL] if expect_express else lg.truck_days[NORTH, CENTRAL]
    assert o.last_delivery == expected_days


def test_smart_doesnt_pay_express_when_late_anyway():
    env = empty_env(NO_DEMAND)
    env.inv.add(NORTH, CEREAL, 10, expiry=40)
    o = add_order(env, 0, SOUTH, CEREAL, 10, high=True)  # 620 km: express 2 days, due in 1 -> late either way
    res = fulfil(env, sourcing=SOURCE_FLEXIBLE, carrier=CARRIER_SMART)
    assert res.express_units == 0 and o.last_delivery == 3


def test_late_penalty_amount():
    env = empty_env(NO_DEMAND)
    p = env.params
    env.inv.add(NORTH, BLENDER, 2, expiry=env.inv.never)
    add_order(env, 0, SOUTH, BLENDER, 2, high=True)  # truck 3 days, due day 1 -> 2 days late
    res = fulfil(env, sourcing=SOURCE_FLEXIBLE, carrier=CARRIER_ECONOMY)
    expected = p.late_penalty * p.price[BLENDER] * 2 * 2 * p.high_late_mult
    assert res.late_penalty == pytest.approx(expected)
    assert res.late_unit_days == 4


@pytest.mark.parametrize("carrier,expected_day", [(CARRIER_ECONOMY, 1 + 2), (CARRIER_EXPRESS, 1)])
def test_truck_delay_applies_to_trucks_only(carrier, expected_day):
    env = empty_env(NO_DEMAND)
    env.events.truck_delay[0] = 2
    env.inv.add(NORTH, CEREAL, 10, expiry=40)
    o = add_order(env, 0, NORTH, CEREAL, 10, high=False)
    fulfil(env, carrier=carrier)
    assert o.last_delivery == expected_day


def test_cancellation_after_patience():
    env = empty_env(NO_DEMAND)
    p = env.params
    o = add_order(env, 0, NORTH, MILK, 6, high=False)
    for t in range(o.due + p.patience):
        assert env.book.cancel_overdue(t) == []
    cancelled = env.book.cancel_overdue(o.due + p.patience)
    assert cancelled == [o] and env.book.open == []


def test_truck_billing_ltl():
    env = empty_env(NO_DEMAND)
    lg = env.logistics
    trip = lg.truck_trip_cost[NORTH, CENTRAL]
    load = np.zeros((3, 3))
    load[NORTH, CENTRAL] = 160 + 16  # one full truck + 10% of a truck
    trucks, cost, km = lg.bill_trucks(load)
    assert trucks.sum() == 2
    assert cost == pytest.approx(trip * (1 + max(lg.ltl_min, lg.ltl_premium * 0.1)))
    load[NORTH, CENTRAL] = 150  # almost full -> capped at one full trip
    _, cost, _ = lg.bill_trucks(load)
    assert cost == pytest.approx(trip)
