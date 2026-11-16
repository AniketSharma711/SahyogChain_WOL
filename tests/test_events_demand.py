import numpy as np
import pytest

from evals.scenarios import ALL_SCENARIOS, list_scenarios, scenario_config
from scsim.config import Params
from scsim.demand import DemandModel
from scsim.events import EventSchedule
from tests.helpers import quiet_config


def schedule(*events, seed=0):
    cfg = quiet_config({"events": {"scheduled": list(events)}})
    return Params(cfg), EventSchedule(Params(cfg), np.random.default_rng(seed))


def test_all_scenario_files_load():
    assert set(ALL_SCENARIOS) == set(list_scenarios())
    for name in ALL_SCENARIOS:
        p = Params(scenario_config(name))
        EventSchedule(p, np.random.default_rng(0))


def test_end_day_is_inclusive():
    _, ev = schedule({"type": "warehouse_outage", "start": 5, "end": 7, "warehouse": 2})
    assert np.flatnonzero(~ev.wh_up[:, 2]).tolist() == [5, 6, 7]
    assert ev.wh_up[:, :2].all()


def test_supplier_and_transport_events():
    _, ev = schedule({"type": "supplier", "start": 3, "end": 4, "capacity_factor": 0.4, "extra_delay": 2,
                      "products": [1]},
                     {"type": "transport", "start": 10, "end": 12, "extra_days": 2, "cost_multiplier": 1.5,
                      "express_available": False})
    assert ev.supplier_factor[3:5, 1].tolist() == [0.4, 0.4] and ev.supplier_factor[:, [0, 2]].min() == 1.0
    assert ev.supplier_delay[3:5].tolist() == [2, 2] and ev.supplier_delay.sum() == 4
    assert ev.truck_delay[10:13].tolist() == [2, 2, 2] and ev.cost_mult[11] == 1.5
    assert not ev.express_ok[10:13].any() and ev.express_ok[13]


def test_demand_events_split_known_and_shock():
    _, ev = schedule({"type": "demand", "start": 2, "end": 3, "multiplier": 2.0, "regions": [0]},
                     {"type": "demand", "start": 2, "end": 2, "multiplier": 1.5, "forecastable": False},
                     {"type": "demand_curve", "knots": [[10, 1.0], [20, 2.0]], "products": [2]})
    assert ev.demand_known[2, 0].tolist() == [2.0, 2.0, 2.0] and ev.demand_known[2, 1, 0] == 1.0
    assert ev.demand_shock[2].min() == 1.5 and ev.demand_shock[3].max() == 1.0
    assert ev.demand_known[15, 1, 2] == pytest.approx(1.5) and ev.demand_known[30, 1, 2] == pytest.approx(2.0)
    assert ev.demand_known[15, 1, 0] == pytest.approx(1.0)


def test_unknown_event_type():
    with pytest.raises(ValueError):
        schedule({"type": "alien_invasion", "start": 1, "end": 2})


def test_random_events_reproducible():
    cfg = quiet_config({"events": {"outage_prob": 0.05, "damage_prob": 0.1, "strike_prob": 0.05}})
    p = Params(cfg)
    a = EventSchedule(p, np.random.default_rng(3))
    b = EventSchedule(p, np.random.default_rng(3))
    np.testing.assert_array_equal(a.wh_up, b.wh_up)
    np.testing.assert_array_equal(a.damage_frac, b.damage_frac)
    assert a.log == b.log and len(a.log) > 0


def test_demand_mean_matches_rate():
    p = Params(quiet_config())
    ev = EventSchedule(p, np.random.default_rng(0))
    units = np.zeros((p.W, p.P))
    n_eps = 60
    for s in range(n_eps):
        dm = DemandModel(p, ev, np.random.default_rng(s))
        for t in range(p.horizon):
            units += dm.units_on_day(t)
    expected = dm.rate_true[:p.horizon].sum(axis=0) * n_eps
    # compound Poisson -> std of the total is sqrt(sum var), allow 4 sigma
    sigma = np.sqrt(expected * ((p.mean_order_size - 1 + p.mean_order_size ** 2) / p.mean_order_size)[None, :])
    assert (np.abs(units - expected) < 4 * sigma).all()


def test_forecast_knows_season_but_not_shocks():
    cfg = quiet_config({"events": {"scheduled": [
        {"type": "demand", "start": 20, "end": 26, "multiplier": 2.0},
        {"type": "demand", "start": 40, "end": 46, "multiplier": 2.0, "forecastable": False}]}})
    p = Params(cfg)
    ev = EventSchedule(p, np.random.default_rng(0))
    dm = DemandModel(p, ev, np.random.default_rng(0))  # forecast_noise = 0 in the quiet config
    base = dm.forecast[0]
    assert dm.forecast[20] == pytest.approx(2 * base, rel=0.05)  # weekly pattern averages out over 7 days
    assert dm.forecast[40] == pytest.approx(base, rel=0.05)
