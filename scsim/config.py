"""
Config handling.

Everything about the simulated company lives in configs/default.yaml. Scenario
files (configs/scenarios/*.yaml) only contain the keys they change and get
deep-merged on top of the default config.
"""
import copy
from pathlib import Path

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"


def deep_merge(base, override):
    """Merge `override` into a copy of `base`. Nested dicts are merged, lists are replaced."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path=None, overrides=None):
    cfg = load_yaml(path or DEFAULT_CONFIG_PATH)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def _arr(x, dtype=float):
    return np.asarray(x, dtype=dtype)


class Params:
    """Flattens the nested config dict into numpy arrays the simulator uses.

    Index convention used everywhere:
        w = warehouse, r = customer region, p = product
    Region r is the home region of warehouse r, so W == R.
    """

    N_PROD_LEVELS = 3  # off / single shift / double shift

    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg.get("scenario", "default")
        self.horizon = int(cfg["horizon"])

        wh = cfg["warehouses"]
        self.wh_names = list(wh["names"])
        self.W = len(self.wh_names)
        self.capacity = _arr(wh["capacity"])
        self.hold_mult = _arr(wh["holding_multiplier"])
        self.lead_time = _arr(wh["lead_time"], int)
        self.exp_lead_time = _arr(wh["expedited_lead_time"], int)
        self.inbound_km = _arr(wh["inbound_distance_km"])
        self.initial_cover_days = float(wh["initial_cover_days"])

        pr = cfg["products"]
        self.prod_names = list(pr["names"])
        self.P = len(self.prod_names)
        self.price = _arr(pr["price"])
        self.unit_cost = _arr(pr["unit_cost"])
        self.volume = _arr(pr["volume"])
        self.holding_cost = _arr(pr["holding_cost"])
        self.order_lot = _arr(pr["order_lot"], int)
        self.transfer_lot = _arr(pr["transfer_lot"], int)
        self.freight_rate = _arr(pr["freight_per_100km"])
        self.demand_scale = _arr(pr["demand_scale"])
        shelf = pr["shelf_life"]
        self.perishable = np.array([s is not None for s in shelf])
        self.shelf_life = np.array([int(s) if s is not None else 0 for s in shelf])

        dm = cfg["demand"]
        self.base_demand = _arr(dm["base"])  # (R, P)
        self.mean_order_size = _arr(dm["mean_order_size"])
        self.weekly_amp = float(dm["weekly_amplitude"])
        self.p_high = float(dm["high_priority_prob"])
        self.forecast_noise = float(dm["forecast_noise"])
        self.forecast_window = int(dm["forecast_window"])

        sv = cfg["service"]
        self.due_high = int(sv["due_days_high"])
        self.due_std = int(sv["due_days_standard"])
        self.patience = int(sv["patience"])

        sp = cfg["supplier"]
        self.sup_capacity = _arr(sp["capacity"])
        self.level_mult = _arr(sp["level_multiplier"])
        self.shift_cost = _arr(sp["shift_cost"])  # (P, levels)
        self.max_backlog_days = float(sp["max_backlog_days"])
        self.expedite_mult = float(sp["expedite_freight_multiplier"])

        lg = cfg["logistics"]
        self.distance = _arr(lg["distance_km"])  # (W, R)
        self.truck = dict(lg["truck"])
        self.express = dict(lg["express"])
        self.handling = float(lg["handling_per_unit"])
        self.transfer_rate = float(lg["transfer_cost_per_unit_km"])
        self.packaging_kg = float(lg["packaging_kg_per_shipment"])

        self.overflow_cost = float(cfg["costs"]["overflow_per_unit"])
        self.disposal_cost = float(cfg["costs"]["disposal_per_unit"])

        self.initial_cash = float(cfg["finance"]["initial_cash"])
        self.credit_limit = float(cfg["finance"]["credit_limit"])

        rw = cfg["reward"]
        self.reward_scale = float(rw["scale"])
        self.late_penalty = float(rw["late_penalty"])
        self.high_late_mult = float(rw["high_priority_late_mult"])
        self.lost_penalty = float(rw["lost_sale_penalty"])
        self.term_penalty_per_day = float(rw["termination_penalty_per_day"])

        self.collapse_window = int(cfg["termination"]["collapse_window"])
        self.collapse_ratio = float(cfg["termination"]["collapse_lost_ratio"])
        self.overstock_days = float(cfg["metrics"]["overstock_days"])

        self.events = dict(cfg["events"])
        self._validate()

    def _validate(self):
        W, P = self.W, self.P
        for name in ["capacity", "hold_mult", "lead_time", "exp_lead_time", "inbound_km"]:
            assert getattr(self, name).shape == (W,), f"warehouses.{name} needs {W} values"
        for name in ["price", "unit_cost", "volume", "holding_cost", "order_lot", "transfer_lot",
                     "freight_rate", "demand_scale", "mean_order_size", "sup_capacity"]:
            assert getattr(self, name).shape == (P,), f"{name} needs {P} values"
        assert self.base_demand.shape == (W, P), "demand.base must be regions x products"
        assert self.distance.shape == (W, W), "logistics.distance_km must be W x W"
        assert self.shift_cost.shape == (P, self.N_PROD_LEVELS)
        assert self.level_mult.shape == (self.N_PROD_LEVELS,)
        assert (self.lead_time >= 1).all() and (self.exp_lead_time >= 1).all()
        assert (self.exp_lead_time <= self.lead_time).all(), "expediting should not be slower"
        assert (self.mean_order_size >= 1).all()
        assert (self.shelf_life[self.perishable] >= 2).all()
