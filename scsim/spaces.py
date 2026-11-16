"""
Action / observation layouts.

The action is a MultiDiscrete vector so any algorithm that handles MultiDiscrete
(PPO, A2C, ...) can be plugged in. The layouts below translate between the flat
vectors and named fields, so the heuristic baselines can work with the exact
same observation the RL agent gets.
"""
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

N_ORDER_LEVELS = 5  # order 0..4 lots per warehouse/product per day

ALLOC_FCFS, ALLOC_PRIORITY = 0, 1
SOURCE_LOCAL, SOURCE_FLEXIBLE = 0, 1
CARRIER_ECONOMY, CARRIER_SMART, CARRIER_EXPRESS = 0, 1, 2


@dataclass
class Action:
    order_lots: np.ndarray   # (W, P) lots to order from the supplier
    expedite: bool           # expedited freight for today's orders
    production: np.ndarray   # (P,) shift level per product line: 0 off, 1 single, 2 double
    transfers: np.ndarray    # (P,) 0 = no transfer, k = move one lot along pairs[k-1]
    allocation: int          # ALLOC_*   (order in which the backlog is served)
    sourcing: int            # SOURCE_*  (home warehouse only, or any warehouse)
    carrier: int             # CARRIER_* (truck only / express only when it saves the due date / always express)


class ActionLayout:
    def __init__(self, n_wh, n_prod):
        W, P = n_wh, n_prod
        self.W, self.P = W, P
        self.pairs = [(i, j) for i in range(W) for j in range(W) if i != j]
        self.nvec = np.array([N_ORDER_LEVELS] * (W * P) + [2] + [3] * P
                             + [len(self.pairs) + 1] * P + [2, 2, 3], dtype=np.int64)
        i = W * P
        self.s_orders = slice(0, i)
        self.i_expedite = i
        self.s_production = slice(i + 1, i + 1 + P)
        self.s_transfers = slice(i + 1 + P, i + 1 + 2 * P)
        self.i_alloc = i + 1 + 2 * P
        self.i_source = self.i_alloc + 1
        self.i_carrier = self.i_alloc + 2
        assert self.i_carrier + 1 == len(self.nvec)

    def space(self):
        return spaces.MultiDiscrete(self.nvec)

    def decode(self, a):
        a = np.asarray(a, dtype=np.int64).reshape(-1)
        if a.shape[0] != len(self.nvec) or (a < 0).any() or (a >= self.nvec).any():
            raise ValueError(f"invalid action {a}")
        return Action(
            order_lots=a[self.s_orders].reshape(self.W, self.P),
            expedite=bool(a[self.i_expedite]),
            production=a[self.s_production],
            transfers=a[self.s_transfers],
            allocation=int(a[self.i_alloc]),
            sourcing=int(a[self.i_source]),
            carrier=int(a[self.i_carrier]),
        )

    def encode(self, order_lots, expedite=False, production=None, transfers=None,
               allocation=ALLOC_PRIORITY, sourcing=SOURCE_LOCAL, carrier=CARRIER_SMART):
        a = np.zeros(len(self.nvec), dtype=np.int64)
        lots = np.clip(np.rint(np.asarray(order_lots)), 0, N_ORDER_LEVELS - 1).astype(np.int64)
        a[self.s_orders] = lots.reshape(-1)
        a[self.i_expedite] = int(bool(expedite))
        a[self.s_production] = 1 if production is None else production
        a[self.s_transfers] = 0 if transfers is None else transfers
        a[self.i_alloc], a[self.i_source], a[self.i_carrier] = allocation, sourcing, carrier
        return a

    def transfer_code(self, src, dst):
        return self.pairs.index((src, dst)) + 1


class ObsLayout:
    """Flat float32 observation. Quantities are divided by the product's demand_scale
    (so 1.0 ~ one day of network-wide demand), a few other fields get a constant scale."""

    QTY_FIELDS = ("on_hand", "inbound", "queued", "backlog", "late_risk",
                  "expiring", "forecast", "recent_demand")
    CONST_SCALE = {"lead_time": 5.0, "capacity": 1000.0, "supplier_delay": 3.0, "truck_delay": 3.0}

    def __init__(self, n_wh, n_prod, demand_scale):
        W, P = n_wh, n_prod
        self.demand_scale = np.asarray(demand_scale, dtype=float)
        self.fields = [(name, (W, P)) for name in self.QTY_FIELDS] + [
            ("utilization", (W,)),       # used space / capacity
            ("wh_up", (W,)),             # 1 if the warehouse operates today
            ("lead_time", (W,)),         # static, but differs between scenarios
            ("capacity", (W,)),
            ("supplier_factor", (P,)),   # 1 = normal capacity
            ("supplier_delay", (1,)),    # days today's supplier shipments will be late
            ("truck_delay", (1,)),       # extra days on every truck lane
            ("express_ok", (1,)),
            ("cost_mult", (1,)),         # fuel surcharge
            ("dow", (2,)),               # day of week as sin/cos
            ("cash", (1,)),              # cash / initial cash
        ]
        self.slices, i = {}, 0
        for name, shape in self.fields:
            n = int(np.prod(shape))
            self.slices[name] = (slice(i, i + n), shape)
            i += n
        self.size = i

    def _scale(self, name):
        if name in self.QTY_FIELDS:
            return self.demand_scale  # broadcasts over the last (product) axis
        return self.CONST_SCALE.get(name, 1.0)

    def pack(self, values):
        out = np.zeros(self.size, dtype=np.float32)
        for name, (sl, shape) in self.slices.items():
            v = np.asarray(values[name], dtype=float).reshape(shape) / self._scale(name)
            out[sl] = v.reshape(-1)
        return out

    def unpack(self, obs):
        """Back to raw units (dict of arrays)."""
        obs = np.asarray(obs, dtype=float).reshape(-1)
        return {name: obs[sl].reshape(shape) * self._scale(name) for name, (sl, shape) in self.slices.items()}
