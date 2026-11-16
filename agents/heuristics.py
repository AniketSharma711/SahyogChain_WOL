"""
Hand-written baselines. They see the same observation as the RL agent (decoded back
into units) plus static network data like lead times and lot sizes.

    rule_based  static (s, S) reorder policy based on long-run average demand
    fcfs        same ordering as rule_based, but first-come-first-serve, trucks only
    greedy      myopic just-in-time: covers lead-time demand only, always the cheapest option
    forecast    forecast-driven base-stock with safety stock, expediting, rebalancing
                transfers and re-routing around closed warehouses (the strong baseline)
"""
import numpy as np

from agents.base import Agent
from scsim.spaces import (ALLOC_FCFS, ALLOC_PRIORITY, CARRIER_ECONOMY, CARRIER_SMART,
                          N_ORDER_LEVELS, SOURCE_FLEXIBLE, SOURCE_LOCAL)
from scsim.supplier import DOUBLE, OFF, SINGLE


class HeuristicAgent(Agent):
    def reset(self, env, seed=None):
        self.env = env.unwrapped
        self.p = self.env.params
        self.layout = self.env.action_layout
        # variance of daily demand per unit of mean for the compound Poisson demand
        # (#orders ~ Poisson(mu/m), size = 1 + Poisson(m-1))  ->  Var = mu * (m - 1 + m^2) / m
        m = self.p.mean_order_size
        self.var_factor = (m - 1 + m ** 2) / m

    def decode(self, obs):
        return self.env.decode_obs(obs)

    @staticmethod
    def inventory_position(o, backlog=None):
        backlog = o["backlog"] if backlog is None else backlog
        return o["on_hand"] + o["inbound"] + o["queued"] - backlog

    def lots_for(self, gap, rounding=np.ceil):
        """Units still missing -> number of lots (0..4)."""
        lots = rounding(np.maximum(gap, 0) / self.p.order_lot[None, :])
        return np.clip(lots, 0, N_ORDER_LEVELS - 1).astype(int)

    def production_levels(self, o, order_lots, allow_overtime=True):
        """Pay for the smallest shift that clears the supplier queue (incl. today's orders)."""
        p = self.p
        queue = o["queued"].sum(axis=0) + (order_lots * p.order_lot[None, :]).sum(axis=0)
        factor = o["supplier_factor"]
        single_cap = p.sup_capacity * p.level_mult[SINGLE] * factor
        levels = np.full(p.P, SINGLE)
        levels[queue > single_cap] = DOUBLE if allow_overtime else SINGLE
        levels[(queue <= 0) | (factor <= 0)] = OFF
        return levels


class RuleBasedAgent(HeuristicAgent):
    """Textbook (s, S): reorder when the inventory position drops below s, order up to S.
    Uses the long-run average demand, so it doesn't react to seasons or promotions."""

    name = "rule_based"
    safety_days = 2.0
    cycle_days = 2.0
    allocation, carrier = ALLOC_PRIORITY, CARRIER_SMART

    def reset(self, env, seed=None):
        super().reset(env, seed)
        mu = self.p.base_demand
        L = self.p.lead_time[:, None]
        self.s = mu * (L + 1 + self.safety_days)
        self.S = self.s + mu * self.cycle_days

    def act(self, obs):
        o = self.decode(obs)
        ip = self.inventory_position(o)
        lots = np.where(ip <= self.s, self.lots_for(self.S - ip), 0)
        return self.layout.encode(lots, expedite=False, production=self.production_levels(o, lots),
                                  allocation=self.allocation, sourcing=SOURCE_LOCAL, carrier=self.carrier)


class FCFSAgent(RuleBasedAgent):
    """Same replenishment as rule_based but serves orders strictly in arrival order and
    never pays for express. Shows what prioritisation is worth."""

    name = "fcfs"
    allocation, carrier = ALLOC_FCFS, CARRIER_ECONOMY


class GreedyAgent(HeuristicAgent):
    """Myopic / lean planner: only covers the demand expected during the lead time
    (no safety stock), never pays overtime or express, ships from wherever stock is."""

    name = "greedy"

    def act(self, obs):
        o = self.decode(obs)
        target = o["forecast"] * self.p.lead_time[:, None]
        lots = self.lots_for(target - self.inventory_position(o), rounding=np.rint)
        return self.layout.encode(lots, expedite=False,
                                  production=self.production_levels(o, lots, allow_overtime=False),
                                  allocation=ALLOC_PRIORITY, sourcing=SOURCE_FLEXIBLE,
                                  carrier=CARRIER_ECONOMY)


class ForecastAgent(HeuristicAgent):
    """Base-stock policy driven by the 7-day forecast.

    target = mu * T + z * sigma * sqrt(T),  T = lead time + 1 day review + known delays
    Extras: expedite when a warehouse would run dry before a normal order arrives, move a
    lot between warehouses when one is about to run out and another has excess, and when a
    warehouse is closed plan its region's demand at the nearest open warehouse instead.
    """

    name = "forecast"
    z = 1.65               # ~95% cycle service level  TODO: different z per product (margin vs holding cost)
    flex_threshold = 1000  # $ of overdue orders before we start cross-shipping (rough guess, not tuned)

    def act(self, obs):
        o = self.decode(obs)
        p, lg = self.p, self.env.logistics
        up = o["wh_up"] > 0.5
        mu = np.maximum(o["forecast"], 1e-6)
        backlog = o["backlog"].copy()

        # closed warehouse -> its customers get served from the nearest open warehouse
        if up.any():
            for w in np.flatnonzero(~up):
                alt = next(x for x in lg.nearest[w] if up[x])
                mu[alt] += mu[w]
                backlog[alt] += backlog[w]
                mu[w], backlog[w] = 1e-6, 0

        delay = o["truck_delay"][0] + o["supplier_delay"][0]
        protect = p.lead_time[:, None] + 1 + delay
        sigma = np.sqrt(mu * self.var_factor[None, :] * protect)
        target = mu * protect + self.z * sigma
        ip = self.inventory_position(o, backlog)
        lots = self.lots_for(target - ip)
        lots[~up] = 0

        # expedite if that saves at least 2 days of stockout somewhere we're ordering for
        # (expedited freight skips trucking delays, supplier delays hit both the same way)
        net = o["on_hand"] + o["inbound"] - backlog
        days_of_stock = net / mu
        normal_arrival = p.lead_time[:, None] + o["truck_delay"][0]
        saved = normal_arrival - np.maximum(days_of_stock, p.exp_lead_time[:, None])
        expedite = bool(((lots > 0) & (saved >= 2)).any())

        # cross-shipping is expensive (part-loaded trucks), only allow it when it's needed:
        # a warehouse is closed, or there is a real pile of orders that are already late
        overdue_value = float((o["late_risk"] * p.price[None, :]).sum())
        sourcing = SOURCE_FLEXIBLE if (not up.all() or overdue_value > self.flex_threshold) else SOURCE_LOCAL

        transfers = np.zeros(p.P, dtype=int)
        for j in range(p.P):
            if up.sum() < 2:
                break
            days_left = np.where(up, net[:, j] / mu[:, j], np.inf)
            dst = int(np.argmin(days_left))
            if days_left[dst] >= p.lead_time[dst]:
                continue  # nobody is in trouble
            excess = np.where(up, ip[:, j] - target[:, j], -np.inf)
            excess[dst] = -np.inf
            src = int(np.argmax(excess))
            if (excess[src] >= p.transfer_lot[j] and o["on_hand"][src, j] >= p.transfer_lot[j]
                    and lg.transfer_days[src, dst] <= p.lead_time[dst]):
                transfers[j] = self.layout.transfer_code(src, dst)

        return self.layout.encode(lots, expedite=expedite, production=self.production_levels(o, lots),
                                  transfers=transfers, allocation=ALLOC_PRIORITY,
                                  sourcing=sourcing, carrier=CARRIER_SMART)
