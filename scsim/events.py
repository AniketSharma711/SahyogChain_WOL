"""
Disruptions, both random background noise and the big scheduled scenario events.

The whole episode is generated at reset() from its own random stream, so it only
depends on the seed and the scenario, never on what the agent does. That way two
policies evaluated on the same seed face exactly the same disruptions and the same
customers (common random numbers -> much less noisy comparisons).

Scheduled event types (end day is inclusive):
    supplier          start, end, capacity_factor, extra_delay, products
    warehouse_outage  start, end, warehouse
    transport         start, end, extra_days, cost_multiplier, express_available
    demand            start, end, multiplier, regions, products, forecastable
    demand_curve      knots [[day, mult], ...] (linear interpolation), regions, products, forecastable
"""
import numpy as np


class EventSchedule:
    LOOKAHEAD = 10  # extra days so that forecasts near the end of the horizon still work

    def __init__(self, params, rng):
        self.params = params
        T = params.horizon
        n = T + self.LOOKAHEAD
        self.n = n
        W, P = params.W, params.P

        self.supplier_factor = np.ones((n, P))        # production capacity multiplier
        self.supplier_delay = np.zeros(n, dtype=int)  # extra days for shipments dispatched that day
        self.wh_up = np.ones((n, W), dtype=bool)
        self.truck_delay = np.zeros(n, dtype=int)     # extra transit days for anything on a truck
        self.cost_mult = np.ones(n)                   # fuel surcharge on transport
        self.express_ok = np.ones(n, dtype=bool)
        self.damage_frac = np.zeros((n, W))           # share of arriving supplier goods that is damaged
        self.demand_known = np.ones((n, W, P))        # demand multipliers the forecast knows about
        self.demand_shock = np.ones((n, W, P))        # ...and the ones it doesn't

        self.log = []  # (day, text) for the dashboard / render
        self._random_events(params.events, rng, T)
        for ev in params.events.get("scheduled") or []:
            self._scheduled(ev)
        self.log.sort(key=lambda x: x[0])

    def _random_events(self, cfg, rng, T):
        W = self.wh_up.shape[1]
        names = self.params.wh_names
        # draw everything with fixed shapes up front, so the random stream never depends
        # on which events actually fire
        u_delay, l_delay = rng.random(T), rng.integers(1, 4, size=T)
        u_dmg, f_dmg = rng.random((T, W)), rng.uniform(0.1, 0.3, size=(T, W))
        u_spike, m_spike = rng.random((T, W)), rng.uniform(1.5, 2.5, size=(T, W))
        l_spike = rng.integers(1, 4, size=(T, W))
        u_out, l_out = rng.random((T, W)), rng.integers(1, 3, size=(T, W))
        u_strike, l_strike = rng.random(T), rng.integers(2, 5, size=T)

        for t in range(T):
            if u_delay[t] < cfg.get("supplier_delay_prob", 0.0):
                self.supplier_delay[t] = max(self.supplier_delay[t], l_delay[t])
                self.log.append((t, f"supplier shipments delayed {l_delay[t]}d"))
            if u_strike[t] < cfg.get("strike_prob", 0.0):
                sl = slice(t, t + l_strike[t])
                self.truck_delay[sl] = np.maximum(self.truck_delay[sl], 1)
                self.log.append((t, f"trucking slowdown for {l_strike[t]}d"))
            for w in range(W):
                if u_dmg[t, w] < cfg.get("damage_prob", 0.0):
                    self.damage_frac[t, w] = f_dmg[t, w]
                    self.log.append((t, f"damaged delivery at {names[w]} ({f_dmg[t, w]:.0%})"))
                if u_spike[t, w] < cfg.get("demand_spike_prob", 0.0):
                    sl = slice(t, t + l_spike[t, w])
                    self.demand_shock[sl, w, :] = np.maximum(self.demand_shock[sl, w, :], m_spike[t, w])
                    self.log.append((t, f"demand spike x{m_spike[t, w]:.1f} in region {names[w]}"))
                if u_out[t, w] < cfg.get("outage_prob", 0.0):
                    self.wh_up[t:t + l_out[t, w], w] = False
                    self.log.append((t, f"{names[w]} warehouse outage {l_out[t, w]}d"))

    def _scheduled(self, ev):
        kind = ev["type"]
        P, W = self.params.P, self.params.W
        start = int(ev.get("start", 0))
        end = min(int(ev.get("end", self.n - 1)), self.n - 1)
        days = np.arange(start, end + 1)
        prods = list(ev.get("products", range(P)))
        regions = list(ev.get("regions", range(W)))

        if kind == "supplier":
            idx = np.ix_(days, prods)
            self.supplier_factor[idx] = np.minimum(self.supplier_factor[idx], ev.get("capacity_factor", 1.0))
            self.supplier_delay[days] = np.maximum(self.supplier_delay[days], int(ev.get("extra_delay", 0)))
            text = f"supplier at {ev.get('capacity_factor', 1.0):.0%} capacity"
        elif kind == "warehouse_outage":
            self.wh_up[days, int(ev["warehouse"])] = False
            text = f"{self.params.wh_names[int(ev['warehouse'])]} warehouse down"
        elif kind == "transport":
            self.truck_delay[days] = np.maximum(self.truck_delay[days], int(ev.get("extra_days", 0)))
            self.cost_mult[days] = np.maximum(self.cost_mult[days], float(ev.get("cost_multiplier", 1.0)))
            if not ev.get("express_available", True):
                self.express_ok[days] = False
            text = f"transport disruption (+{ev.get('extra_days', 0)}d trucks)"
        elif kind in ("demand", "demand_curve"):
            target = self.demand_known if ev.get("forecastable", True) else self.demand_shock
            if kind == "demand":
                target[np.ix_(days, regions, prods)] *= float(ev["multiplier"])
                text = f"demand x{ev['multiplier']}"
            else:
                knots = np.asarray(ev["knots"], dtype=float)
                days = np.arange(self.n)
                mult = np.interp(days, knots[:, 0], knots[:, 1])  # flat outside the knots
                target[np.ix_(days, regions, prods)] *= mult[:, None, None]
                start, end = int(knots[0, 0]), int(knots[-1, 0])
                text = f"seasonal demand curve (peak x{knots[:, 1].max():.1f})"
        else:
            raise ValueError(f"unknown event type: {kind}")
        self.log.append((start, f"[scheduled d{start}-{end}] {text}"))
