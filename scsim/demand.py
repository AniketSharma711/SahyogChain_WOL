"""
Customer demand.

Expected demand per region/product/day =
    base * weekly pattern * known multipliers (season, promos) * shocks (unforecast spikes)

Customers arrive as individual orders: #orders ~ Poisson(rate / mean_size) and each
order is 1 + Poisson(mean_size - 1) units, so the expected units per day equal `rate`.

The forecast the planner sees is the average *known* rate over the next week times a
noisy multiplicative error, i.e. it knows about Black Friday but not about random spikes.
"""
import numpy as np


class DemandModel:
    def __init__(self, params, events, rng):
        self.params = params
        T, n = params.horizon, events.n
        R, P = params.W, params.P

        days = np.arange(n)
        weekly = 1.0 + params.weekly_amp * np.sin(2 * np.pi * days / 7.0)
        self.rate_known = params.base_demand[None] * weekly[:, None, None] * events.demand_known  # (n, R, P)
        self.rate_true = self.rate_known * events.demand_shock

        # forecast issued on day t (needed up to t = T for the last observation)
        win = params.forecast_window
        assert T + win <= n, "EventSchedule.LOOKAHEAD too small for the forecast window"
        csum = np.concatenate([np.zeros((1, R, P)), np.cumsum(self.rate_known, axis=0)])
        rolling = (csum[win:T + 1 + win] - csum[0:T + 1]) / win  # (T+1, R, P)
        noise = np.clip(rng.normal(1.0, params.forecast_noise, size=(T + 1, R, P)), 0.3, None)
        self.forecast = rolling * noise

        # pre-generate every customer order of the episode
        m = params.mean_order_size
        self.orders = []
        for t in range(T):
            counts = rng.poisson(self.rate_true[t] / m[None, :])  # (R, P)
            region = np.repeat(np.repeat(np.arange(R), P), counts.ravel())
            product = np.repeat(np.tile(np.arange(P), R), counts.ravel())
            qty = 1 + rng.poisson(m[product] - 1.0)
            high = rng.random(len(product)) < params.p_high
            self.orders.append((region, product, qty, high))

    def units_on_day(self, t):
        """(R, P) units ordered on day t."""
        region, product, qty, _ = self.orders[t]
        out = np.zeros((self.params.W, self.params.P), dtype=np.int64)
        np.add.at(out, (region, product), qty)
        return out
