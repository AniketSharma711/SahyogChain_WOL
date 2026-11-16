"""
Customer orders and fulfilment.

Rules (all decided by the agent's action for the day):
  * allocation  - FCFS (oldest order first) or priority (high priority first, then earliest due date)
  * sourcing    - local (only the home warehouse of the customer's region) or flexible.
                  Flexible runs in two passes: first every order is served from its home
                  warehouse, then whatever is still open may use the stock the other
                  warehouses have left, nearest first (-> split shipments). So cross-shipping
                  never takes stock away from another warehouse's own customers that day.
  * carrier     - economy (truck), smart (express only if that turns a late delivery into an
                  on-time one) or express (always, when available)
Orders can be shipped partially; whatever is still open after due + patience days is
cancelled and counts as a lost sale.
"""
from dataclasses import dataclass

import numpy as np

from scsim.spaces import ALLOC_PRIORITY, CARRIER_EXPRESS, CARRIER_SMART, SOURCE_LOCAL


@dataclass(slots=True)
class CustomerOrder:
    oid: int
    region: int
    product: int
    qty: int
    high: bool
    day: int
    due: int
    shipped: int = 0
    last_delivery: int = -1   # delivery day of the last shipped part
    n_shipments: int = 0
    wh_mask: int = 0          # bitmask of warehouses that shipped something

    @property
    def remaining(self):
        return self.qty - self.shipped


class FulfilResult:
    def __init__(self, W, P):
        self.revenue = 0.0
        self.cogs = 0.0
        self.late_penalty = 0.0
        self.late_unit_days = 0
        self.units = 0
        self.shipped = np.zeros((W, P), dtype=np.int64)    # by source warehouse
        self.truck_load = np.zeros((W, W))                 # volume per lane (warehouse -> region)
        self.express_units = 0
        self.express_cost = 0.0
        self.express_unit_km = 0.0                         # volume * km, for CO2
        self.unit_km = 0.0                                 # units * km, for avg delivery distance
        self.n_shipments = 0
        self.closed = []                                   # orders completed today


class OrderBook:
    def __init__(self, params):
        self.params = params
        self.open = []
        self._next_id = 0

    def add_day(self, day, region, product, qty, high):
        p = self.params
        for r, pr, q, h in zip(region.tolist(), product.tolist(), qty.tolist(), high.tolist()):
            due = day + (p.due_high if h else p.due_std)
            self.open.append(CustomerOrder(self._next_id, r, pr, q, h, day, due))
            self._next_id += 1

    def backlog(self):
        """(R, P) units ordered but not shipped yet."""
        out = np.zeros((self.params.W, self.params.P), dtype=np.int64)
        for o in self.open:
            out[o.region, o.product] += o.remaining
        return out

    def late_risk(self, day):
        """(R, P) open units that can't arrive on time any more (every lane takes >= 1 day)."""
        out = np.zeros((self.params.W, self.params.P), dtype=np.int64)
        for o in self.open:
            if o.due <= day:
                out[o.region, o.product] += o.remaining
        return out

    def fulfil(self, t, inv, logistics, events, allocation, sourcing, carrier, shipment_log=None):
        # TODO: no stock reservation - "priority" only changes the order of today's queue, it can't
        #  hold units back for high priority orders that will come in tomorrow
        p, lg = self.params, logistics
        if allocation == ALLOC_PRIORITY:
            self.open.sort(key=lambda o: (not o.high, o.due, o.day, o.oid))
        else:
            self.open.sort(key=lambda o: (o.day, o.oid))

        up = events.wh_up[t]
        extra_days = int(events.truck_delay[t])
        express_ok = bool(events.express_ok[t])
        res = FulfilResult(p.W, p.P)
        # pass 1: every order from its home warehouse. pass 2 (flexible only): what is still
        # open may use stock the other warehouses didn't need for their own customers today.
        passes = (False,) if sourcing == SOURCE_LOCAL else (False, True)
        for cross in passes:
            for o in self.open:
                if o.remaining > 0:
                    sources = lg.nearest[o.region][1:] if cross else (o.region,)
                    self._ship(o, sources, t, inv, lg, up, extra_days, express_ok, carrier, res, shipment_log)

        still_open = []
        for o in self.open:
            (still_open if o.remaining > 0 else res.closed).append(o)
        self.open = still_open
        return res

    def _ship(self, o, sources, t, inv, lg, up, extra_days, express_ok, carrier, res, shipment_log):
        """Ship as much of order `o` as possible from `sources` (in that order)."""
        p = self.params
        need = o.remaining
        for w in sources:
            if need == 0:
                break
            if not up[w]:
                continue
            k = min(need, int(inv.total[w, o.product]))
            if k == 0:
                continue
            inv.remove(w, o.product, k, t)

            truck_days = int(lg.truck_days[w, o.region]) + extra_days
            exp_days = int(lg.express_days[w, o.region])
            use_express = False
            if express_ok and carrier == CARRIER_EXPRESS:
                use_express = True
            elif express_ok and carrier == CARRIER_SMART:
                use_express = t + truck_days > o.due and t + exp_days <= o.due
            delivery = t + (exp_days if use_express else truck_days)

            price = p.price[o.product]
            res.revenue += price * k
            res.cogs += p.unit_cost[o.product] * k
            days_late = max(0, delivery - o.due)
            if days_late > 0:
                mult = p.high_late_mult if o.high else 1.0
                res.late_penalty += p.late_penalty * price * k * days_late * mult
                res.late_unit_days += k * days_late
            vol = k * p.volume[o.product]
            if use_express:
                res.express_units += k
                res.express_cost += k * lg.express_unit_cost[w, o.region, o.product]
                res.express_unit_km += vol * lg.distance[w, o.region]
            else:
                res.truck_load[w, o.region] += vol
            res.shipped[w, o.product] += k
            res.units += k
            res.unit_km += k * lg.distance[w, o.region]
            res.n_shipments += 1

            o.shipped += k
            o.n_shipments += 1
            o.wh_mask |= 1 << w
            o.last_delivery = max(o.last_delivery, delivery)
            need -= k
            if shipment_log is not None:
                shipment_log.append((t, o.oid, w, o.region, o.product, k,
                                     "express" if use_express else "truck", delivery, days_late))

    def cancel_overdue(self, t):
        """Customers give up on orders still open at the end of day due + patience."""
        keep, cancelled = [], []
        for o in self.open:
            (cancelled if t >= o.due + self.params.patience else keep).append(o)
        self.open = keep
        return cancelled
