"""
Stock bookkeeping.

On-hand stock is stored per expiry day so that we can
  * always sell the oldest units first (FEFO = first expired, first out)
  * throw away whatever is still on the shelf when it expires
Non-perishable products just sit in the last bucket, which an episode never reaches.
"""
from dataclasses import dataclass

import numpy as np


class Inventory:
    """qty[w, p, e] = units of product p at warehouse w that can still be sold up to
    and including day e."""

    def __init__(self, n_wh, n_prod, n_days):
        self.qty = np.zeros((n_wh, n_prod, n_days), dtype=np.int64)
        self.total = np.zeros((n_wh, n_prod), dtype=np.int64)  # cached sum over expiry
        self.never = n_days - 1  # bucket for non-perishables

    def add(self, w, p, qty, expiry):
        if qty <= 0:
            return
        self.qty[w, p, expiry] += qty
        self.total[w, p] += qty

    def remove(self, w, p, qty, today):
        """Take `qty` units, earliest expiry first. Returns [(expiry, units), ...]."""
        if qty > self.total[w, p]:
            raise ValueError(f"trying to remove {qty} units but only {self.total[w, p]} on hand")
        taken = []
        if qty <= 0:
            return taken
        row = self.qty[w, p]
        left = qty
        # buckets before `today` are always empty (expired stock is removed every evening)
        for e in np.flatnonzero(row[today:]):
            e = int(e) + today
            k = min(left, int(row[e]))
            row[e] -= k
            taken.append((e, k))
            left -= k
            if left == 0:
                break
        self.total[w, p] -= qty
        return taken

    def expire(self, today):
        """End of day: stock with expiry == today is waste. Returns (W, P) wasted units."""
        wasted = self.qty[:, :, today].copy()
        self.qty[:, :, today] = 0
        self.total -= wasted
        return wasted

    def expiring_within(self, today, days):
        """Units that expire in the next `days` days (today included)."""
        return self.qty[:, :, today:today + days].sum(axis=2)


@dataclass(slots=True)
class Shipment:
    """Goods on their way to one of our warehouses."""
    arrival: int
    warehouse: int
    product: int
    qty: int
    expiry: int
    kind: str  # "supplier" or "transfer"


class Pipeline:
    """Everything currently in transit towards our warehouses."""

    def __init__(self, n_wh, n_prod):
        self.items = []
        self.qty = np.zeros((n_wh, n_prod), dtype=np.int64)

    def add(self, shipment):
        if shipment.qty <= 0:
            return
        self.items.append(shipment)
        self.qty[shipment.warehouse, shipment.product] += shipment.qty

    def pop_arrivals(self, day, wh_up):
        """Shipments due by `day`. Deliveries to a closed warehouse wait until it reopens."""
        arrived, keep = [], []
        for s in self.items:
            if s.arrival <= day and wh_up[s.warehouse]:
                arrived.append(s)
            else:
                keep.append(s)
        self.items = keep
        for s in arrived:
            self.qty[s.warehouse, s.product] -= s.qty
        return arrived
