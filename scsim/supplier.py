"""
Upstream contract manufacturer.

Make-to-order: every replenishment order (PO) goes into a FIFO queue per product
line, and each day the line produces up to its capacity. Capacity depends on the
shift level the agent pays for (off / single / double) and on disruptions.
Whatever is produced ships to the destination warehouse the same day.
"""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

OFF, SINGLE, DOUBLE = 0, 1, 2


@dataclass(slots=True)
class PurchaseOrder:
    po_id: int
    warehouse: int
    product: int
    qty: int
    remaining: int
    expedite: bool
    day: int


class Supplier:
    def __init__(self, params):
        self.params = params
        self.queues = [deque() for _ in range(params.P)]
        self.queued = np.zeros((params.W, params.P), dtype=np.int64)  # unproduced units per destination
        # the supplier only accepts orders it can make within `max_backlog_days` on a single shift
        self.book_limit = np.floor(params.sup_capacity * params.level_mult[SINGLE]
                                   * params.max_backlog_days + 1e-9).astype(int)
        self._next_id = 0

    def place_order(self, w, p, qty, expedite, day):
        """Returns how many units were accepted (rest is rejected, nothing is charged for it)."""
        room = int(self.book_limit[p] - self.queued[:, p].sum())
        accepted = max(0, min(int(qty), room))
        if accepted > 0:
            self.queues[p].append(PurchaseOrder(self._next_id, w, p, accepted, accepted, bool(expedite), day))
            self.queued[w, p] += accepted
            self._next_id += 1
        return accepted

    def produce(self, levels, capacity_factor):
        """One day of production.

        Returns
            dispatches: list of (warehouse, product, units, expedite)
            produced:   (P,) units made today
            paid_capacity: (P,) units we paid for (before disruptions) - for the efficiency metric
        """
        P = self.params.P
        dispatches = []
        produced = np.zeros(P, dtype=np.int64)
        paid = np.zeros(P)
        for p in range(P):
            paid[p] = self.params.sup_capacity[p] * self.params.level_mult[levels[p]]
            cap = int(math.floor(paid[p] * capacity_factor[p] + 1e-9))
            q = self.queues[p]
            while cap > 0 and q:
                po = q[0]
                k = min(cap, po.remaining)
                po.remaining -= k
                cap -= k
                produced[p] += k
                self.queued[po.warehouse, p] -= k
                dispatches.append((po.warehouse, p, k, po.expedite))
                if po.remaining == 0:
                    q.popleft()
        return dispatches, produced, paid
