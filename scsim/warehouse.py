"""Warehouse level calculations (space, holding cost, overflow)."""
import numpy as np


class WarehouseNetwork:
    def __init__(self, params):
        self.names = params.wh_names
        self.capacity = params.capacity          # (W,) volume units
        self.hold_mult = params.hold_mult        # (W,)
        self.volume = params.volume              # (P,)
        self.holding_cost = params.holding_cost  # (P,)
        self.overflow_rate = params.overflow_cost

    def volume_used(self, on_hand):
        """(W, P) units -> (W,) volume units."""
        return on_hand @ self.volume

    def utilization(self, on_hand):
        """Used space / capacity. Can go above 1 (the extra goes to rented overflow space)."""
        return self.volume_used(on_hand) / self.capacity

    def storage_costs(self, on_hand):
        """Holding + overflow cost for one day of storing `on_hand`.

        Overflow: anything above capacity is stored in rented space at a premium. We
        don't throw goods away, it just gets expensive.
        """
        holding = float((on_hand * self.holding_cost[None, :] * self.hold_mult[:, None]).sum())
        overflow_units = np.maximum(0.0, self.volume_used(on_hand) - self.capacity)
        overflow = float(overflow_units.sum() * self.overflow_rate)
        return holding, overflow, overflow_units
