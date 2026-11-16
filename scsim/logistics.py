"""
Transport model.

Two carriers for customer deliveries:
  * truck   - our contracted fleet. Paid per truck per lane per day, so it is cheap
              when trucks are full and expensive for a handful of units. Slower.
  * express - 3rd party parcel service, paid per unit, faster, and not affected by
              trucking disruptions (strikes, road closures) unless the scenario says so.
Transfers between warehouses and supplier deliveries are paid per unit.
"""
import numpy as np

TRUCK, EXPRESS = 0, 1


def _days(km, km_per_day):
    # -1e-9 so that e.g. exactly 250 km at 250 km/day is 1 day and not 2
    return np.maximum(1, np.ceil(km / km_per_day - 1e-9)).astype(int)


class Logistics:
    def __init__(self, params):
        self.params = params
        d = params.distance  # (W, R), warehouse w sits in region w
        tr, ex = params.truck, params.express

        self.distance = d
        self.truck_days = _days(d, tr["km_per_day"])
        self.express_days = _days(d, ex["km_per_day"])
        self.truck_capacity = float(tr["capacity"])
        self.ltl_premium = float(tr["ltl_premium"])
        self.ltl_min = float(tr["ltl_min_fraction"])
        self.truck_trip_cost = tr["base_fee"] + tr["cost_per_km"] * d  # $ per truck on lane (w, r)
        # parcel price per unit (W, R, P)
        self.express_unit_cost = ((ex["fee_per_unit"] + ex["cost_per_unit_km"] * d)[:, :, None]
                                  * params.volume[None, None, :])

        # flexible sourcing: try warehouses in order of distance to the customer (home first)
        self.nearest = [[int(w) for w in np.argsort(d[:, r], kind="stable")] for r in range(params.W)]

        # warehouse -> warehouse transfers go by truck
        self.transfer_days = self.truck_days.copy()
        self.transfer_unit_cost = params.transfer_rate * d[:, :, None] * params.volume[None, None, :]

        # supplier -> warehouse freight, $/unit (W, P)
        self.inbound_freight = params.freight_rate[None, :] * params.inbound_km[:, None] / 100.0

        self.co2_truck = float(tr["co2_kg_per_km"])
        self.fuel_truck = float(tr["fuel_l_per_km"])
        self.co2_express = float(ex["co2_kg_per_unit_km"])

    def bill_trucks(self, load, cost_mult=1.0):
        """load: (W, R) volume sent by truck today.

        Full trucks pay the whole trip. The last, partly filled truck on a lane is billed
        LTL-style: pro rata with a premium and a minimum charge (never more than a full trip).
        """
        cap = self.truck_capacity
        full = np.floor(load / cap + 1e-9)
        rest = np.where(load - full * cap > 1e-9, load - full * cap, 0.0)
        partial = np.where(rest > 0, np.clip(self.ltl_premium * rest / cap, self.ltl_min, 1.0), 0.0)
        trucks = (full + (rest > 0)).astype(int)
        cost = float(((full + partial) * self.truck_trip_cost).sum() * cost_mult)
        km = float((trucks * self.distance).sum())
        return trucks, cost, km

    def inbound_co2(self, volume, w):
        """Supplier deliveries are on shared trucks, so count them as truck-equivalents."""
        return volume / self.truck_capacity * self.params.inbound_km[w] * self.co2_truck
