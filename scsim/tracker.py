"""
Episode logging. The env writes one record per day plus one record per closed
customer order; evals/metrics.py turns these into KPIs and the dashboard plots them.
"""
import numpy as np

ORDER_COLS = ["oid", "region", "product", "qty", "shipped", "high", "day", "due",
              "last_delivery", "n_shipments", "n_warehouses", "status", "closed_day"]
SHIPMENT_COLS = ["day", "oid", "warehouse", "region", "product", "qty", "carrier", "delivery_day", "days_late"]


class EpisodeTracker:
    def __init__(self, params, record_shipments=False):
        self.params = params
        self.days = []
        self.orders = []
        self.shipments = [] if record_shipments else None
        self.begin_inventory_value = 0.0
        self.end_inventory_value = None
        self.termination_reason = None
        self.events_log = []

    def log_day(self, **fields):
        self.days.append(fields)

    def log_order(self, o, status, day):
        self.orders.append((o.oid, o.region, o.product, o.qty, o.shipped, o.high, o.day, o.due,
                            o.last_delivery, o.n_shipments, bin(o.wh_mask).count("1"), status, day))

    def finalize(self, open_orders, end_inventory_value, reason=None):
        for o in open_orders:
            self.log_order(o, "open", None)
        self.end_inventory_value = end_inventory_value
        self.termination_reason = reason

    # ---- export helpers -------------------------------------------------------
    def orders_frame(self):
        import pandas as pd
        return pd.DataFrame(self.orders, columns=ORDER_COLS)

    def shipments_frame(self):
        import pandas as pd
        return pd.DataFrame(self.shipments or [], columns=SHIPMENT_COLS)

    def daily_frame(self):
        """One row per day. Array fields get flattened into e.g. `on_hand|North|fresh_milk`."""
        import pandas as pd  # not imported at the top so the env worker processes stay small
        p = self.params
        rows = []
        for rec in self.days:
            row = {}
            for k, v in rec.items():
                v = np.asarray(v)
                if v.ndim == 0:
                    row[k] = v.item()
                elif v.ndim == 1:
                    labels = p.wh_names if k in _PER_WAREHOUSE else p.prod_names
                    assert len(labels) == len(v), f"don't know how to label field {k}"
                    for lab, x in zip(labels, v):
                        row[f"{k}|{lab}"] = x
                else:
                    for i, a in enumerate(p.wh_names):
                        for j, b in enumerate(p.prod_names):
                            row[f"{k}|{a}|{b}"] = v[i, j]
            rows.append(row)
        return pd.DataFrame(rows)


# 1-d fields indexed by warehouse (everything else 1-d is indexed by product)
_PER_WAREHOUSE = {"utilization", "overflow_volume", "wh_up"}
