"""
KPIs computed from an episode log (scsim.tracker.EpisodeTracker).

These are independent of the training reward - e.g. the agent is never directly
rewarded for "on-time rate" or "turnover", we just measure them.

Conventions
  * order-level service metrics use every order placed during the episode; orders still
    open when the episode ends count as not fulfilled (same treatment for every policy)
  * net_profit is accrual based (inventory valued at unit cost), cash_profit is cash based
  * score = sum of rewards in dollars = net_profit - late/lost-sale penalties - termination penalty
"""
import numpy as np

from scsim.spaces import ALLOC_PRIORITY, CARRIER_EXPRESS, CARRIER_SMART, SOURCE_FLEXIBLE
from scsim.supplier import DOUBLE, OFF

METRIC_GROUPS = {
    "service": ["fill_rate", "otif_rate", "high_priority_otif", "avg_delay_days", "csat",
                "lost_units", "split_shipment_rate"],
    "inventory": ["avg_inventory_units", "avg_inventory_value", "inventory_turnover",
                  "stockout_rate", "overstock_rate", "waste_units", "waste_rate"],
    "financial": ["revenue", "cogs", "holding_cost", "shipping_cost", "inbound_freight", "production_cost",
                  "shift_cost", "waste_cost", "service_penalty", "termination_penalty", "net_profit",
                  "cash_profit", "score"],
    "operational": ["warehouse_utilization", "overflow_days", "vehicle_utilization",
                    "avg_delivery_km", "production_efficiency", "express_share", "units_transferred",
                    "supplier_rejected_units"],
    "sustainability": ["co2_kg", "fuel_l", "packaging_kg"],
    "behaviour": ["share_priority_days", "share_flexible_days", "share_smart_days", "share_express_days",
                  "share_expedite_days", "share_double_shift", "share_line_off", "ordered_units_per_day"],
}


def _safe_div(a, b):
    return float(a) / float(b) if b else float("nan")


def compute_metrics(tracker):
    p = tracker.params
    days = tracker.days
    n = len(days)

    def total(key):
        return float(sum(np.sum(d[key]) for d in days))

    def stack(key):
        return np.stack([np.asarray(d[key], dtype=float) for d in days])

    m = {"days": n, "terminated": tracker.termination_reason or ""}

    # ---------------------------------------------------------------- service
    orders = tracker.orders_frame()
    fulfilled = orders["status"] == "fulfilled"
    shipped_any = orders["shipped"] > 0
    days_late = np.where(shipped_any, (orders["last_delivery"] - orders["due"]).clip(lower=0), 0)
    on_time = fulfilled & (days_late == 0)

    m["fill_rate"] = _safe_div(orders["shipped"].sum(), orders["qty"].sum())
    m["otif_rate"] = float(on_time.mean()) if len(orders) else float("nan")  # on time AND in full
    m["high_priority_otif"] = float(on_time[orders["high"]].mean()) if orders["high"].any() else float("nan")
    m["avg_delay_days"] = float(days_late[fulfilled].mean()) if fulfilled.any() else float("nan")
    m["lost_units"] = total("lost")
    m["split_shipment_rate"] = float((orders.loc[fulfilled, "n_shipments"] > 1).mean()) if fulfilled.any() else 0.0
    # satisfaction per closed order: share delivered, minus 25% per day late
    closed = orders["status"] != "open"
    sat = (orders["shipped"] / orders["qty"]) * np.clip(1 - 0.25 * days_late, 0, 1)
    m["csat"] = 100 * float(sat[closed].mean()) if closed.any() else float("nan")

    # -------------------------------------------------------------- inventory
    on_hand = stack("on_hand")                         # (n, W, P)
    inv_value = np.array([d["inventory_value"] for d in days])
    m["avg_inventory_units"] = float(on_hand.sum(axis=(1, 2)).mean())
    m["avg_inventory_value"] = float(inv_value.mean())
    cogs = total("cogs")
    # annualised: how many times per year the average stock is sold
    m["inventory_turnover"] = _safe_div(cogs, m["avg_inventory_value"]) * 365 / n
    m["stockout_rate"] = float((on_hand == 0).mean())  # share of warehouse-product-days with nothing on the shelf
    expected = stack("expected_demand")                # home-region demand (n, W, P)
    cover = on_hand / np.maximum(expected, 1e-9)
    m["overstock_rate"] = float((cover > p.overstock_days).mean())
    waste = total("expired") + total("damaged") + total("spoiled_in_transit")
    m["waste_units"] = waste
    m["waste_rate"] = _safe_div(waste, waste + total("units_shipped"))

    # -------------------------------------------------------------- financial
    revenue = total("revenue")
    opex_keys = ["shift", "inbound_freight", "transfer", "truck", "express", "handling",
                 "holding", "overflow", "disposal"]
    opex = sum(total(k) for k in opex_keys)
    m["revenue"] = revenue
    m["cogs"] = cogs
    m["holding_cost"] = total("holding") + total("overflow")
    m["shipping_cost"] = total("truck") + total("express") + total("handling") + total("transfer")
    m["inbound_freight"] = total("inbound_freight")
    m["production_cost"] = total("procurement") + total("shift")
    m["shift_cost"] = total("shift")
    m["waste_cost"] = total("waste_value") + total("disposal")
    m["service_penalty"] = total("late_penalty") + total("lost_penalty")
    m["termination_penalty"] = total("termination")
    m["net_profit"] = revenue - cogs - total("waste_value") - opex
    m["cash_profit"] = revenue - total("procurement") - opex
    m["score"] = total("reward")

    # ------------------------------------------------------------ operational
    util = stack("utilization")
    m["warehouse_utilization"] = float(np.minimum(util, 1.0).mean())
    m["overflow_days"] = float((stack("overflow_volume") > 0).sum())  # warehouse-days over capacity
    m["vehicle_utilization"] = _safe_div(total("truck_volume"), total("trucks") * p.truck["capacity"])
    m["avg_delivery_km"] = _safe_div(total("unit_km"), total("units_shipped"))
    m["production_efficiency"] = _safe_div(total("produced"), total("paid_capacity"))
    m["express_share"] = _safe_div(total("express_units"), total("units_shipped"))
    m["units_transferred"] = total("transfers")
    m["supplier_rejected_units"] = total("rejected")

    # --------------------------------------------------------- sustainability
    m["co2_kg"] = total("co2_kg")
    m["fuel_l"] = total("fuel_l")
    m["packaging_kg"] = total("packaging_kg")

    # -------------------------------------------- behaviour (what the policy chose to do)
    levels = stack("levels")
    m["share_priority_days"] = float(np.mean([d["allocation"] == ALLOC_PRIORITY for d in days]))
    m["share_flexible_days"] = float(np.mean([d["sourcing"] == SOURCE_FLEXIBLE for d in days]))
    m["share_smart_days"] = float(np.mean([d["carrier"] == CARRIER_SMART for d in days]))
    m["share_express_days"] = float(np.mean([d["carrier"] == CARRIER_EXPRESS for d in days]))
    m["share_expedite_days"] = float(np.mean([d["expedite"] for d in days]))
    m["share_double_shift"] = float((levels == DOUBLE).mean())
    m["share_line_off"] = float((levels == OFF).mean())
    m["ordered_units_per_day"] = total("placed") / n
    return m
