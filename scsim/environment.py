"""
SupplyChainEnv - Gymnasium environment for a 3-echelon supply chain
(supplier -> regional warehouses -> customers).

One step = one day. Order of events inside step(t):
    1. agent picks production shifts, replenishment orders, transfers and fulfilment rules
    2. replenishment orders join the supplier queue, transfers leave the source warehouse
    3. supplier produces (capacity-limited, FIFO) and ships to warehouses
    4. today's customer orders come in
    5. fulfilment from on-hand stock, trucks are billed
    6. orders older than due + patience are cancelled (lost sales)
    7. holding / overflow cost on what's left, expired stock is thrown away
    8. night: deliveries arrive for tomorrow morning (damaged / spoiled-in-transit goods = waste)
So the observation for day t+1 already contains the morning deliveries of day t+1.

Reward (dollars / reward.scale) is accrual based:
    revenue - cost of goods sold - value of wasted goods - operating costs - service penalties
COGS is charged when a unit is sold (or wasted), not when it is bought. Summed over an
episode this equals accounting profit minus service penalties, so the RL objective and
the benchmark score are the same thing. Cash (purchases paid at production) is tracked
separately and only matters for the insolvency check.

The horizon end is a time-limit truncation (the business would keep running), early
termination (insolvency / inventory collapse) is a real termination with a penalty.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scsim.config import Params, load_config
from scsim.demand import DemandModel
from scsim.events import EventSchedule
from scsim.inventory import Inventory, Pipeline, Shipment
from scsim.logistics import Logistics
from scsim.orders import OrderBook
from scsim.spaces import ActionLayout, ObsLayout
from scsim.supplier import Supplier
from scsim.tracker import EpisodeTracker
from scsim.warehouse import WarehouseNetwork

MONEY_KEYS = ["revenue", "cogs", "waste_value", "disposal", "procurement", "shift",
              "inbound_freight", "transfer", "truck", "express", "handling", "holding",
              "overflow", "late_penalty", "lost_penalty", "termination"]
# everything that is subtracted from revenue in the reward
REWARD_COST_KEYS = ["cogs", "waste_value", "disposal", "shift", "inbound_freight", "transfer",
                    "truck", "express", "handling", "holding", "overflow", "late_penalty",
                    "lost_penalty", "termination"]
CASH_COST_KEYS = ["procurement", "shift", "inbound_freight", "transfer", "truck", "express",
                  "handling", "holding", "overflow", "disposal"]


class SupplyChainEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, config=None, render_mode=None, record_shipments=False):
        """
        config: a config dict, or a list of config dicts. With a list, every reset() samples
                one of them (that's how we train on a mix of scenarios). All configs need the
                same number of warehouses/products and the same products.demand_scale.
        """
        if config is None:
            config = load_config()
        self.configs = list(config) if isinstance(config, (list, tuple)) else [config]
        self._params = [Params(c) for c in self.configs]
        self._logistics = [Logistics(p) for p in self._params]
        p0 = self._params[0]
        for p in self._params[1:]:
            assert (p.W, p.P) == (p0.W, p0.P), "all configs must have the same network size"
            assert np.allclose(p.demand_scale, p0.demand_scale), "demand_scale must not change between configs"

        self.action_layout = ActionLayout(p0.W, p0.P)
        self.obs_layout = ObsLayout(p0.W, p0.P, p0.demand_scale)
        self.action_space = self.action_layout.space()
        self.observation_space = spaces.Box(low=-10.0, high=100.0, shape=(self.obs_layout.size,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.record_shipments = record_shipments
        self.params = p0
        self.t = 0
        self._done = True

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        # derive independent streams for the scenario pick / customers / disruptions
        self.episode_seed = int(self.np_random.integers(0, 2**63 - 1))
        pick_ss, demand_ss, event_ss = np.random.SeedSequence(self.episode_seed).spawn(3)
        if "config_index" in options:
            idx = int(options["config_index"])
        elif len(self.configs) > 1:
            idx = int(np.random.default_rng(pick_ss).integers(len(self.configs)))
        else:
            idx = 0
        self.config_index = idx
        p = self.params = self._params[idx]
        self.logistics = self._logistics[idx]

        self.events = EventSchedule(p, np.random.default_rng(event_ss))
        self.demand = DemandModel(p, self.events, np.random.default_rng(demand_ss))
        self.network = WarehouseNetwork(p)
        self.supplier = Supplier(p)
        self.inv = Inventory(p.W, p.P, n_days=p.horizon + int(p.shelf_life.max()) + 12)
        self.pipeline = Pipeline(p.W, p.P)
        self.book = OrderBook(p)
        self.tracker = EpisodeTracker(p, self.record_shipments)
        self.tracker.events_log = list(self.events.log)

        self.demand_hist = np.zeros((p.horizon, p.W, p.P))
        self.lost_hist = np.zeros(p.horizon)
        self.cash = p.initial_cash
        self.t = 0
        self._done = False
        self._last = None

        self._initial_stock()
        damaged, spoiled = self._receive(0)  # nothing is due on day 0, but keep it uniform
        assert damaged.sum() == 0 and spoiled.sum() == 0
        self.tracker.begin_inventory_value = self._inventory_value()
        return self._obs(), {"scenario": p.name, "episode_seed": self.episode_seed}

    def _initial_stock(self):
        """A few days of stock on the shelf (of mixed age) plus whatever was already ordered."""
        p = self.params
        for w in range(p.W):
            for j in range(p.P):
                q = int(round(p.initial_cover_days * p.base_demand[w, j]))
                if p.perishable[j]:
                    S = int(p.shelf_life[j])
                    parts = [q // 3, q // 3, q - 2 * (q // 3)]
                    for k, part in enumerate(parts, start=1):  # 1/3, 2/3 and all of the shelf life left
                        self.inv.add(w, j, part, int(np.ceil(S * k / 3)) - 1)
                else:
                    self.inv.add(w, j, q, self.inv.never)

                L = int(p.lead_time[w])
                daily = int(round(p.base_demand[w, j]))
                for arrival in range(1, L):  # orders placed before day 0, produced on day arrival - L
                    expiry = arrival - L + int(p.shelf_life[j]) - 1 if p.perishable[j] else self.inv.never
                    self.pipeline.add(Shipment(arrival, w, j, daily, expiry, "supplier"))

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self._done:
            raise RuntimeError("episode is over, call reset()")
        p, ev, lg = self.params, self.events, self.logistics
        W, P, t = p.W, p.P, self.t
        a = self.action_layout.decode(action)
        m = dict.fromkeys(MONEY_KEYS, 0.0)
        co2 = 0.0

        # 1) production shifts are paid for the day, whatever happens
        m["shift"] = float(p.shift_cost[np.arange(P), a.production].sum())

        # 2) replenishment orders
        placed = np.zeros((W, P), dtype=np.int64)
        rejected = 0
        for w in range(W):
            for j in range(P):
                if a.order_lots[w, j] > 0:
                    qty = int(a.order_lots[w, j] * p.order_lot[j])
                    placed[w, j] = self.supplier.place_order(w, j, qty, a.expedite, t)
                    rejected += qty - placed[w, j]

        # 3) transfers between warehouses (one lot per product per day at most)
        up = ev.wh_up[t]
        moved = np.zeros(P, dtype=np.int64)
        for j in range(P):
            code = int(a.transfers[j])
            if code == 0:
                continue
            src, dst = self.action_layout.pairs[code - 1]
            if not (up[src] and up[dst]):
                continue
            qty = min(int(p.transfer_lot[j]), int(self.inv.total[src, j]))
            if qty <= 0:
                continue
            arrive = t + int(lg.transfer_days[src, dst]) + int(ev.truck_delay[t])
            for expiry, k in self.inv.remove(src, j, qty, t):
                self.pipeline.add(Shipment(arrive, dst, j, k, expiry, "transfer"))
            moved[j] = qty
            m["transfer"] += qty * lg.transfer_unit_cost[src, dst, j] * ev.cost_mult[t]
            co2 += qty * p.volume[j] / lg.truck_capacity * lg.distance[src, dst] * lg.co2_truck

        # 4) production + dispatch to warehouses
        dispatches, produced, paid_cap = self.supplier.produce(a.production, ev.supplier_factor[t])
        for w, j, k, expedited in dispatches:
            if expedited:
                lead, freight_mult = int(p.exp_lead_time[w]), p.expedite_mult
            else:
                lead, freight_mult = int(p.lead_time[w]) + int(ev.truck_delay[t]), 1.0
            lead += int(ev.supplier_delay[t])
            expiry = t + int(p.shelf_life[j]) - 1 if p.perishable[j] else self.inv.never
            self.pipeline.add(Shipment(t + lead, w, j, k, expiry, "supplier"))
            m["procurement"] += k * p.unit_cost[j]
            m["inbound_freight"] += k * lg.inbound_freight[w, j] * freight_mult * ev.cost_mult[t]
            co2 += lg.inbound_co2(k * p.volume[j], w) * (2.0 if expedited else 1.0)

        # 5) new customer orders
        region, product, qty, high = self.demand.orders[t]
        self.book.add_day(t, region, product, qty, high)
        demand_units = self.demand.units_on_day(t)
        self.demand_hist[t] = demand_units

        # 6) fulfilment
        fr = self.book.fulfil(t, self.inv, lg, ev, a.allocation, a.sourcing, a.carrier,
                              self.tracker.shipments)
        trucks, truck_cost, truck_km = lg.bill_trucks(fr.truck_load, ev.cost_mult[t])
        m["revenue"], m["cogs"], m["late_penalty"] = fr.revenue, fr.cogs, fr.late_penalty
        m["truck"] = truck_cost
        m["express"] = fr.express_cost * ev.cost_mult[t]
        m["handling"] = fr.units * p.handling
        co2 += truck_km * lg.co2_truck + fr.express_unit_km * lg.co2_express
        for o in fr.closed:
            self.tracker.log_order(o, "fulfilled", t)

        # 7) cancellations
        lost = np.zeros((W, P), dtype=np.int64)
        for o in self.book.cancel_overdue(t):
            lost[o.region, o.product] += o.remaining
            m["lost_penalty"] += p.lost_penalty * p.price[o.product] * o.remaining
            self.tracker.log_order(o, "cancelled", t)
        self.lost_hist[t] = lost.sum()

        # 8) storage costs, then spoilage at the end of the day
        # TODO: goods in transit don't pay holding cost (they're unavoidable given the lead time anyway)
        holding, overflow, overflow_vol = self.network.storage_costs(self.inv.total)
        m["holding"], m["overflow"] = holding, overflow
        utilization = self.network.utilization(self.inv.total)
        expired = self.inv.expire(t)
        # end-of-day snapshot for the logs (before the overnight deliveries, otherwise a warehouse
        # that sold out today but gets a truck tomorrow morning would never count as a stockout)
        on_hand_eod = self.inv.total.copy()
        inbound_eod = self.pipeline.qty.copy()

        # 9) night -> tomorrow morning's deliveries
        self.t = t + 1
        damaged, spoiled = self._receive(self.t)
        waste = expired + damaged + spoiled
        m["waste_value"] = float((waste * p.unit_cost[None, :]).sum())
        m["disposal"] = float(waste.sum() * p.disposal_cost)

        # 10) money
        self.cash += m["revenue"] - sum(m[k] for k in CASH_COST_KEYS)
        reward_dollars = m["revenue"] - sum(m[k] for k in REWARD_COST_KEYS)

        terminated, reason = False, None
        if self.cash < -p.credit_limit:
            terminated, reason = True, "insolvent"
        elif self.t >= p.collapse_window:
            lo = self.t - p.collapse_window
            demanded = self.demand_hist[lo:self.t].sum()
            if demanded > 0 and self.lost_hist[lo:self.t].sum() / demanded >= p.collapse_ratio:
                terminated, reason = True, "inventory_collapse"
        if terminated:
            # never cheaper than limping on until the end of the horizon
            m["termination"] = p.term_penalty_per_day * (p.horizon - self.t)
            reward_dollars -= m["termination"]
        truncated = (not terminated) and self.t >= p.horizon

        self.tracker.log_day(
            day=t, reward=reward_dollars, cash=self.cash, **m,
            on_hand=on_hand_eod, inbound=inbound_eod, queued=self.supplier.queued.copy(),
            backlog=self.book.backlog(), demand=demand_units, lost=lost,
            forecast=self.demand.forecast[t], expected_demand=self.demand.rate_known[t],
            shipped=fr.shipped, placed=placed, rejected=rejected, expedite=int(a.expedite),
            levels=a.production.copy(), produced=produced, paid_capacity=paid_cap,
            supplier_factor=ev.supplier_factor[t].copy(), transfers=moved,
            allocation=a.allocation, sourcing=a.sourcing, carrier=a.carrier,
            expired=expired, damaged=damaged, spoiled_in_transit=spoiled,
            utilization=utilization, overflow_volume=overflow_vol, wh_up=up.astype(int),
            inventory_value=float((on_hand_eod * p.unit_cost[None, :]).sum()),
            units_shipped=fr.units, n_shipments=fr.n_shipments, express_units=fr.express_units,
            late_unit_days=fr.late_unit_days, unit_km=fr.unit_km,
            trucks=int(trucks.sum()), truck_volume=float(fr.truck_load.sum()), truck_km=truck_km,
            co2_kg=co2, fuel_l=truck_km * lg.fuel_truck, packaging_kg=fr.n_shipments * p.packaging_kg,
            truck_delay=int(ev.truck_delay[t]), cost_mult=float(ev.cost_mult[t]),
        )

        if terminated or truncated:
            self._done = True
            self.tracker.finalize(self.book.open, self._inventory_value(), reason)

        info = {"day": t, "cash": self.cash, "money": m, "scenario": p.name}
        if reason:
            info["termination_reason"] = reason
        self._last = (t, m, reason)
        if self.render_mode == "human":
            print(self.render())
        return self._obs(), float(reward_dollars / p.reward_scale), terminated, truncated, info

    # ---------------------------------------------------------------- helpers
    def _receive(self, day):
        """Unload everything that arrives by `day` at an operating warehouse."""
        p, ev = self.params, self.events
        damaged = np.zeros((p.W, p.P), dtype=np.int64)
        spoiled = np.zeros((p.W, p.P), dtype=np.int64)
        for s in self.pipeline.pop_arrivals(day, ev.wh_up[day]):
            q = s.qty
            frac = ev.damage_frac[day, s.warehouse]
            if s.kind == "supplier" and frac > 0:
                d = int(q * frac)
                damaged[s.warehouse, s.product] += d
                q -= d
            if s.expiry < day:
                spoiled[s.warehouse, s.product] += q  # went off while on the road
            else:
                self.inv.add(s.warehouse, s.product, q, s.expiry)
        return damaged, spoiled

    def _inventory_value(self):
        """On-hand + in-transit stock valued at unit cost."""
        units = self.inv.total + self.pipeline.qty
        return float((units * self.params.unit_cost[None, :]).sum())

    def _obs(self):
        p, ev, t = self.params, self.events, self.t
        win = p.forecast_window
        if t > 0:
            recent = self.demand_hist[max(0, t - win):t].mean(axis=0)
        else:
            recent = self.demand.forecast[0]  # no history yet
        values = {
            "on_hand": self.inv.total,
            "inbound": self.pipeline.qty,
            "queued": self.supplier.queued,
            "backlog": self.book.backlog(),
            "late_risk": self.book.late_risk(t),
            "expiring": self.inv.expiring_within(t, 3),
            "forecast": self.demand.forecast[t],
            "recent_demand": recent,
            "utilization": self.network.utilization(self.inv.total),
            "wh_up": ev.wh_up[t],
            "lead_time": p.lead_time,
            "capacity": p.capacity,
            "supplier_factor": ev.supplier_factor[t],
            "supplier_delay": [ev.supplier_delay[t]],
            "truck_delay": [ev.truck_delay[t]],
            "express_ok": [ev.express_ok[t]],
            "cost_mult": [ev.cost_mult[t]],
            "dow": [np.sin(2 * np.pi * t / 7), np.cos(2 * np.pi * t / 7)],
            "cash": [self.cash / p.initial_cash],
        }
        obs = self.obs_layout.pack(values)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def decode_obs(self, obs):
        """Observation vector -> dict of arrays in real units (used by the heuristics)."""
        return self.obs_layout.unpack(obs)

    def render(self):
        if self._last is None:
            return f"[{self.params.name}] day {self.t}: (no step yet)"
        t, m, reason = self._last
        p = self.params
        lines = [f"[{p.name}] day {t:3d}  cash ${self.cash:,.0f}  revenue ${m['revenue']:,.0f}  "
                 f"lost-sale penalty ${m['lost_penalty']:,.0f}" + (f"  ** {reason} **" if reason else "")]
        lines.append("            " + "".join(f"{n[:10]:>11}" for n in p.prod_names) + "   util")
        util = self.network.utilization(self.inv.total)
        for w, name in enumerate(p.wh_names):
            status = "" if self.events.wh_up[self.t, w] else "  (DOWN)"
            lines.append(f"  {name:<9} " + "".join(f"{x:>11d}" for x in self.inv.total[w])
                         + f"  {util[w]:5.0%}{status}")
        lines.append(f"  backlog   " + "".join(f"{x:>11d}" for x in self.book.backlog().sum(axis=0)))
        return "\n".join(lines)
