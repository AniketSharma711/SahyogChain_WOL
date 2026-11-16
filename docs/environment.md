# Environment specification (`SupplyChain-v0`)

Everything here is configurable in `configs/default.yaml`; the numbers below are the defaults.

## Network

```
                    lead time 2d (exp. 1d)     +-----------+   1d truck
                 +-------------------------->  |  North    |  -----------> North customers
                 |                             +-----------+
+------------+   |  lead time 3d (exp. 1d)     +-----------+   1d truck
|  Supplier  |---+-------------------------->  |  Central  |  -----------> Central customers
| (3 lines)  |   |                             +-----------+
+------------+   |  lead time 4d (exp. 2d)     +-----------+   1d truck
                 +-------------------------->  |  South    |  -----------> South customers
                                               +-----------+
              transfers between warehouses (truck, 2-3 days), cross-shipping to other regions
```

| product | price | unit cost | shelf life | volume | base demand / day (N, C, S) |
|---|---|---|---|---|---|
| fresh_milk | $12 | $6 | 12 days | 1 | 40, 50, 30 |
| cereal | $20 | $11 | 45 days | 1 | 25, 30, 20 |
| blender | $150 | $90 | never expires | 4 | 4, 6, 3 |

Warehouse capacity (volume units): North 1000, Central 1500, South 900. Above capacity, goods go to rented
overflow space at $1 per volume unit per day (nothing is thrown away).

**Supplier.** Make-to-order with one FIFO queue per product line. Each day the agent pays for a shift level per line:
off (0 units, $0), single shift (110 / 70 / 12 units, $150 / $110 / $130) or double shift (1.6x capacity, a bit more
than double the cost). Single-shift capacity is deliberately *below* average demand, so the planner has to decide
when overtime is worth it. The supplier rejects orders beyond 6 days of single-shift output. Produced goods ship the
same day and start ageing from the production day.

**Customers.** Orders arrive per region and product: `#orders ~ Poisson(rate / m)`, size `1 + Poisson(m - 1)` with
mean size m = 4 / 3 / 1.3. Rate = base x weekly pattern (+-15%) x scenario multipliers. 25% of orders are high
priority and promised for the next day, standard orders within 3 days. Orders can be shipped in parts. Whatever is
not shipped by `due + 4` days is cancelled (lost sale).

**Transport.**
* truck (own contracted fleet): 160 volume units per truck, paid per trip (`$35 + $1/km`). A part-loaded truck is
  billed LTL-style: 1.5x its share of a trip, minimum 25% of a trip. Speed 250 km/day.
* express (parcel service): paid per unit (`$1 + $0.004/km` per volume unit), 500 km/day, not affected by trucking
  disruptions unless the scenario says so.
* distances: same region 50-70 km (1 day), neighbour ~300 km (2 days by truck, 1 by express), far 620 km (3 / 2 days).

## Daily timeline (one `step`)

1. The agent's action is applied: shift levels are paid, orders join the supplier queue, transfers leave.
2. The supplier produces up to capacity, FIFO, and ships to the warehouses.
3. Today's customer orders arrive.
4. Fulfilment from on-hand stock (FEFO: earliest expiry first), trucks are billed per lane.
5. Orders past `due + patience` are cancelled.
6. Holding / overflow costs on the remaining stock, then expired stock is thrown away.
7. Night: deliveries for tomorrow morning are unloaded. Damaged goods and goods that expired on the way are waste.
   Deliveries to a closed warehouse wait at the gate until it reopens.

So the observation of day t+1 already includes the deliveries that arrived that morning.

## Observation (94 floats)

All quantities are divided by the product's network-wide daily demand (`products.demand_scale`), so `1.0` is roughly one
day of demand. `env.decode_obs(obs)` converts back to units (the heuristics use that).

| field | shape | meaning |
|---|---|---|
| on_hand | W x P | stock on the shelf |
| inbound | W x P | in transit to the warehouse (supplier + transfers) |
| queued | W x P | ordered from the supplier but not produced yet |
| backlog | R x P | open customer units per region |
| late_risk | R x P | open units that can't arrive on time any more |
| expiring | W x P | units expiring within 3 days |
| forecast | R x P | forecast of average daily demand over the next 7 days (knows seasons/promos, not random spikes, +-10% noise) |
| recent_demand | R x P | average actual demand over the last 7 days |
| utilization | W | used space / capacity |
| wh_up | W | 1 if the warehouse operates today |
| lead_time, capacity | W | static network data (differs between scenarios) |
| supplier_factor | P | capacity multiplier today (disruptions) |
| supplier_delay, truck_delay | 1 each | known extra days today |
| express_ok, cost_mult | 1 each | parcel service available, fuel surcharge |
| dow | 2 | day of week (sin, cos) |
| cash | 1 | cash / initial cash |

There is deliberately no "days left" feature: the episode end is a time-limit *truncation* (the business keeps
running), SB3 bootstraps the value there, and the policy stays stationary instead of learning end-of-horizon tricks
like letting the stock run out on the last days.

## Action (MultiDiscrete, 19 dims)

| part | dims | choices |
|---|---|---|
| replenishment order per warehouse x product | 9 | 0-4 lots (lots: 40 milk, 25 cereal, 5 blenders) |
| expedite today's orders | 1 | no / yes (faster, 3x freight) |
| shift level per product line | 3 | off / single / double |
| transfer per product | 3 | none or one lot along one of the 6 warehouse pairs |
| allocation rule | 1 | FCFS / priority (high priority first, then earliest due date) |
| sourcing rule | 1 | local only / flexible (cross-ship what other warehouses have left after serving their own customers) |
| carrier rule | 1 | economy (truck) / smart (express only if it saves the due date) / express |

Order-level decisions (which warehouse ships which order) are made by these rules instead of one action per order,
which keeps the action space fixed-size. Any algorithm that supports MultiDiscrete works (PPO, A2C, ...).

## Reward

Dollars per day divided by 1000:

```
reward = revenue
       - cost of goods sold (unit cost of what was sold)       - value of wasted goods (expired / damaged)
       - shift costs - inbound freight - transfers - trucks - express - handling
       - holding - overflow - disposal
       - late penalty     (5% of price per unit per day late, x3 for high priority)
       - lost-sale penalty (50% of price per cancelled unit, goodwill)
       - termination penalty (only if the episode ends early)
```

Goods are charged when they are *sold or wasted* (accrual accounting), not when they are bought. Over an episode the
reward therefore sums to exactly `accounting profit - service penalties`, which is also the benchmark score, so the
RL objective and the evaluation agree (`tests/test_accounting.py` checks this identity, plus unit conservation).
Cash is tracked separately on a cash basis (goods paid when produced) and only matters for insolvency.

The reward in the original project brief was a list of event bonuses (+10 order fulfilled, -15 stockout, ...). I
went with money instead because the products are worth very different amounts (a late blender is not the same as a
late carton of milk) and because dollars make the trade-offs between e.g. overtime and lost sales well defined.

## Episode end

* 60 days -> `truncated=True`
* insolvency: cash below -$25k (credit line used up) -> `terminated=True`
* inventory collapse: 90%+ of the demand of the last 14 days was lost -> `terminated=True`

Early termination costs $8000 per remaining day, which is more than the worst plausible day of continuing (all
demand lost at 3x peak demand), so ending the episode on purpose is never a way to stop losses.
The brief's "all orders fulfilled" condition doesn't apply because new orders keep arriving every day.

## Randomness

`reset(seed)` derives separate random streams for customers and for disruptions. Both are generated for the whole
episode at reset time and don't depend on the agent's actions, so every policy evaluated on the same seed sees exactly
the same customers and disruptions (common random numbers). Comparing two policies episode by episode (paired) is
then about 2-2.5x less noisy than comparing their averages, as long as their results are correlated (true for PPO
and the heuristics, not for the random policy).
