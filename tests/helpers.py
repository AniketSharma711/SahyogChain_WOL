import numpy as np

from scsim.config import deep_merge, load_config
from scsim.environment import SupplyChainEnv

QUIET = {"events": {"supplier_delay_prob": 0.0, "damage_prob": 0.0, "demand_spike_prob": 0.0,
                    "outage_prob": 0.0, "strike_prob": 0.0, "scheduled": []},
         "demand": {"forecast_noise": 0.0}}
NO_DEMAND = {"demand": {"base": [[0, 0, 0]] * 3}}


def quiet_config(*overrides):
    cfg = load_config(overrides=QUIET)
    for o in overrides:
        cfg = deep_merge(cfg, o)
    return cfg


def empty_env(*overrides, seed=0):
    """Env with no random events, after reset, with all stock / pipeline / orders removed."""
    env = SupplyChainEnv(quiet_config(*overrides))
    env.reset(seed=seed)
    env.inv.qty[:] = 0
    env.inv.total[:] = 0
    env.pipeline.items.clear()
    env.pipeline.qty[:] = 0
    env.book.open.clear()
    return env


def idle_action(env, **kw):
    """Do nothing: no orders, production off, local / priority / economy."""
    L = env.action_layout
    args = dict(order_lots=np.zeros((env.params.W, env.params.P)), production=np.zeros(env.params.P, dtype=int),
                allocation=1, sourcing=0, carrier=0)
    args.update(kw)
    return L.encode(**args)
