"""Benchmark scenarios = small YAML override files in configs/scenarios/."""
from scsim.config import CONFIG_DIR, deep_merge, load_config, load_yaml
from scsim.environment import SupplyChainEnv

SCENARIO_DIR = CONFIG_DIR / "scenarios"

# used for training (with different seeds than the test episodes)
TRAIN_SCENARIOS = ["normal", "peak_season", "supply_shortage", "transport_disruption",
                   "warehouse_failure", "multi_warehouse"]
# only ever used for testing, to see whether the policy generalises
HELD_OUT_SCENARIOS = ["compound_crisis"]
ALL_SCENARIOS = TRAIN_SCENARIOS + HELD_OUT_SCENARIOS


def list_scenarios():
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


def scenario_config(name, base=None):
    path = SCENARIO_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown scenario '{name}', have: {list_scenarios()}")
    cfg = deep_merge(base or load_config(), load_yaml(path))
    cfg["scenario"] = name
    return cfg


def make_env(scenarios, **env_kwargs):
    """One scenario name -> env for that scenario. A list -> env that samples one per episode."""
    if isinstance(scenarios, str):
        return SupplyChainEnv(scenario_config(scenarios), **env_kwargs)
    return SupplyChainEnv([scenario_config(s) for s in scenarios], **env_kwargs)
