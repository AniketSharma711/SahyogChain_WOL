"""scsim - a small supply chain simulator with a Gymnasium interface."""
from gymnasium.envs.registration import register

from scsim.config import Params, load_config
from scsim.environment import SupplyChainEnv

register(id="SupplyChain-v0", entry_point="scsim.environment:SupplyChainEnv")

__all__ = ["SupplyChainEnv", "Params", "load_config"]
