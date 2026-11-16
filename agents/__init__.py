from agents.base import Agent
from agents.heuristics import FCFSAgent, ForecastAgent, GreedyAgent, RuleBasedAgent
from agents.random_agent import RandomAgent

BASELINES = {cls.name: cls for cls in [RandomAgent, RuleBasedAgent, FCFSAgent, GreedyAgent, ForecastAgent]}


def make_agent(name, model_path=None):
    """Baselines by name, or an RL agent ("ppo", "a2c", ...) loaded from `model_path`."""
    if name in BASELINES:
        return BASELINES[name]()
    from agents.rl_agent import RLAgent  # lazy import, pulls in torch
    if model_path is None:
        raise ValueError(f"agent '{name}' needs a model_path")
    return RLAgent(model_path, name=name)
