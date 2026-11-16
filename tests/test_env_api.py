import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import scsim  # noqa: F401  (registers SupplyChain-v0)
from agents import BASELINES
from evals.runner import run_episode
from evals.scenarios import ALL_SCENARIOS, make_env
from scsim.environment import SupplyChainEnv


def test_gymnasium_env_checker():
    check_env(SupplyChainEnv(), skip_render_check=True)


def test_sb3_env_checker():
    sb3_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    sb3_checker.check_env(SupplyChainEnv(), warn=True)


def test_gym_make():
    env = gym.make("SupplyChain-v0")
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_full_episode_obs_in_space(scenario):
    env = make_env(scenario)
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    steps, done = 0, False
    while not done:
        obs, r, term, trunc, info = env.step(rng.integers(0, env.action_space.nvec))
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
        steps += 1
        done = term or trunc
    if not term:
        assert trunc and steps == env.params.horizon


def test_same_seed_same_episode():
    def rollout():
        env = SupplyChainEnv()
        obs, _ = env.reset(seed=42)
        rng = np.random.default_rng(7)
        out = [obs]
        for _ in range(30):
            obs, r, *_ = env.step(rng.integers(0, env.action_space.nvec))
            out.append(obs)
            out.append(np.array([r]))
        return np.concatenate(out)

    np.testing.assert_array_equal(rollout(), rollout())


def test_customers_dont_depend_on_policy():
    """Common random numbers: same seed -> same customer orders and disruptions for every policy."""
    env = make_env("compound_crisis")
    a = run_episode(env, BASELINES["random"](), seed=5)
    demand_a = np.stack([d["demand"] for d in a.days])
    events_a = list(a.events_log)
    b = run_episode(env, BASELINES["forecast"](), seed=5)
    demand_b = np.stack([d["demand"] for d in b.days])
    n = min(len(demand_a), len(demand_b))
    np.testing.assert_array_equal(demand_a[:n], demand_b[:n])
    assert events_a == b.events_log


def test_step_after_done_raises():
    env = SupplyChainEnv()
    env.reset(seed=0)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        done = term or trunc
    with pytest.raises(RuntimeError):
        env.step(env.action_space.sample())


def test_invalid_action_rejected():
    env = SupplyChainEnv()
    env.reset(seed=0)
    bad = env.action_space.nvec.copy()  # one past the max in every slot
    with pytest.raises(ValueError):
        env.step(bad)


def test_mixed_scenarios_get_sampled():
    env = make_env(["normal", "peak_season", "warehouse_failure"])
    seen = set()
    for s in range(30):
        _, info = env.reset(seed=s)
        seen.add(info["scenario"])
    assert seen == {"normal", "peak_season", "warehouse_failure"}


def test_render_ansi():
    env = SupplyChainEnv(render_mode="ansi")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    text = env.render()
    assert "North" in text and "cash" in text
