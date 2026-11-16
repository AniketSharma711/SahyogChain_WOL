import numpy as np
import pytest

from agents import BASELINES
from evals.metrics import compute_metrics
from evals.runner import run_episode
from evals.scenarios import ALL_SCENARIOS, make_env
from scsim.spaces import ActionLayout


@pytest.mark.parametrize("name", list(BASELINES))
def test_baselines_give_valid_actions(name):
    for scen in ALL_SCENARIOS:
        env = make_env(scen)
        obs, _ = env.reset(seed=0)
        agent = BASELINES[name]()
        agent.reset(env, seed=0)
        for _ in range(25):
            a = agent.act(obs)
            assert env.action_space.contains(np.asarray(a, dtype=np.int64)), (scen, a)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break


def test_action_layout_roundtrip():
    lay = ActionLayout(3, 3)
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.integers(0, lay.nvec)
        d = lay.decode(a)
        b = lay.encode(d.order_lots, d.expedite, d.production, d.transfers, d.allocation, d.sourcing, d.carrier)
        np.testing.assert_array_equal(a, b)
    assert lay.pairs[lay.transfer_code(2, 0) - 1] == (2, 0)


def test_obs_layout_roundtrip():
    env = make_env("normal")
    obs, _ = env.reset(seed=0)
    o = env.decode_obs(obs)
    np.testing.assert_allclose(o["on_hand"], env.inv.total, atol=1e-3)
    np.testing.assert_allclose(o["inbound"], env.pipeline.qty, atol=1e-3)
    np.testing.assert_allclose(o["lead_time"], env.params.lead_time, atol=1e-5)


def test_forecast_heuristic_beats_random():
    env = make_env("normal")
    score = {}
    for name in ["random", "forecast"]:
        score[name] = np.mean([compute_metrics(run_episode(env, BASELINES[name](), seed=s))["score"]
                               for s in range(3)])
    assert score["forecast"] > score["random"] + 20_000
