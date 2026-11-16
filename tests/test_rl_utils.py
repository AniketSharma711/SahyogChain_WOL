import functools

import numpy as np
import pytest

from evals.scenarios import make_env

pytest.importorskip("stable_baselines3")


def test_light_subproc_vec_env_matches_dummy():
    """Our torch-free worker has to behave exactly like SB3's own vec envs."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rl.vec_env import LightSubprocVecEnv

    fns = [functools.partial(make_env, ["normal", "peak_season"])] * 2
    light, dummy = LightSubprocVecEnv(fns), DummyVecEnv(fns)
    try:
        light.seed(3)
        dummy.seed(3)
        np.testing.assert_array_equal(light.reset(), dummy.reset())
        rng = np.random.default_rng(0)
        n_done = 0
        for _ in range(130):  # crosses the 60-day episode boundary twice
            act = rng.integers(0, light.action_space.nvec, size=(2, len(light.action_space.nvec)))
            o1, r1, d1, i1 = light.step(act)
            o2, r2, d2, i2 = dummy.step(act)
            np.testing.assert_array_equal(o1, o2)
            # DummyVecEnv keeps rewards in a float32 buffer, SubprocVecEnv (SB3's and ours) returns float64
            np.testing.assert_array_equal(r1.astype(np.float32), r2)
            np.testing.assert_array_equal(d1, d2)
            for a, b in zip(i1, i2):
                assert a["TimeLimit.truncated"] == b["TimeLimit.truncated"]
                if "terminal_observation" in b:
                    np.testing.assert_array_equal(a["terminal_observation"], b["terminal_observation"])
            n_done += int(d1.sum())
        assert n_done >= 2
        assert light.get_attr("t") == dummy.get_attr("t")
    finally:
        light.close()
