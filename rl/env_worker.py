"""
Worker loop for rl.vec_env.LightSubprocVecEnv.

Same message protocol as the worker in stable_baselines3's SubprocVecEnv (copied from SB3 2.x),
but this module does NOT import torch / SB3. On Windows every worker is a freshly spawned Python
process, and with SB3's own worker each of them loaded torch (~250 MB each). With 16 workers per
run, two trainings at the same time ran my laptop out of memory.
"""
import gymnasium as gym


def _is_wrapped(env, wrapper_class):
    while isinstance(env, gym.Wrapper):
        if isinstance(env, wrapper_class):
            return True
        env = env.env
    return False


def env_worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    reset_info = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = obs
                    obs, reset_info = env.reset()
                remote.send((obs, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                obs, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((obs, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(_is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except (EOFError, KeyboardInterrupt):
            break
