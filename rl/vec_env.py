import multiprocessing as mp

from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from rl.env_worker import env_worker


class LightSubprocVecEnv(SubprocVecEnv):
    """SubprocVecEnv whose worker processes don't import torch (see rl/env_worker.py).

    Only __init__ is different: it starts our worker instead of SB3's. Everything else
    (step / reset / close / get_attr ...) is inherited and talks the same protocol.
    env_fns have to be picklable with plain pickle, e.g. functools.partial(make_env, scenarios).
    """

    def __init__(self, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        if start_method is None:
            start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in env_fns])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = ctx.Process(target=env_worker, args=(work_remote, remote, env_fn), daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
