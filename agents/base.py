class Agent:
    """Minimal interface every planner implements.

    reset(env, seed) is called after env.reset(). Agents may read *static* network data
    from env.unwrapped.params (lead times, lot sizes, capacities...) but anything that
    changes during the episode has to come from the observation.
    """

    name = "agent"

    def reset(self, env, seed=None):
        pass

    def act(self, obs):
        raise NotImplementedError
