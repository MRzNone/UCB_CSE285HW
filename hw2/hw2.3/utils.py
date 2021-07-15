import numpy as np


def _safe_append(arr, itms, checks, preproc=None):
    for a, it, c in zip(arr, itms, checks):
        if c:
            if preproc is not None:
                it = preproc(it)
            a.append(it)


class SeedWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def seed(self, seed=None):
        state = np.random.RandomState(seed)

        seeds = state.randint(np.iinfo(np.int32).max, size=self.env.num_envs)
        while len(set(seeds)) != len(seeds):
            seeds = state.randint(np.iinfo(np.int32).max,
                                  size=self.env.num_envs)
        seeds = seeds.astype(np.int32).tolist()

        self.env.seed(seeds)