import ray
import gym
from functools import partial, reduce
import operator
from copy import deepcopy
import numpy as np
import dill


def ray_nested_get(obj):
    """
    Recursively get all ray ObjectRef and return its deep copy. Free object from ray's shared memory.

    Args:
        obj ([Any]): The ray object to get.
    """
    if obj is None:
        return None

    if isinstance(obj, ray.ObjectRef):
        o = ray.get(obj)
        del obj

        if 'copy' in dir(o):
            o = o.copy()
        else:
            o = deepcopy(o)
        return o

    if isinstance(obj, list) or isinstance(obj, tuple):
        return [ray_nested_get(o) for o in obj]

    return obj


def strip_hidden_func(ls):
    """
    Take out all str stating with "__".
    """
    return list(filter(lambda n: not n.startswith('__'), ls))


def transpose(args):
    if args[0] is None:
        return args
    return list(zip(*args))


class DAO:
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, str(k), v)


@ray.remote(num_cpus=1, num_gpus=0.3)
class ParallelGymEnv(object):
    """
    A ray version of the environment. Forwards all call to the
    environment automatically for methods not starting with __.
    Support properties and functions.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.env = gym.make(*args, **kwargs)

        self._callable_dic = self._get_attr_callable_dict()

    def __dir__(self):
        orig_ls = super().__dir__()

        gym_ls = []
        cur_env = self.env
        while True:
            gym_ls.extend(strip_hidden_func(dir(cur_env)))
            if 'env' in dir(cur_env):
                cur_env = cur_env.env
            else:
                break
        return set(orig_ls + gym_ls)

    def get_attr_callable_dict(self):
        return self._callable_dic

    def _get_attr_callable_dict(self):
        attr_dic = {}
        cur_env = self.env
        while True:
            #             gym_ls += strip_hidden_func(dir(cur_env))
            attr_dic.update({
                k: callable(getattr(cur_env, k))
                for k in strip_hidden_func(dir(cur_env))
            })
            if 'env' in dir(cur_env):
                cur_env = cur_env.env
            else:
                break

        return attr_dic

    def ray_remote_invoke(self, name, *args, **kwargs):
        return getattr(self, name)(*args, **kwargs)

    def __getattr__(self, name):
        """
        Forward any call to env.
        """
        def method(self, name, *args, **kwargs):
            # support gettign attribute
            attr = getattr(self.env, name)

            if callable(attr):
                return attr(*args, **kwargs)
            else:
                # check picklable

                if dill.pickles(attr):
                    return attr
                else:
                    pickable_attrs = {
                        k: getattr(attr, k)
                        for k in dir(attr)
                        if dill.pickles(getattr(attr, k)) and '__' not in k
                    }

                    dao = DAO(pickable_attrs)
                    return dao

        return partial(method, self, name)


class VectorGym(object):
    """
    Make a vectorized gym environment using ray.
    """

    _inited = False

    # specify supported forward func names
    #     SUPPORTED_FORARD_FUNC_DICT = {
    #         '_deal_gym_methods': strip_hidden_func(dir(gym.Env))
    #     }

    #     SUPPORTED_FORARD_FUNC_NAMES = reduce(
    #         operator.concat, list(SUPPORTED_FORARD_FUNC_DICT.values()))

    def __init__(self,
                 env_name,
                 num_envs=1,
                 block=True,
                 return_attr_on_call=True,
                 **kwargs) -> None:

        self.envs = [
            ParallelGymEnv.remote(env_name, **kwargs) for _ in range(num_envs)
        ]
        self.num_envs = num_envs
        self.block = block
        self.return_attr_on_call = return_attr_on_call

        self._env_attr_callable = ray_nested_get(
            self.envs[0].get_attr_callable_dict.remote())

        # make support list
        self._supported_forward_func_dict = {
            '_deal_gym_methods': list(self._env_attr_callable.keys())
        }
        self._supported_forward_func_names = reduce(
            operator.concat, list(self._supported_forward_func_dict.values()))

        VectorGym._inited = True

    def __dir__(self):
        orig_ls = super().__dir__()
        gym_ls = self._supported_forward_func_names

        return set(orig_ls + gym_ls)

    def _deal_gym_methods(self, name):
        """
        Generate function for the call to remote environments.
        """
        def method(self, *args, name=name, select=None, **kwargs):
            if select is None:
                select = [True] * self.num_envs
            assert len(select) == self.num_envs

            # check args are arrays
            for arg in args:
                assert '__iter__' in dir(arg), "Need list arg for vector env"
            for _, v in kwargs:
                assert '__iter__' in dir(v), "Need list arg for vector env"

            # make payload
            payload = []
            for i in range(self.num_envs):
                t_arg = [arg[i] for arg in args]
                t_kwarg = {k: v[i] for k, v in kwargs.items()}

                payload.append((t_arg, t_kwarg))

            # dispatch and wait
            res_remote = [
                env.ray_remote_invoke.remote(name, *t_args, **t_kwargs)
                if t_select else None for env, (
                    t_args,
                    t_kwargs), t_select in zip(self.envs, payload, select)
            ]

            return res_remote

        return partial(method, name=name)

    def __getattr__(self, name: str):
        """
        Forward funtion management. Forward all function calls
        belonging to gym to the remote/parallel environment.
        """
        # find dealer
        dealer_func = None
        for k, v in self._supported_forward_func_dict.items():
            if name in v:
                dealer_func = getattr(self, k)
                break

        # double check funciton is implemented
        if dealer_func is None:
            raise NotImplementedError

        method = dealer_func(name)
        m_res = partial(method, self)

        if self._env_attr_callable[name] and self.block:
            res = lambda *args, **kwargs: transpose(
                ray_nested_get(m_res(*args, **kwargs)))
        elif not self._env_attr_callable[name] and (self.return_attr_on_call
                                                    or self.block):
            # take only the first arg for compitability for regular env
            res = ray_nested_get(m_res())[0]

        return res
