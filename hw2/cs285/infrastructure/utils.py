import numpy as np
import time
import copy
from tqdm import tqdm

import time
from collections import defaultdict

from cs285.infrastructure.parallel_gym import SeedWrapper
from VectorGym import VectorGym

############################################
############################################


def calculate_mean_prediction_error(env, action_sequence, models,
                                    data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0], 0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac, 0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states


def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def mean_squared_error(a, b):
    return np.mean((a - b)**2)


############################################
############################################


def if_async_env(env):
    return isinstance(env, VectorGym) or isinstance(env, SeedWrapper)


def sample_trajectory(env,
                      policy,
                      max_path_length,
                      render=False,
                      render_mode=('rgb_array')):
    func = sample_trajectory_async if if_async_env(
        env) else sample_trajectory_sync
    return func(env=env,
                policy=policy,
                max_path_length=max_path_length,
                render=render,
                render_mode=render_mode)


def sample_trajectory_async(env: VectorGym,
                            policy,
                            max_path_length,
                            render=False,
                            render_mode=('rgb_array')):
    # initialize env for the beginning of a new rollout
    ob = env.reset()

    dones = np.array([False] * env.num_envs)

    def _safe_append(arr, itms, checks, preproc=None):
        for a, it, c in zip(arr, itms, checks):
            if c:
                if preproc is not None:
                    it = preproc(it)
                a.append(it)

    # init vars
    # property: env id
    obs, acs, rewards, next_obs, terminals, image_obs = [
        [[] for _ in range(env.num_envs)] for _ in range(6)
    ]
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            env.render(select=~dones)
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    rendered_im = env.sim.render(camera_name='track',
                                                 height=500,
                                                 width=500,
                                                 select=~dones)
                    _safe_append(image_obs, rendered_im, ~dones,
                                 lambda x: x[::-1])
                else:
                    rendered_im = env.render(mode=render_mode, select=~dones)
                    _safe_append(image_obs, rendered_im, ~dones)
            if 'human' in render_mode:
                env.render(mode=render_mode, select=~dones)
                time.sleep(env.model.opt.timestep)

        # use the most recent ob to decide what to do
        _safe_append(obs, ob, ~dones)
        t_ob = [o for o in ob if o is not None]
        ac = policy.get_action(np.array(t_ob))
        t_ac = [None for _ in range(env.num_envs)]
        val_idx = np.where(~dones)[0]
        for idx, tt_ac in zip(val_idx, ac):
            t_ac[idx] = tt_ac
        _safe_append(acs, t_ac, ~dones)

        # record result of taking that action
        steps += 1

        ob = []
        for i, ret in enumerate(env.step(t_ac, select=~dones)):
            if ret is None:
                ob.append(None)
                # dones[i] = True
                continue
            t_ob, t_rew, t_done, _ = ret

            if t_done:
                dones[i] = True
            ob.append(t_ob)
            next_obs[i].append(t_ob)
            rewards[i].append(t_rew)

        # end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = np.logical_or(steps >= max_path_length, dones)
        terminals.extend(rollout_done)

        if np.all(rollout_done):
            break

    return [
        Path(*path)
        for path in zip(obs, image_obs, acs, rewards, next_obs, terminals)
    ]


def sample_trajectory_sync(env,
                           policy,
                           max_path_length,
                           render=False,
                           render_mode=('rgb_array')):
    # initialize env for the beginning of a new rollout
    ob = env.reset()

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            env.render()
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obs.append(
                        env.sim.render(camera_name='track',
                                       height=500,
                                       width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = 1 if done or steps >= max_path_length else 0
        terminals.append(rollout_done)

        if rollout_done:
            break
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(env,
                        policy,
                        min_timesteps_per_batch,
                        max_path_length,
                        render=False,
                        render_mode=('rgb_array')):
    if_parallel_env = if_async_env(env)

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render,
                                 render_mode)

        if if_parallel_env:
            timesteps_this_batch = sum([p['terminal'] for p in path])
            paths.extend(path)
        else:
            timesteps_this_batch += len(path['terminal'])
            paths.append(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(env,
                          policy,
                          ntraj,
                          max_path_length,
                          render=False,
                          render_mode=('rgb_array')):
    if_parallel_env = if_async_env(env)
    # ntraj = (ntraj // env.num_envs) + 1 if if_parallel_env else ntraj

    step_size = env.num_envs if if_parallel_env else 1

    tbar = tqdm(total=ntraj, desc="Sample")
    paths = []
    traj_cnt = 0
    while traj_cnt < ntraj:
        path = sample_trajectory(env, policy, max_path_length, render,
                                 render_mode)
        paths.append(path)

        tbar.update(step_size)

        traj_cnt += step_size

    tbar.close()

    if if_parallel_env:
        paths = np.concatenate(paths)

    return paths


############################################
############################################


def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32)
    }


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate(
        [path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards


############################################
############################################


def get_pathlength(path):
    return len(path["reward"])


def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean


def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp)  #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] +
                             np.random.normal(0, np.absolute(std_of_noise[j]),
                                              (data.shape[0], )))

    return data