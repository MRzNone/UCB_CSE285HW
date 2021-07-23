import numpy as np
import torch
from tqdm import trange

from utils import _safe_append


class Trainer:
    def __init__(self, params, env, agent, logger=None) -> None:
        self.env = env
        self.agent = agent
        self.params = params
        self.logger = logger

        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.seed(seed)

    def run(self):
        for itr in range(self.params['num_iters']):
            # collect training paths
            paths = self.sample_trajectory(self.agent,
                                           self.params['max_path_len'],
                                           self.params['num_batch'])

            # train
            train_metrics = self.agent.learn_batch(paths)

            # calc_metrics
            metrics = {'train/' + k: v for k, v in train_metrics.items()}
            metrics['train/ep_len'] = np.mean([len(p[0]) for p in paths])
            metrics['train/ep_len_std'] = np.std([len(p[0]) for p in paths])
            metrics['train/rewards'] = np.mean([np.sum(p[3]) for p in paths])
            metrics['train/rewards_std'] = np.std(
                [np.sum(p[3]) for p in paths])
            metrics['train/max_ep_len'] = np.max([len(p[0]) for p in paths])

            # Eval
            eval_paths = self.sample_trajectory(self.agent,
                                                self.params['max_path_len'],
                                                self.params['num_eval_batch'])
            metrics['eval/ep_len'] = np.mean([len(p[0]) for p in eval_paths])
            metrics['eval/ep_len_std'] = np.std(
                [len(p[0]) for p in eval_paths])
            metrics['eval/rewards'] = np.mean(
                [np.sum(p[3]) for p in eval_paths])
            metrics['eval/rewards_std'] = np.std(
                [np.sum(p[3]) for p in eval_paths])
            metrics['eval/max_ep_len'] = np.max(
                [len(p[0]) for p in eval_paths])

            if self.logger is not None:
                for k, v in metrics.items():
                    self.logger.add_scalar(k, v, itr)

    def sample_trajectory(self, agent, max_path_len, num_batch):

        paths = []
        for _ in trange(num_batch // self.env.num_envs + 1):
            # initialize arrays to record path
            obs, acs, rewards, next_obs, terminals, image_obs = [
                [[] for _ in range(self.env.num_envs)] for _ in range(6)
            ]

            # keep track of done for only running unfinished env
            done = np.array([False] * self.env.num_envs)

            ob = self.env.reset()
            step = 0
            while True:
                num_alive = sum(~done)

                # render and store images
                # rendered_im = self.env.render(select=~done)
                # _safe_append(image_obs, rendered_im, ~done)

                # get observations from running envs
                t_ob = [o for o in ob if o is not None]
                # get actions
                actions_pred = agent.select_action(t_ob)
                # add 'None' paddings back for the API
                actions = [None] * self.env.num_envs
                val_idx = np.where(~done)[0]
                for idx, t_ac in zip(val_idx, actions_pred):
                    actions[idx] = t_ac
                # store actions
                _safe_append(acs, actions, ~done)

                #[(observation, reward, done, info), (observation, reward, done, info)]
                ret = self.env.step(actions, select=~done)

                _safe_append(obs, ob, ~done)
                # process the transition (return from step) to avoid the None(s)
                ob = []
                for i, ret in enumerate(self.env.step(actions, select=~done)):
                    if ret is None:
                        ob.append(None)
                        continue
                    t_ob, t_rew, t_done, _ = ret

                    if t_done:
                        done[i] = True

                    ob.append(t_ob)
                    next_obs[i].append(t_ob)
                    rewards[i].append(t_rew)

                step += 1
                # end the rollout if the rollout ended
                # HINT: rollout can end due to done, or due to max_path_length
                rollout_done = np.logical_or(step >= max_path_len, done)
                terminals.extend(rollout_done)

                if np.all(rollout_done):
                    break
            t_paths = [
                path for path in zip(obs, image_obs, acs, rewards, next_obs,
                                     terminals)
            ]
            paths.extend(t_paths)

        return paths