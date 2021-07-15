import numpy as np
from itertools import accumulate
import torch
from torch import nn
from torch.nn import functional as F

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation='tanh',
    output_activation='identity',
):
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)


class Policy(nn.Module):
    def __init__(self, params, in_dim, out_dim):
        super().__init__()

        if not isinstance(in_dim, int):
            in_dim = np.prod(in_dim)
        if not isinstance(out_dim, int):
            out_dim = np.prod(out_dim)

        self.params = params
        self.out_dim = out_dim
        # policy net
        self.policy_net = build_mlp(in_dim, out_dim * 2, 4, 64).cuda()
        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(),
                                             self.params['lr'])

        # value net
        self.val_net = build_mlp(in_dim, 1, 3, 64).cuda()
        self.val_optim = torch.optim.Adam(self.val_net.parameters(),
                                          self.params['lr'])

    def forward(self, obs):
        pred = self.policy_net(obs)

        mean = pred[:, :self.out_dim]
        std = pred[:, self.out_dim:]

        dist = torch.distributions.Normal(mean, std.exp())

        return dist

    def _preprocess_obs(self, obs):
        if len(obs.shape) > 1:
            obs = obs
        else:
            obs = obs[None]

        return torch.from_numpy(obs).float().cuda()

    def state_val(self, obs):
        with torch.no_grad():
            x = self._preprocess_obs(obs)
            val = self.val_net(x).cpu().detach().numpy()
        return val

    def select_action(self, obs):
        with torch.no_grad():
            x = self._preprocess_obs(obs)
            dist = self.forward(x)
            actions = dist.sample().cpu().detach().numpy()

        return actions

    def update(self, obs, actions, advantages, q_vals):
        obs = torch.from_numpy(obs).float().cuda()
        actions = torch.from_numpy(actions).float().cuda()
        advantages = torch.from_numpy(advantages).float().cuda()
        q_vals = torch.from_numpy(q_vals).float().cuda()

        action_log_probs = self.forward(obs).log_prob(actions)

        if action_log_probs.dim() > 1:
            action_log_probs = action_log_probs.prod(1)
        action_log_probs = action_log_probs.flatten()

        loss = -action_log_probs * advantages.detach()
        loss = loss.sum()

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

        q_vals = (q_vals - q_vals.mean()) / q_vals.std()
        val_pred = self.val_net(obs).flatten()
        val_loss = F.mse_loss(val_pred, q_vals)
        self.val_optim.zero_grad()
        val_loss.backward()
        self.val_optim.step()

        return {
            'policy_loss': loss.cpu().detach().item(),
            'val_loss': val_loss.cpu().detach().item(),
        }


class Agent:
    def __init__(self, params, ob_dim, ac_dim) -> None:
        self.ac_dim = ac_dim
        self.params = params

        self.actor = Policy(params, ob_dim, ac_dim)

    def learn_batch(self, batch):
        obs, image_obs, acs, rewards, next_obs, dones = [
            [np.array(i) for i in b] for b in zip(*batch)
        ]

        # calc q_values
        q_vals = [self._calc_q_values(rews) for rews in rewards]

        # calc advantage
        advantages = self.estimate_with_gae(rewards, obs, next_obs)

        # learn
        obs = np.concatenate(obs)
        acs = np.concatenate(acs)
        q_vals = np.concatenate(q_vals)
        metrics = self.actor.update(obs, acs, advantages, q_vals)

        return metrics

    def select_action(self, obs):
        obs = np.array(obs)
        return self.actor.select_action(obs)

    def state_value(self, obs):
        obs = np.array(obs)
        return self.actor.state_val(obs)

    def estimate_with_gae(self, rewards, obs, next_obs):
        # split obs into chunks

        advantages = np.concatenate([
            self._calculate_gae(rew, ob, next_ob)
            for rew, ob, next_ob in zip(rewards, obs, next_obs)
        ])

        # normalize
        advantages = (advantages - np.mean(advantages)) / np.std(advantages)

        return advantages

    def _calculate_gae(self, rewards, obs, next_obs):
        current_v = self.state_value(obs).flatten()
        next_v = self.state_value(next_obs).flatten()

        deltas = rewards + self.params['gamma'] * next_v - current_v
        gae = self._discounted_cumsum(
            deltas, self.params['gamma'] * self.params['lam'])

        return gae

    def _calc_q_values(self, rewards, rate=None):
        rate = self.params['gamma'] if rate is None else rate

        rewards = np.array(rewards)
        disounted_return = list(
            accumulate(rewards[::-1], lambda ret, rew: rate * ret + rew))
        disounted_return = np.array(disounted_return)[::-1]
        return disounted_return

    def _discounted_cumsum(self, rewards, rate=None):
        """
                Helper function which
                -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
                -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            """
        # HINT1: note that each entry of the output should now be unique,
        # because the summation happens over [t, T] instead of [0, T]
        # HINT2: it is possible to write a vectorized solution, but a solution
        # using a for loop is also fine
        rate = self.gamma if rate is None else rate

        rewards = np.array(rewards)
        disounted_return = list(
            accumulate(rewards[::-1], lambda ret, rew: rate * ret + rew))
        disounted_return = np.array(disounted_return)[::-1]
        return disounted_return
