from VectorGym import VectorGym
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from agent import Agent
from trainer import Trainer
from utils import SeedWrapper

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',
                        type=str,
                        default='LunarLanderContinuous-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_envs', type=int, default=5)
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--max_path_len', type=int, default=1000)
    parser.add_argument('--num_batch', type=int, default=4000)
    parser.add_argument('--num_eval_batch', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.005)
    args = parser.parse_args()

    params = vars(args)

    env = VectorGym(params['env_name'], params['num_envs'])
    env = SeedWrapper(env)
    logger = SummaryWriter(Path(__file__).parent.joinpath('log'))

    agent = Agent(params, env.observation_space.spaces[0].shape,
                  env.action_space.spaces[0].shape)

    trainer = Trainer(params, env, agent, logger)
    trainer.run()

    env.close()
