from typing import Union, Callable, Tuple
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
import gym
import numpy as np
from tqdm import tqdm

DEVICE = torch.device('cpu')


class ReacherEvaluationCallback(BaseCallback):
    def __init__(
            self,
            eval_env: Union[VecEnv, gym.Env],
            eval_freq: int,
            n_eval_episodes: int,
            n_agents: int,
            eval_path: Path,
            normalization: bool,
            verbose=0):
        super(ReacherEvaluationCallback, self).__init__(verbose)
        self.env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = max(1, int(n_eval_episodes / n_agents))
        self.eval_path = eval_path
        self.best_average_reward = -float('inf')
        self.normalization = normalization
        self.logger = None
        self.last_callback_step = 0

    def _init_callback(self) -> None:
        self.logger = logger

    def _on_step(self) -> bool:
        if 0 < self.eval_freq <= (self.num_timesteps - self.last_callback_step):
            agent_rewards, episode_lengths = evaluate(self.model, self.env, self.n_eval_episodes)

            mean_reward, std_reward = agent_rewards.mean(), agent_rewards.std()
            mean_ep_length, std_ep_length = episode_lengths.mean(), episode_lengths.std()
            self.last_mean_reward = mean_reward

            print(f"Eval num_timesteps={self.num_timesteps}, "
                  f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            self.logger.record('eval/mean_reward', float(mean_reward))
            self.logger.record('eval/mean_ep_length', float(mean_ep_length))

            if mean_reward > self.best_average_reward:
                print(f'New best mean reward: {self.best_average_reward} -> {mean_reward}')

                self.model.save(str(self.eval_path))
                if self.normalization:
                    normalization_path = str(self.eval_path / 'vecnormalize.pkl')
                    self.model.get_vec_normalize_env().save(normalization_path)
                self.best_average_reward = mean_reward

            self.last_callback_step = self.num_timesteps

        return True


def evaluate(
    model: BaseAlgorithm,
    env: Union[VecEnv, gym.Env],
    number_of_episodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate for a given number of episodes."""
    rewards = []
    episode_lengths = []
    for _ in tqdm(list(range(number_of_episodes))):
        state = env.reset()
        reward_cum = 0
        steps = 0
        while True:
            actions = model.predict(state)[0]
            state, reward, done, _ = env.step(actions)
            reward_cum += reward
            steps += 1
            if np.any(done):
                break
        rewards.append(reward_cum)
        episode_lengths.append(steps)

    return np.array(rewards), np.array(episode_lengths)
