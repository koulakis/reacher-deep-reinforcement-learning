from typing import Any, Tuple, Dict, Optional
from enum import Enum
import random

from gym.spaces import Box
from unityagents import UnityEnvironment, BrainInfo
import gym
import numpy as np

from reacher.definitions import ROOT_DIR


class SingleOrMultiAgent(str, Enum):
    single_agent = 'single'
    multi_agent = 'multi'


def get_environment_path(agent: SingleOrMultiAgent) -> str:
    sub_paths_mapping = {
        SingleOrMultiAgent.single_agent: 'unity_reacher_environment_single_agent/Reacher.x86_64',
        SingleOrMultiAgent.multi_agent: 'unity_reacher_environment_multi_agent/Reacher.x86_64'
    }
    return str(ROOT_DIR / sub_paths_mapping[agent])


class UnityEnvironmentToGymWrapper(gym.Env):
    def __init__(
            self,
            *args,
            train_mode: bool = True,
            agent_type: SingleOrMultiAgent,
            seed: Optional[int] = None,
            environment_port: Optional[int] = None,
            **kwargs):
        kwargs['file_name'] = get_environment_path(agent_type)
        kwargs['seed'] = random.randint(0, int(1e6)) if not seed else seed
        if environment_port:
            kwargs['base_port'] = environment_port

        self.unity_env = UnityEnvironment(*args, **kwargs)

        self.brain_name = self.unity_env.brain_names[0]
        self.brain = self.unity_env.brains[self.brain_name]
        self.train_mode = train_mode

        action_space_size = self.brain.vector_action_space_size
        observation_space_size = self.brain.vector_observation_space_size
        self.action_space = Box(
            low=np.array(action_space_size * [-1.0]),
            high=np.array(action_space_size * [1.0]))
        self.observation_space = Box(
            low=np.array(observation_space_size*[-float('inf')]),
            high=np.array(observation_space_size*[float('inf')]))
        self.reward_range = (0.0, float('inf'))
        self.number_of_agents = len(self.brain.agents) if hasattr(self.brain, 'agents') else 1
        self.episode_step = 0
        self.episode_reward = 0.0

    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        brain_info = self.unity_env.step(action)[self.brain_name]
        state, reward, done = self._parse_brain_info(brain_info)

        self.episode_step += 1
        self.episode_reward += reward
        if done:
            info = dict(
                episode=dict(
                    r=self.episode_reward,
                    l=self.episode_step)
            )
        else:
            info = dict()

        return state, reward, done, info

    def reset(self) -> np.ndarray:
        brain_info = self.unity_env.reset(train_mode=self.train_mode)[self.brain_name]
        self.episode_step = 0
        self.episode_reward = 0.0

        return self._parse_brain_info(brain_info)[0]

    def render(self, mode='human') -> None:
        pass

    def close(self):
        self.unity_env.close()

    @staticmethod
    def _parse_brain_info(info: BrainInfo) -> Tuple[Any, float, bool]:
        observation = info.vector_observations[0]
        reward = info.rewards[0]
        done = info.local_done[0]

        return observation, reward, done
