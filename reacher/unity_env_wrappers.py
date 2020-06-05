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
            **kwargs):
        kwargs['file_name'] = get_environment_path(agent_type)
        kwargs['seed'] = random.randint(0, int(1e6)) if not seed else seed

        self.unity_env = UnityEnvironment(*args, **kwargs)

        self.brain_name = self.unity_env.brain_names[0]
        self.brain = self.unity_env.brains[self.brain_name]
        self.train_mode = train_mode

        action_space_size = self.brain.vector_action_space_size
        observation_space_size = self.brain.vector_observation_space_size
        self.action_space = Box(
            low=-np.ones(action_space_size, dtype=np.float32),
            high=np.ones(action_space_size, dtype=np.float32))
        self.observation_space = Box(
            low=-np.ones(observation_space_size, dtype=np.float32),
            high=np.ones(observation_space_size, dtype=np.float32))
        self.reward_range = (0.0, float('inf'))
        self.number_of_agents = len(self.brain.agents) if hasattr(self.brain, 'agents') else 1

    def step(self, action) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        brain_info = self.unity_env.step(action)[self.brain_name]

        return self._parse_brain_info(brain_info)

    def reset(self) -> np.ndarray:
        brain_info = self.unity_env.reset(train_mode=self.train_mode)[self.brain_name]

        return self._parse_brain_info(brain_info)[0]

    def render(self, mode='human') -> None:
        pass

    def close(self):
        self.unity_env.close()

    @staticmethod
    def _parse_brain_info(info: BrainInfo) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        observation = info.vector_observations[0]
        reward = info.rewards[0]
        done = info.local_done[0]
        info = dict()

        return observation, reward, done, info
