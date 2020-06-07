from typing import Any, Tuple, Dict, Optional, List, Union, Sequence
from enum import Enum
import random
import logging

from gym.spaces import Box
from stable_baselines3.common.vec_env import VecEnv
from unityagents import UnityEnvironment, BrainInfo, BrainParameters
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


class UnityEnviromentToVecEnvWrapper(VecEnv):
    def __init__(
            self,
            *args,
            train_mode: bool = True,
            seed: Optional[int] = None,
            environment_port: Optional[int] = None,
            **kwargs):
        self.train_mode = train_mode
        self.unity_env, self.brain_name, self.brain = self.setup_unity_environment(
            *args,
            environment_port=environment_port,
            seed=seed,
            **kwargs)

        action_space_size = self.brain.vector_action_space_size
        observation_space_size = self.brain.vector_observation_space_size
        self.num_envs = len(self._parse_brain_info(
            self.unity_env.reset(train_mode=self.train_mode)[self.brain_name])[1])

        self.action_space = Box(
            low=np.array(action_space_size * [-1.0]),
            high=np.array(action_space_size * [1.0]))
        self.observation_space = Box(
            low=np.array(observation_space_size * [-float('inf')]),
            high=np.array(observation_space_size * [float('inf')]))
        self.reward_range = (0.0, float('inf'))

        super().__init__(self.num_envs, self.observation_space, self.action_space)

        self.episode_steps = 0
        self.episode_rewards = np.zeros(self.num_envs)
        self.actions = None

    def reset(self):
        brain_info = self.unity_env.reset(train_mode=self.train_mode)[self.brain_name]
        new_states, _, dones = self._parse_brain_info(brain_info)

        self.episode_steps = 0
        self.episode_rewards = np.zeros(self.num_envs)

        return new_states

    def step_async(self, actions) -> None:
        self.actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        brain_info = self.unity_env.step(self.actions)[self.brain_name]
        states, rewards, dones = self._parse_brain_info(brain_info)

        self.episode_rewards += rewards
        self.episode_steps += 1

        if any(dones):
            if not all(dones):
                logging.warning('All agent episodes were supposed to finish simultaneously, but this was not the case.'
                                f'{sum(dones)}/{len(dones)} agents done.')
            info = [
                dict(
                    episode=dict(
                        r=self.episode_rewards[i],
                        l=self.episode_steps)
                )
                for i in range(self.num_envs)
            ]
            self.reset()
        else:
            info = [dict() for _ in dones]

        return states, rewards, dones, info

    def close(self):
        self.unity_env.close()

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def get_images(self, *args, **kwargs) -> Sequence[np.ndarray]:
        pass

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        pass

    @staticmethod
    def setup_unity_environment(
            *args,
            environment_port: Optional[int],
            seed: Optional[int],
            **kwargs
    ) -> Tuple[UnityEnvironment, str, BrainParameters]:
        kwargs['file_name'] = get_environment_path(SingleOrMultiAgent.multi_agent)
        kwargs['seed'] = random.randint(0, int(1e6)) if not seed else seed
        if environment_port:
            kwargs['base_port'] = environment_port

        unity_env = UnityEnvironment(*args, **kwargs)
        brain_name = unity_env.brain_names[0]
        brain = unity_env.brains[brain_name]

        return unity_env, brain_name, brain

    @staticmethod
    def _parse_brain_info(info: BrainInfo) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return info.vector_observations, np.array(info.rewards), np.array(info.local_done)
