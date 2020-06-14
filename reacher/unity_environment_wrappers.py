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


SINGLE_AGENT_ENVIRONMENT_PATH = str(ROOT_DIR / 'unity_reacher_environment_single_agent/Reacher.x86_64')
MULTI_AGENT_ENVIRONMENT_PATH = str(ROOT_DIR / 'unity_reacher_environment_multi_agent/Reacher.x86_64')


class UnitySingleAgentEnvironmentWrapper(gym.Env):
    def __init__(
            self,
            *args,
            train_mode: bool = True,
            seed: Optional[int] = None,
            environment_port: Optional[int] = None,
            **kwargs):
        """A wrapper class which translates the given Unity environment to a gym environment. It is setup to work with
        the single agent reacher environment.

        Args:
            *args: arguments which are directly passed to the Unity environment. This is supposed to make the
                the initialization of the wrapper very similar to the initialization of the Unity environment.
            train_mode: toggle to set the unity environment to train mode
            seed: sets the seed of the environment - if not given, a random seed will be used
            environment_port: port of the environment, used to be able to run multiple environment concurrently
        """
        self.train_mode = train_mode
        self.unity_env, self.brain_name, self.brain = _setup_unity_environment(
            *args,
            path=SINGLE_AGENT_ENVIRONMENT_PATH,
            environment_port=environment_port,
            seed=seed,
            **kwargs)

        self.action_space, self.observation_space, self.reward_range = _environment_specs(self.brain)

        self.num_envs = 1
        self.episode_step = 0
        self.episode_reward = 0.0

    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        brain_info = self.unity_env.step(action)[self.brain_name]
        state, reward, done = self._parse_brain_info(brain_info)

        self.episode_reward += reward
        info = (
            dict(episode=dict(
                r=self.episode_reward,
                l=self.episode_step))
            if done else dict())

        self.episode_step += 1
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
        """Extract the state, reward and done information from an environment brain."""
        observation = info.vector_observations[0]
        reward = info.rewards[0]
        done = info.local_done[0]

        return observation, reward, done


class UnityMultiAgentEnvironmentWrapper(VecEnv):
    def __init__(
            self,
            *args,
            n_envs: Optional[int] = None,
            train_mode: bool = True,
            seed: Optional[int] = None,
            environment_port: Optional[int] = None,
            **kwargs):
        """A wrapper class which translates the given Unity environment to a stable baselines' VecEnv type environment.
        It is setup to work with the multiple agent (20) reacher environment. WARNING: The wrapper makes the assumption
        that all agents finish their episode at the same step. During the `step_wait` this implicit information is
        used to reset the environment when any of the agents returns a done signal.

        Args:
            *args: arguments which are directly passed to the Unity environment. This is supposed to make the
                the initialization of the wrapper very similar to the initialization of the Unity environment.
            n_envs: the number of agents used during training. This is applicable only in multi agent training and the
                maximum number of agents is 20. In fact all 20 agents of the unity environment will be active but only
                the first 'n_envs' will take active part in training.
            train_mode: toggle to set the unity environment to train mode
            seed: sets the seed of the environment - if not given, a random seed will be used
            environment_port: port of the environment, used to be able to run multiple environment concurrently
        """
        self.train_mode = train_mode
        self.unity_env, self.brain_name, self.brain = _setup_unity_environment(
            *args,
            path=MULTI_AGENT_ENVIRONMENT_PATH,
            environment_port=environment_port,
            seed=seed,
            **kwargs)

        self.action_space, self.observation_space, self.reward_range = _environment_specs(self.brain)

        self.num_envs = len(self._parse_brain_info(
            self.unity_env.reset(train_mode=self.train_mode)[self.brain_name],
            n_envs)[1])

        super().__init__(self.num_envs, self.observation_space, self.action_space)

        self.episode_steps = 0
        self.episode_rewards = np.zeros(self.num_envs)
        self.actions = None
        self.n_envs = n_envs

    def reset(self):
        brain_info = self.unity_env.reset(train_mode=self.train_mode)[self.brain_name]
        new_states, _, dones = self._parse_brain_info(brain_info, n_envs=self.n_envs)

        self.episode_steps = 0
        self.episode_rewards = np.zeros(self.num_envs)

        return new_states

    def step_async(self, actions) -> None:
        if self.n_envs:
            zero_padded_actions = np.zeros((20, actions.shape[1]), dtype=np.float32)
            zero_padded_actions[:self.n_envs] = actions
            self.actions = zero_padded_actions
        else:
            self.actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        brain_info = self.unity_env.step(self.actions)[self.brain_name]
        states, rewards, dones = self._parse_brain_info(brain_info, n_envs=self.n_envs)

        self.episode_rewards += rewards
        self.episode_steps += 1

        if any(dones):
            if not all(dones):
                logging.warning('All agent episodes were supposed to finish simultaneously, but this was not the case.'
                                f'{sum(dones)}/{len(dones)} agents done.')
            info = [
                dict(episode=dict(
                    r=self.episode_rewards[i],
                    l=self.episode_steps))
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
    def _parse_brain_info(info: BrainInfo, n_envs: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract the states, rewards and dones information from an environment brain."""
        if n_envs:
            return info.vector_observations[:n_envs], \
                   np.array(info.rewards)[:n_envs], \
                   np.array(info.local_done)[:n_envs]
        else:
            return info.vector_observations, np.array(info.rewards), np.array(info.local_done)


def _setup_unity_environment(
        *args,
        path: str,
        environment_port: Optional[int],
        seed: Optional[int],
        **kwargs
) -> Tuple[UnityEnvironment, str, BrainParameters]:
    """Setup a Unity environment and return it and its brain."""
    kwargs['file_name'] = path
    kwargs['seed'] = random.randint(0, int(1e6)) if not seed else seed
    if environment_port:
        kwargs['base_port'] = environment_port

    unity_env = UnityEnvironment(*args, **kwargs)
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]

    return unity_env, brain_name, brain


def _environment_specs(brain: BrainParameters) -> Tuple[Box, Box, Tuple[float, float]]:
    """Extract the action space, observation space and reward range info from an environment brain."""
    action_space_size = brain.vector_action_space_size
    observation_space_size = brain.vector_observation_space_size
    action_space = Box(
        low=np.array(action_space_size * [-1.0]),
        high=np.array(action_space_size * [1.0]))
    observation_space = Box(
        low=np.array(observation_space_size * [-float('inf')]),
        high=np.array(observation_space_size * [float('inf')]))
    reward_range = (0.0, float('inf'))

    return action_space, observation_space, reward_range
