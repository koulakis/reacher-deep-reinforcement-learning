from typing import Optional
from enum import Enum

import stable_baselines3.ppo as ppo
import stable_baselines3.a2c as a2c
import stable_baselines3.td3 as td3
import stable_baselines3.sac as sac
from stable_baselines3.common.vec_env import VecNormalize

from reacher.unity_environment_wrappers import \
    UnitySingleAgentEnvironmentWrapper, SingleOrMultiAgent, UnityMultiAgentEnvironmentWrapper


class RLAlgorithm(str, Enum):
    ppo = 'ppo'
    a2c = 'a2c'
    td3 = 'td3'
    sac = 'sac'


algorithm_and_policy = {
    RLAlgorithm.ppo: (ppo.PPO, ppo.MlpPolicy),
    RLAlgorithm.a2c: (a2c.A2C, a2c.MlpPolicy),
    RLAlgorithm.td3: (td3.TD3, td3.MlpPolicy),
    RLAlgorithm.sac: (sac.SAC, sac.MlpPolicy)
}


def create_environment(
        agent_type: SingleOrMultiAgent,
        normalize: bool,
        n_envs: int,
        environment_port: int,
        training_mode: bool,
        env_seed: Optional[int] = None,
        no_graphics=True):
    environment_parameters = dict(
        seed=env_seed,
        no_graphics=no_graphics,
        train_mode=training_mode,
        environment_port=environment_port)

    if agent_type == SingleOrMultiAgent.single_agent:
        env = UnitySingleAgentEnvironmentWrapper(**environment_parameters)
    else:
        env = UnityMultiAgentEnvironmentWrapper(n_envs=n_envs, **environment_parameters)

    if normalize:
        env = VecNormalize(env, norm_reward=training_mode)

    return env
