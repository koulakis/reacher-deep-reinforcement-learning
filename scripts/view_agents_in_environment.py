from typing import Optional
import random
import time

import numpy as np
import typer
import torch
from unityagents import UnityEnvironment
from stable_baselines3.a2c import A2C

from reacher.unity_env_wrappers import UnityEnvironmentToGymWrapper, SingleOrMultiAgent


DEVICE = torch.device('cpu')


class RandomAgent:
    def __init__(self, number_of_agents: int, action_size: int):
        self.number_of_agents = number_of_agents
        self.action_size = action_size

    def act(self, state: np.ndarray) -> np.ndarray:
        actions = np.random.randn(self.number_of_agents, self.action_size)
        return np.clip(actions, -1, 1)


class A2CAgent:
    def __init__(self, parameters_path: str = 'reacher_a2c'):
        self.model = A2C.load(parameters_path)

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.model.predict(state)[0]


def run_environment(
        agent_type: SingleOrMultiAgent = SingleOrMultiAgent.single_agent,
        agent_parameters_path: Optional[str] = None,
        random_agent: bool = False,
        seed: Optional[int] = None
):
    """Run the reacher environment and visualize the actions of the agents.

    Args:
        agent_type: choice between single and multi agent environments
        agent_parameters_path: an optional path to load the agent parameters from
        random_agent: if true, agent(s) use a random policy
        seed: seed for the environment; if not set, it will be picked randomly
    """
    env = UnityEnvironmentToGymWrapper(agent_type=agent_type, seed=seed, train_mode=False)
    number_of_agents = env.number_of_agents
    action_size = env.action_space.shape[0]

    if random_agent:
        agent = RandomAgent(number_of_agents=number_of_agents, action_size=action_size)
    else:
        agent = A2CAgent(agent_parameters_path)

    score = 0
    state = env.reset()
    while True:
        actions = agent.act(state)
        state, reward, done, _ = env.step(actions)
        score += reward
        time.sleep(0.005)
        if np.any(done):
            break
    print(f'Total score this episode: {score}')

    env.close()


if __name__ == '__main__':
    typer.run(run_environment)
