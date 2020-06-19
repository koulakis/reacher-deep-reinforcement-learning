from typing import Optional
import time

import numpy as np
import typer
import torch
from stable_baselines3.ppo import PPO

from reacher.unity_environment_wrappers import \
    UnitySingleAgentEnvironmentWrapper, UnityMultiAgentEnvironmentWrapper, SingleOrMultiAgent


DEVICE = torch.device('cpu')


class RandomAgent:
    def __init__(self, number_of_agents: int, action_size: int):
        self.number_of_agents = number_of_agents
        self.action_size = action_size

    def act(self, state: np.ndarray) -> np.ndarray:
        actions = np.random.randn(self.number_of_agents, self.action_size)
        return np.clip(actions, -1, 1)


class PPOAgent:
    def __init__(self, parameters_path: str = 'reacher_a2c'):
        self.model = PPO.load(parameters_path)

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.model.predict(state)[0]


def run_environment(
        agent_type: SingleOrMultiAgent = SingleOrMultiAgent.single_agent,
        agent_parameters_path: Optional[str] = None,
        random_agent: bool = False,
        seed: Optional[int] = None,
        environment_port: Optional[int] = None
):
    """Run the reacher environment and visualize the actions of the agents.

    Args:
        agent_type: choice between single and multi agent environments
        agent_parameters_path: an optional path to load the agent parameters from
        random_agent: if true, agent(s) use a random policy
        seed: seed for the environment; if not set, it will be picked randomly
        environment_port: the port used from python to communicate with the C# environment backend. By using different
            values, one can run multiple environments in parallel.
    """
    environment_parameters = dict(
        seed=seed,
        train_mode=False,
        no_graphics=False,
        environment_port=environment_port
    )
    env = (
        UnitySingleAgentEnvironmentWrapper(**environment_parameters)
        if agent_type == SingleOrMultiAgent.single_agent
        else UnityMultiAgentEnvironmentWrapper(**environment_parameters))
    
    number_of_agents = env.num_envs
    action_size = env.action_space.shape[0]

    if random_agent:
        agent = RandomAgent(number_of_agents=number_of_agents, action_size=action_size)
    else:
        agent = PPOAgent(agent_parameters_path)

    score = 0
    state = env.reset()
    while True:
        actions = agent.act(state)
        state, reward, done, _ = env.step(actions)
        score += reward
        time.sleep(0.005)
        if np.any(done):
            break

    if agent_type == SingleOrMultiAgent.single_agent:
        print(f'Total score this episode: {score}')
    else:
        print(f'Average total score this episode: {np.array(score).mean()}')

    env.close()


if __name__ == '__main__':
    typer.run(run_environment)
