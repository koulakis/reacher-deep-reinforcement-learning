from typing import Optional
import time
from pathlib import Path

import numpy as np
import typer
import torch
from stable_baselines3.common.vec_env import VecNormalize

from scripts.utils import create_environment, RLAlgorithm, algorithm_and_policy
from reacher.unity_environment_wrappers import SingleOrMultiAgent


DEVICE = torch.device('cpu')


class RandomAgent:
    def __init__(self, number_of_agents: int, action_size: int):
        self.number_of_agents = number_of_agents
        self.action_size = action_size

    def act(self, state: np.ndarray) -> np.ndarray:
        actions = np.random.randn(self.number_of_agents, self.action_size)
        return np.clip(actions, -1, 1)


class TrainedAgent:
    def __init__(self, algorithm: RLAlgorithm, parameters_path: str):
        self.model = algorithm_and_policy[algorithm][0].load(parameters_path)

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.model.predict(state)[0]


def run_environment(
        algorithm: RLAlgorithm = typer.Option(...),
        agent_type: SingleOrMultiAgent = SingleOrMultiAgent.single_agent,
        agent_parameters_path: Optional[Path] = None,
        random_agent: bool = False,
        seed: Optional[int] = None,
        environment_port: Optional[int] = None,
        normalize: bool = False,
        n_envs: Optional[int] = None
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
    env = create_environment(
        agent_type=agent_type,
        normalize=False,
        n_envs=n_envs,
        env_seed=seed,
        environment_port=environment_port,
        training_mode=False,
        no_graphics=False)

    if normalize:
        env = VecNormalize.load(str(agent_parameters_path.parent / 'vecnormalize.pkl'), env)

    action_size = env.action_space.shape[0]

    if random_agent:
        agent = RandomAgent(number_of_agents=n_envs, action_size=action_size)
    else:
        agent = TrainedAgent(algorithm=algorithm, parameters_path=str(agent_parameters_path))

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
