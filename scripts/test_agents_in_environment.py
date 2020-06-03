from typing import Optional, Union
from pathlib import Path
import random
import time

import numpy as np
import typer
import torch

from unityagents import UnityEnvironment

from scripts.definitions import ROOT_DIR

DEFAULT_ENVIRONMENT_EXECUTABLE_PATH = str(ROOT_DIR / 'unity_reacher_environment/Reacher.x86_64')
DEVICE = torch.device('cpu')


class RandomAgent:
    def __init__(self, num_agents: int, action_size: int):
        self.num_agents = num_agents
        self.action_size = action_size

    def act(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        actions = np.random.randn(self.num_agents, self.action_size)
        return np.clip(actions, -1, 1)


def run_environment(
        environment_path: str = DEFAULT_ENVIRONMENT_EXECUTABLE_PATH,
        agent_parameters_path: Optional[Path] = None,
        random_agent: bool = False
):
    """Run the reacher environment and visualize the actions of the agents.

    Args:
        environment_path: path of the executable which runs the banana environment
        agent_parameters_path: an optional path to load the agent parameters from
    """
    seed = random.randint(0, int(1e6))
    print(environment_path)

    env = UnityEnvironment(file_name=environment_path, seed=seed)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size

    agent: Optional[Union[RandomAgent]] = None
    if random_agent:
        agent = RandomAgent(num_agents=num_agents, action_size=action_size)
    scores = np.zeros(num_agents)

    while True:
        actions = agent.act()
        env_info = env.step(actions)[brain_name]
        dones = env_info.local_done
        scores += env_info.rewards
        time.sleep(0.01)
        if np.any(dones):
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    env.close()


if __name__ == '__main__':
    typer.run(run_environment)
