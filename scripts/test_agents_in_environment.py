from typing import Optional, Union
from pathlib import Path
import random
import time
from enum import Enum

import numpy as np
import typer
import torch

from unityagents import UnityEnvironment

from scripts.definitions import ROOT_DIR


class SingleOrMultiAgent(str, Enum):
    single_agent = 'single'
    multi_agent = 'multi'


def get_environment_path(agent: SingleOrMultiAgent) -> str:
    sub_paths_mapping = {
        SingleOrMultiAgent.single_agent: 'unity_reacher_environment_single_agent/Reacher.x86_64',
        SingleOrMultiAgent.multi_agent: 'unity_reacher_environment_multi_agent/Reacher.x86_64'
    }
    return str(ROOT_DIR / sub_paths_mapping[agent])


DEVICE = torch.device('cpu')


class RandomAgent:
    def __init__(self, num_agents: int, action_size: int):
        self.num_agents = num_agents
        self.action_size = action_size

    def act(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        actions = np.random.randn(self.num_agents, self.action_size)
        return np.clip(actions, -1, 1)


def run_environment(
        agent_type: SingleOrMultiAgent = SingleOrMultiAgent.single_agent,
        agent_parameters_path: Optional[Path] = None,
        random_agent: bool = False
):
    """Run the reacher environment and visualize the actions of the agents.

    Args:
        agent_type: choice between single and multi agent environments
        agent_parameters_path: an optional path to load the agent parameters from
        random_agent: if true, agent(s) use a random policy
    """
    seed = random.randint(0, int(1e6))

    env = UnityEnvironment(file_name=get_environment_path(agent_type), seed=seed)

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
