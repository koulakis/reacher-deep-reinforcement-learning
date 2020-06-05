import random
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import typer

from reacher.unity_env_wrappers import UnityEnvironmentToGymWrapper, SingleOrMultiAgent


def train(
        total_timesteps: int = 25000,
        output_path: str = 'reacher_a2c',
        input_path: Optional[str] = None,
        agent_type: SingleOrMultiAgent = SingleOrMultiAgent.single_agent,
        env_seed: int = random.randint(0, int(1e6)),
        tensorboard_log: Optional[str] = None,
        environment_port: Optional[int] = None):

    env = UnityEnvironmentToGymWrapper(
        agent_type=agent_type,
        seed=env_seed,
        no_graphics=True,
        train_mode=True,
        environment_port=environment_port)

    if input_path:
        model = PPO.load(input_path, env=env)
    else:
        model = PPO(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_log, device='cpu')

    model.learn(
        total_timesteps=total_timesteps,
        eval_env=env,
        eval_freq=100000,
        n_eval_episodes=5,
        eval_log_path='eval'
    )
    model.save(output_path)


if __name__ == '__main__':
    typer.run(train)
