import random
from typing import Optional

from stable_baselines3 import A2C
from stable_baselines3.ppo import MlpPolicy
import typer
from reacher.unity_env_wrappers import UnityEnvironmentToGymWrapper

from scripts.definitions import ROOT_DIR

DEFAULT_ENVIRONMENT_EXECUTABLE_PATH = str(ROOT_DIR / 'unity_reacher_environment_single_agent/Reacher.x86_64')


def train(
        total_timesteps: int = 25000,
        output_path: str = 'reacher_a2c',
        input_path: Optional[str] = None,
        environment_path: str = DEFAULT_ENVIRONMENT_EXECUTABLE_PATH,
        env_seed: int = random.randint(0, int(1e6)),
        tensorboard_log: Optional[str] = None):

    env = UnityEnvironmentToGymWrapper(file_name=environment_path, seed=env_seed, no_graphics=True)

    if input_path:
        model = A2C.load(input_path, env=env)
    else:
        model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_log, device='cpu')

    model.learn(total_timesteps=total_timesteps)
    model.save(output_path)


if __name__ == '__main__':
    typer.run(train)
