import random
from typing import Optional

from stable_baselines3.ppo import MlpPolicy, PPO
import typer

from reacher.unity_env_wrappers import UnityEnvironmentToGymWrapper, SingleOrMultiAgent
from reacher.definitions import ROOT_DIR

EXPERIMENTS_DIR = ROOT_DIR / 'experiments'


def train(
        experiment_name: str,
        total_timesteps: int = 500000,
        input_path: Optional[str] = None,
        agent_type: SingleOrMultiAgent = SingleOrMultiAgent.single_agent,
        env_seed: int = random.randint(0, int(1e6)),
        tensorboard_log: Optional[str] = None,
        environment_port: Optional[int] = None,
        device: str = 'cuda'):
    """Train an agent in the reacher environment."""
    experiment_path = EXPERIMENTS_DIR / experiment_name
    model_path = experiment_path / 'model'
    eval_path = experiment_path / 'eval'
    for path in [experiment_path, model_path, eval_path]:
        path.mkdir(exist_ok=True, parents=True)

    env = UnityEnvironmentToGymWrapper(
        agent_type=agent_type,
        seed=env_seed,
        no_graphics=True,
        train_mode=True,
        environment_port=environment_port)

    if input_path:
        model = PPO.load(input_path, env=env)
    else:
        model = PPO(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_log, device=device)

    model.learn(
        total_timesteps=total_timesteps,
        eval_env=env,
        eval_freq=100000,
        n_eval_episodes=5,
        eval_log_path=str(eval_path)
    )
    model.save(str(model_path))


if __name__ == '__main__':
    typer.run(train)
