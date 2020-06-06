import random
from typing import Optional

from stable_baselines3.ppo import MlpPolicy, PPO
import typer
from torch import nn

from reacher.unity_env_wrappers import UnityEnvironmentToGymWrapper, SingleOrMultiAgent
from reacher.definitions import ROOT_DIR

EXPERIMENTS_DIR = ROOT_DIR / 'experiments'


def train(
        experiment_name: str = typer.Option(...),
        total_timesteps: int = 500000,
        input_path: Optional[str] = None,
        agent_type: SingleOrMultiAgent = SingleOrMultiAgent.single_agent,
        env_seed: int = random.randint(0, int(1e6)),
        environment_port: Optional[int] = None,
        device: str = 'cpu',
        gamma: float = 0.99,
        learning_rate: float = 5e-5,
        target_kl: float = 0.1,
        policy_layers_comma_sep: str = '128,128,128',
        value_layers_comma_sep: str = '128,128,128'
):
    """Train an agent in the reacher environment."""
    experiment_path = EXPERIMENTS_DIR / experiment_name
    model_path = experiment_path / 'model'
    eval_path = experiment_path / 'eval'
    tensorboard_log_path = experiment_path / 'tensorboard_logs'
    for path in [experiment_path, eval_path, tensorboard_log_path]:
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
        policy_layers = [int(layer_width) for layer_width in policy_layers_comma_sep.split(',')]
        value_layers = [int(layer_width) for layer_width in value_layers_comma_sep.split(',')]

        policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(vf=value_layers, pi=policy_layers)])

        model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            tensorboard_log=str(tensorboard_log_path),
            device=device,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            target_kl=target_kl,
            learning_rate=learning_rate
        )

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
