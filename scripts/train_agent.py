import random
from typing import Optional

import typer
from stable_baselines3.common.noise import NormalActionNoise

from torch import nn
import numpy as np

from scripts.utils import create_environment, RLAlgorithm, algorithm_and_policy
from reacher.unity_environment_wrappers import SingleOrMultiAgent
from reacher.definitions import ROOT_DIR
from reacher.callbacks import ReacherEvaluationCallback

EXPERIMENTS_DIR = ROOT_DIR / 'experiments'


def train(
        experiment_name: str = typer.Option(...),
        total_timesteps: int = 3000000,
        input_path: Optional[str] = None,
        agent_type: SingleOrMultiAgent = SingleOrMultiAgent.single_agent,
        env_seed: int = random.randint(0, int(1e6)),
        environment_port: int = 5005,
        device: str = 'cuda',
        gamma: float = 0.99,
        learning_rate: float = 5e-5,
        policy_layers_comma_sep: str = '128,128,128',
        value_layers_comma_sep: str = '128,128,128',
        eval_freq: int = 100000,
        n_eval_episodes: int = 40,
        rl_algorithm: RLAlgorithm = RLAlgorithm.ppo,
        n_envs: Optional[int] = None,
        batch_size: Optional[int] = None,
        n_steps: Optional[int] = None,
        ppo_target_kl: Optional[float] = 0.1,
        ppo_a2c_gae_lambda: float = 0.95,
        ppo_n_epochs: int = 10,
        ppo_clip_range: float = 0.2,
        log_std_init: Optional[float] = None,
        ppo_a2c_ortho_init: Optional[bool] = None,
        td3_sac_buffer_size: Optional[int] = None,
        sac_tau: Optional[float] = None,
        sac_train_freq: Optional[int] = None,
        td3_sac_gradient_steps: Optional[int] = None,
        td3_sac_learning_starts: Optional[int] = None,
        td3_noise_type: Optional[str] = None,
        td3_noise_std: Optional[float] = None,
        use_sde: Optional[bool] = None,
        sde_sample_freq: Optional[int] = None,
        normalize=False,
        normalize_advantage: Optional[bool] = None,
        use_rms_prop: Optional[bool] = None,
        activation_function: Optional[str] = None
):
    """Train an agent in the reacher environment.

    Args:
        experiment_name: the name of the experiment which will be used to create a directory under 'experiments' and
            store there all training artifacts along with the final and best models
        total_timesteps: the number of timestamps to run till stopping training
        input_path: in case provided, the model from that path is loaded - this is used to continue a previous training
        agent_type: specifies whether to use the environment with one agent or the environment with 20 agents
        env_seed: a seed for the environment random initialization - if not set, defaults to random
        environment_port: this is the port used by the unity environment to communicate with the C# backend. One needs
            to set different ports to different environments which run in parallel.
        device: the device used to train the model, can be 'cpu' or 'cuda:x'
        gamma: the discount rate applied to future actions
        learning_rate: the learning rate used by the policy and value network optimizer
        ppo_target_kl: an upper limit to the target KL divergence. This violates a bit the idea of PPO to reduce the
            amount of hyper-parameters but can still be useful since the agents can still experience catastrophic
            forgetting if this value becomes to high. The idea is to use it as a safe-guard, rather than a tunable
            hyper-parameter.
        policy_layers_comma_sep: a sequence of layer width for the policy network as a comma-separated list
        value_layers_comma_sep: a sequence of layer width for the value network as a comma-separated list
        eval_freq: the number of steps after which a validation round will take place. Whenever there is an improvement,
            the best model will be saved under the 'eval' directory in the experiment. Available only for the single
            agent environment.
        n_eval_episodes: number of episodes run during evaluation, available only for the single agent environment
        rl_algorithm: the algorithm used to train an agent
        n_envs: the number of agents used during training. This is applicable only in multi agent training and the
            maximum number of agents is 20. In fact all 20 agents of the unity environment will be active but only
            the first 'n_envs' will take active part in training.
        batch_size: the batch size used during training
        n_steps: number of steps run during rollout
    """
    experiment_path = EXPERIMENTS_DIR / experiment_name
    model_path = experiment_path / 'model'
    eval_path = experiment_path / 'eval'
    tensorboard_log_path = experiment_path / 'tensorboard_logs'
    for path in [experiment_path, eval_path, tensorboard_log_path]:
        path.mkdir(exist_ok=True, parents=True)

    env = create_environment(
        agent_type=agent_type,
        normalize=normalize,
        n_envs=n_envs,
        env_seed=env_seed,
        environment_port=environment_port,
        training_mode=True)

    algorithm_class, policy = algorithm_and_policy[rl_algorithm]

    if input_path:
        model = algorithm_class.load(input_path, env=env)
    else:
        policy_layers = [int(layer_width) for layer_width in policy_layers_comma_sep.split(',')]
        value_layers = [int(layer_width) for layer_width in value_layers_comma_sep.split(',')]

        net_arch = (
            policy_layers
            if rl_algorithm in [RLAlgorithm.td3, RLAlgorithm.sac]
            else [dict(vf=value_layers, pi=policy_layers)])

        policy_kwargs = remove_none_entries(dict(
            activation_fn=nn.ReLU if activation_function == 'ReLU' else None,
            net_arch=net_arch,
            log_std_init=log_std_init,
            ortho_init=ppo_a2c_ortho_init)
        )

        if rl_algorithm == RLAlgorithm.ppo:
            algorithm_specific_parameters = dict(
                target_kl=ppo_target_kl,
                gae_lambda=ppo_a2c_gae_lambda,
                n_epochs=ppo_n_epochs,
                clip_range=ppo_clip_range
            )
        elif rl_algorithm == RLAlgorithm.a2c:
            algorithm_specific_parameters = dict(
                normalize_advantage=normalize_advantage,
                use_rms_prop=use_rms_prop
            )
        elif rl_algorithm == RLAlgorithm.sac:
            algorithm_specific_parameters = dict(
                buffer_size=td3_sac_buffer_size,
                tau=sac_tau,
                train_freq=sac_train_freq,
                gradient_steps=td3_sac_gradient_steps,
                learning_starts=td3_sac_learning_starts
            )
        elif rl_algorithm == RLAlgorithm.td3:
            action_shape = (env.num_envs, env.action_space.shape[0])
            action_noise = (
                NormalActionNoise(
                    np.zeros(action_shape, dtype=np.float32),
                    td3_noise_std * np.ones(action_shape, dtype=np.float32))
                if td3_noise_type == 'normal'
                else None)
            algorithm_specific_parameters = remove_none_entries(dict(
                buffer_size=td3_sac_buffer_size,
                gradient_steps=td3_sac_gradient_steps,
                learning_starts=td3_sac_learning_starts,
                action_noise=action_noise)
            )
        else:
            algorithm_specific_parameters = dict()

        model_optional_parameters = remove_none_entries(dict(
            batch_size=batch_size,
            n_steps=n_steps,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq
        ))

        model = algorithm_class(
            policy,
            env,
            verbose=1,
            tensorboard_log=str(tensorboard_log_path),
            device=device,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            **model_optional_parameters,
            **remove_none_entries(algorithm_specific_parameters)
        )

    eval_callback = ReacherEvaluationCallback(
        eval_env=env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        n_agents=n_envs if n_envs else 1,
        eval_path=eval_path,
        normalization=normalize
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback]
    )

    model.save(str(model_path / model))
    model.get_vec_normalize_env().save(str(model_path / 'vecnormalize.pkl'))


def remove_none_entries(d):
    return {k: v for k, v in list(d.items()) if v is not None}


if __name__ == '__main__':
    typer.run(train)
