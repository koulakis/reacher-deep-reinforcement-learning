import typer

from scripts.train_agent import train, RLAlgorithm, SingleOrMultiAgent


def run_trainings(
        experiment_name: str = typer.Option(...),
        algorithm: RLAlgorithm = typer.Option(...),
        port: int = 5005):
    if algorithm == RLAlgorithm.ppo:
        train(
            experiment_name=experiment_name,
            agent_type=SingleOrMultiAgent.multi_agent,
            rl_algorithm=algorithm,
            normalize=True,
            n_envs=16,
            total_timesteps=int(2e6),
            batch_size=128,
            n_steps=512,
            gamma=0.99,
            ppo_a2c_gae_lambda=0.9,
            ppo_n_epochs=20,
            learning_rate=3e-5,
            ppo_clip_range=0.4,
            policy_layers_comma_sep='256,256',
            value_layers_comma_sep='256,256',
            log_std_init=-2,
            ppo_a2c_ortho_init=True,
            ppo_target_kl=None,
            environment_port=port,

            use_sde=True,
            sde_sample_freq=4
        )
    elif algorithm == RLAlgorithm.a2c:
        train(
            experiment_name=experiment_name,
            agent_type=SingleOrMultiAgent.multi_agent,
            rl_algorithm=algorithm,
            normalize=True,
            total_timesteps=int(5e6),
            n_steps=8,
            gamma=0.99,
            ppo_a2c_gae_lambda=0.9,
            learning_rate=3e-4,
            policy_layers_comma_sep='64,64',
            value_layers_comma_sep='64,64',
            environment_port=port,

            use_sde=True
        )
    elif algorithm == RLAlgorithm.td3:
        train(
            experiment_name=experiment_name,
            agent_type=SingleOrMultiAgent.single_agent,
            rl_algorithm=algorithm,
            total_timesteps=int(1e6),
            gamma=0.98,
            td3_sac_buffer_size=200000,
            td3_sac_learning_starts=10000,
            td3_noise_type='normal',
            td3_noise_std=0.1,
            learning_rate=1e-3,
            policy_layers_comma_sep='400,300',
            value_layers_comma_sep='400,300',
            environment_port=port
        )
    elif algorithm == RLAlgorithm.sac:
        train(
            experiment_name=experiment_name,
            agent_type=SingleOrMultiAgent.single_agent,
            rl_algorithm=algorithm,
            total_timesteps=int(5e5),
            learning_rate=7.3e-4,
            td3_sac_buffer_size=300000,
            batch_size=256,
            gamma=0.98,
            sac_tau=0.02,
            sac_train_freq=64,
            td3_sac_gradient_steps=64,
            td3_sac_learning_starts=10000,
            log_std_init=-3,
            policy_layers_comma_sep='400,300',
            value_layers_comma_sep='400,300',
            environment_port=port,

            use_sde=True
        )
    else:
        # noinspection PyUnresolvedReferences
        raise ValueError(f'Please select an algorithm from {[a.name for a in RLAlgorithm]}')


if __name__ == '__main__':
    typer.run(run_trainings)
