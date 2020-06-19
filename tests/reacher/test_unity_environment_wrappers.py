import warnings

from stable_baselines3.common.env_checker import check_env

from reacher.unity_environment_wrappers import UnitySingleAgentEnvironmentWrapper


def test_validate_single_agent_environment_with_baselines_checker():
    env = UnitySingleAgentEnvironmentWrapper(seed=63463, train_mode=False, no_graphics=True, environment_port=7007)

    with warnings.catch_warnings(record=True) as w:
        check_env(env)
        assert len(w) == 0, f'The single agent environment check raised the following warnings: {[str(m) for m in w]}'
