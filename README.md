# Reacher
## Introduction
This is a solution for the second project of the [Udacity deep reinforcement learning course](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). 
It includes scripts for training agents using any of the A2C, PPO and TD3 algorithms and for testing it in a simulation environment.
The models were trained using the [Stable Baselines3 project](https://stable-baselines3.readthedocs.io/en/master/#).

## Example agents
The giff shows the behavior of multiple agents using a model trained using PPO in this codebase. The agent parameters can be found under `experiments/ppo_multi_agent_lr_0_00003/model.zip`.
![Agent test run](artifacts/ppo_example_run.gif)

## Problem description
The agent consists of an arm with two joints and the environment contains a sphere which is rotating around the agent.
The goal is to keep touching the ball as long as possible during an episode of 1000 timesteps.

- Rewards:
  - +0.04 for each timestep the agent touches the sphere
- Input state:
  - 33 continuous variables corresponding to position, rotation, velocity, and angular velocities of the arm
- Actions:
  - 4 continuous variables, corresponding to torque applicable to two joints with values in [-1.0, 1.0]
- Goal:
  - Get an average score of at least 30 over 100 consecutive episodes
  
- Environments:
    Two environments are available, one with a single agent and one with 20 agents. The evaluation for the 20 agents
    environment differs in that the reward of each episode is the average of the agent rewards. In training the only
    difference is that one can practically simulate 20 environments in one to speed up exploration. 

## Solution
The problem is solved with A2C, PPO and TD3 using the [stable baselines framework](https://stable-baselines3.readthedocs.io/en/master/).
 For more details and a comparison of the algorithms' behavior look in the [corresponding report](<tbd>). 

## Setup project
To setup the project follow those steps:
- Provide an environment with `python 3.6.x` installed, ideally create a new one with e.g. pyenv or conda
- Clone and install the project: 
```
git clone git@github.com:koulakis/reacher-deep-reinforcement-learning.git
cd reacher-reinforcement-learning
pip install .
```
- Create a directory called `udacity_reacher_environment_single_agent` or `udacity_reacher_environment_multi_agent` (to use with the single or 20 agent environments respectively)
under the root of the project and download and extract there the environment compatible with your architecture. 
You can find the [download links here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started).
- Install a version of pytorch compatible with your architecture. The version used to develop the project was 1.5.0.
e.g. `pip install pytorch==1.5.0`

To check that everything is setup properly, run the following test which loads an environment and runs a random agent:
`python scripts/run_agent_in_environment.py --random-agent`

or 

`python scripts/run_agents_in_environment.py --random-agent --agent-type multi`

which run the 20 agents environment.

## Training and testing the agent
The project comes along with some pre-trained agents, scripts to test them and train your own.

### Scripts
- `train_agent.py`: This one is used to train an agent. The parameter `experiment-name` is used to name your agent and
    the script will create by default a directory under `experiments` with the same name. The trained agent parameters
    will be saved there in the end of the training and during training several metrics are logged to a tensorflow events file
    under the same directory. Here is an example call:
    ```python scripts/train_agent.py --device 'cuda:0' --double-dqn --experiment-name my_fancy_agent_using_double_dqn```
    
    To monitor the metrics one can launch a tensorboard server with:
    ```tensorboard --logdir experiments```
    This will read the metrics of all experiments and make the available under `localhost:6006`
    
- `test_agent_in_environment`: This script can be used to test an agent on a given environment. As mentioned above, one
can access the saved agent models inside the sub-folders of `experiments`. An example usage:
    ```python scripts/test_agent_in_environment.py --agent-parameters-path experiments/dqn_training/checkpoint.pth```
    
### Pre-trained models
Under the `experiments` directory there are several pre-trained agents one can used to run in the environment. Some
examples of models which have solved the environment are: 

- Best A2C model: a2c_lr_0_0001/tensorboard_logs/A2C_1
- Best PPO model: ppo_multi_agent_lr_0_00003/tensorboard_logs/PPO_1
- Best PPO model trained with a single agent: ppo_large_128_128_128_lr_0_00005_3M_steps/tensorboard_logs/PPO_1

## References
Given that this project is an assignment of an online course, it has been influenced heavily by code provided by
Udacity and several mainstream publications. Below you can find some links which can give some broader context.

### Frameworks & codebases
1. All 3 algorithms used were trained using the [Stable Baselines3 project](https://stable-baselines3.readthedocs.io/en/master/#)
1. Most of the simulation setup comes from [this notebook](https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/Continuous_Control.ipynb)
1. The unity environment created by Udacity is a direct copy [from here](https://github.com/udacity/deep-reinforcement-learning/tree/master/python)
 
### Publications
The following publications were used:

1. *Asynchronous Methods for Deep Reinforcement Learning*. Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T.P., Harley, T., Silver, D., & Kavukcuoglu, K. arXiv:1602.01783. 2016.
1. *Proximal Policy Optimization Algorithms*. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. arXiv:1707.06347. 2017.
1. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel. arXiv:1506.02438. 2015.
1. *Continuous control with deep reinforcement learning*. Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra. arXiv:1509.02971. 2015.
1. *Addressing Function Approximation Error in Actor-Critic Methods*. Scott Fujimoto, Herke van Hoof, David Meger. 	arXiv:1802.09477. 2018.