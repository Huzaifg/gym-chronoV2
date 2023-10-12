import gymnasium as gym
import cv2

from typing import Callable
import os

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn
import torch.nn.functional as F


from gym_chrono.envs.driving.cobra_corridor_mefloor import cobra_corridor_mefloor

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        #Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_rew_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        self.logger.record(
            'reward', self.training_env.get_attr('_debug_reward')[0])

        return True


def make_env(rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = cobra_corridor_mefloor()
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq  # Number of training iterations between checkpoints
        self.save_path = save_path  # Directory to save checkpoints

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.model.save(os.path.join(
                self.save_path, f"ppo_checkpoint_{self.num_timesteps}"))
        return True

        
if __name__ == '__main__':
    env_single = cobra_corridor_mefloor()
    check_env(env_single)
    ####### PARALLEL ##################
    
    import torch
    torch.cuda.is_available = lambda : False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_cpu = 8
    n_steps = 500  # Set to make an update after the end of 1 episode (50 s)

    # Set mini batch is the experiences from one episode (50 s) so the whole batch is consumed to make an update
    batch_size = n_steps

    # Set the number of timesteps such that we get 100 updates
    total_timesteps = 100 * n_steps * num_cpu

    policy_kwargs = dict(
                    features_extractor_class=CustomCNN,
                    features_extractor_kwargs=dict(features_dim=128),
                    activation_fn=th.nn.ReLU, 
                    net_arch=dict(pi=[256, 128, 32, 16, 32, 64], 
                                  vf=[256, 128, 32, 16, 32, 64]))

    log_path = "logs/"
    # set up logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    # Vectorized envieroment
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    model = PPO('CnnPolicy', env, learning_rate=5e-4, n_steps=n_steps,batch_size=batch_size, verbose=1, n_epochs=10, policy_kwargs=policy_kwargs,  tensorboard_log=log_path)
    print(model.policy)
    #model.set_logger(new_logger)
    reward_store = []
    std_reward_store = []
    num_of_saves = 100
    training_steps_per_save = total_timesteps // num_of_saves
    print(training_steps_per_save)

    for i in range(num_of_saves):
        print(i)
        model.learn(training_steps_per_save, callback=TensorboardCallback())
        print("tp0")
        checkpoint_dir = 'ppo_checkpoints'
        print("tp1")
        os.makedirs(checkpoint_dir, exist_ok=True)
        mean_reward, std_reward = evaluate_policy(
            model, env_single, n_eval_episodes=5)
        print("tp2")
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        print("tp3")
        reward_store.append(mean_reward)
        std_reward_store.append(std_reward)
        print("tp4")
        model.save(os.path.join(checkpoint_dir, f"ppo_checkpoint{i}"))
        print("tp5")
        model = None
        print("tp5.5")
        model = PPO.load(os.path.join(
            checkpoint_dir, f"ppo_checkpoint{i}"), env)
        print("tp6")
        
        
    # Write the rewards and std_rewards to a file, for plotting purposes
    with open('ppo_rewards.txt', 'w') as f:
        for i in range(len(reward_store)):
            f.write(f"{reward_store[i]} {std_reward_store[i]}\n")
###############################

####### SEQUENTIAL ##################
# model = PPO('MlpPolicy', env, verbose=1).learn(100)
###########################
# model.save(f"PPO_cobra")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
