# =======================================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2021 projectchrono.org
# All right reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
#
# =======================================================================================
# Authors: Huzaifa Unjhawala
# =======================================================================================
#
# This file contains a script to train a PPO Cobra agent to navigate to a goal point
# A Tensorboard is used for logging of training statistics. The training statistics are
# saved in the logs folder.  Checkpoints are saved in the ppo_checkpoints folder.
#
# =======================================================================================
import gymnasium as gym

from typing import Callable
import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
import torch as th


from gym_chrono.envs.driving.off_road_gator import off_road_gator

# from gym_chrono.train.gatorActorCritic import CustomCNN, CustomActorCriticPolicy

from gym_chrono.train.gatorActorCritic import CustomCombinedExtractor

# from gym.wrappers import FlattenObservation


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
            'reward', self.training_env.get_attr('m_debug_reward')[0])

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
        env = off_road_gator()
        env.set_play_mode()
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    env_single = off_road_gator()
    ####### PARALLEL ##################

    num_cpu = 1
    # Set to make an update after the end of 2 episodes (20 s each)- In total we will have 400 * 12 data points
    n_steps = 20 * 2 * 10
    # Set mini batch is the experiences so that 1/4th  batch is consumed to make an update
    batch_size = n_steps // 4

    # Set the number of timesteps such that we get 200 updates
    total_timesteps = 200 * n_steps * num_cpu

    log_path = "gator_logs/"
    # set up logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    # Vectorized envieroment
    env = SubprocVecEnv([make_env(i)
                        for i in range(num_cpu)],  start_method='fork')
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs={'features_dim': 10},
        net_arch=dict(pi=[40, 20, 10], vf=[40, 20, 10]))
    model = PPO('MultiInputPolicy', env, learning_rate=1e-3, n_steps=n_steps,
                batch_size=batch_size, policy_kwargs=policy_kwargs, verbose=1, n_epochs=10,  tensorboard_log=log_path)

    print(model.policy)
    model.set_logger(new_logger)
    reward_store = []
    std_reward_store = []
    num_of_saves = 100  # Get total 100 saves
    training_steps_per_save = total_timesteps // num_of_saves
    for i in range(num_of_saves):
        model.learn(training_steps_per_save, callback=TensorboardCallback())
        checkpoint_dir = 'gator_ppo_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        mean_reward, std_reward = evaluate_policy(
            model, env_single, n_eval_episodes=10)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        reward_store.append(mean_reward)
        std_reward_store.append(std_reward)
        model.save(os.path.join(checkpoint_dir, f"ppo_checkpoint{i}"))
        model = PPO.load(os.path.join(
            checkpoint_dir, f"ppo_checkpoint{i}"), env)

    # Write the rewards and std_rewards to a file, for plotting purposes
    with open('ppo_rewards.txt', 'w') as f:
        for i in range(len(reward_store)):
            f.write(f"{reward_store[i]} {std_reward_store[i]}\n")


####### SEQUENTIAL ##################
    # model = PPO('MultiInputPolicy', env_single, learning_rate=1e-3, n_steps=n_steps,
    #             batch_size=batch_size, policy_kwargs=policy_kwargs, verbose=1, n_epochs=10,  tensorboard_log=log_path).learn(1000)
###########################
# model.save(f"PPO_cobra")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
