import gymnasium as gym

from typing import Callable
import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import cv2


if __name__ == '__main__':
    # env = cobra_corridor()
    ####### PARALLEL ##################

    num_cpu = 8
    # Create the vectorized environment
    env = gym.make("LunarLander-v2",render_mode='human')

    model = PPO('MlpPolicy', env, learning_rate=1e-3, verbose=1)

    for i in range(100):
        print(i)
        model.learn(30000)
        checkpoint_dir = 'ppo_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save(os.path.join(checkpoint_dir, f"ppo_checkpoint{i}"))
        model = PPO.load(os.path.join(
            checkpoint_dir, f"ppo_checkpoint{i}"), env)
        env.render()

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
###############################

####### SEQUENTIAL ##################
# model = PPO('MlpPolicy', env, verbose=1).learn(100)
###########################
# model.save(f"PPO_cobra")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
