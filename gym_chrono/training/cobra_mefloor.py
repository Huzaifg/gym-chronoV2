import gymnasium as gym
import cv2

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


from gym_chrono.envs.driving.cobra_corridor_mefloor import cobra_corridor_mefloor


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
    ####### PARALLEL ##################
    
    import torch
    torch.cuda.is_available = lambda : False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_cpu = 8
    n_steps = 500  # Set to make an update after the end of 1 episode (50 s)

    # Set mini batch is the experiences from one episode (50 s) so the whole batch is consumed to make an update
    batch_size = n_steps

    # Set the number of timesteps such that we get 100 updates
    total_timesteps = 1000 * n_steps * num_cpu

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[1024, 512, 256, 32, 16, 32, 64], vf=[1024, 512, 256, 32, 16, 32, 64]))

    log_path = "logs/"
    # set up logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    # Vectorized envieroment
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    model = PPO('MlpPolicy', env, learning_rate=5e-4, n_steps=n_steps,
                batch_size=batch_size, verbose=1, n_epochs=10, policy_kwargs=policy_kwargs,  tensorboard_log=log_path)
    print(model.policy)
    model.set_logger(new_logger)
    reward_store = []
    std_reward_store = []
    num_of_saves = 100
    training_steps_per_save = total_timesteps // num_of_saves

    for i in range(num_of_saves):
        print(i)
        model.learn(training_steps_per_save, callback=TensorboardCallback())
        checkpoint_dir = 'ppo_checkpoints'
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
###############################

####### SEQUENTIAL ##################
# model = PPO('MlpPolicy', env, verbose=1).learn(100)
###########################
# model.save(f"PPO_cobra")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
