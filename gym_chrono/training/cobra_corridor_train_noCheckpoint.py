import gymnasium as gym

from typing import Callable
import os

from stable_baselines3 import PPO, HerReplayBuffer, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
import torch as th


from gym_chrono.envs.driving.cobra_corridor import cobra_corridor


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
        env = cobra_corridor()
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    # env = cobra_corridor()
    ####### PARALLEL ##################
    # n_updates = total_timesteps // (n_steps * n_envs)
    num_cpu = 12
    n_steps = 500  # Set to make an update after the end of 1 episode (50 s)

    # Set mini batch is the experiences from one episode (50 s) so the whole batch is consumed to make an update
    batch_size = n_steps

    # Set the number of timesteps such that we get 100 updates
    total_timesteps = 100 * n_steps * num_cpu

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[64, 128, 64], vf=[64, 128, 64]))

    log_path = "logs/"
    # Create the vectorized environment
    # set up logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    model = PPO('MlpPolicy', env, learning_rate=1e-3, n_steps=n_steps,
                batch_size=batch_size, verbose=1, n_epochs=10, policy_kwargs=policy_kwargs,  tensorboard_log=log_path)
    print(model.policy)
    model.set_logger(new_logger)
    model.learn(total_timesteps=total_timesteps,
                callback=TensorboardCallback())

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


###############################

####### SEQUENTIAL ##################


# n_steps = 500  # Set to make an update after the end of 1 episode (50 s)

# model = PPO('MlpPolicy', env,n_steps=n_steps, verbose=1)
# model.learn(total_timesteps=total_timesteps)
###########################
    model.save(f"PPO_cobra2")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
