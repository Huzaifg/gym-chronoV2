from typing import Callable
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_chrono.envs.driving.art_wpts import art_wpts


# model = PPO('MlpPolicy', env, verbose=1).learn(300000)
# model.save(f"PPO_tutorial")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# env.reset()


# Vectorized Environments


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
        env = art_wpts()
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    env = art_wpts()
    check_env(env)

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

####### PARALLEL ##################

    # num_cpu = 1
    # # Create the vectorized environment
    # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    # model = PPO('MlpPolicy', env, verbose=1).learn(100000)
###############################

####### SEQUENTIAL ##################
    model = PPO('MlpPolicy', env, verbose=1).learn(100000)
###########################
    model.save(f"PPO_tutorial")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    obs = env.reset()
    n_steps = 10000
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step + 1}")
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print("obs=", obs, "reward=", reward, "done=", done)
        env.render()
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
