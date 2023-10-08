import gymnasium as gym


from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


from gym_chrono.envs.driving.cobra_corridor import cobra_corridor

if __name__ == '__main__':
    env = cobra_corridor()
    check_env(env)

    obs, _ = env.reset()
    env.render()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    # Hardcoded best agent: always go left!
    n_steps = 1000000
    for step in range(n_steps):
        print(f"Step {step + 1}")
        # obs, reward, terminated, truncated, info = env.step(
        #     env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step([0, 0.3])
        done = terminated or truncated
        print("obs=", obs, "reward=", reward, "done=", done)
        env.render()
        if done:
            print("Goal reached!", "reward=", reward)
            break
