import gymnasium as gym


from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import matplotlib.pyplot as plt
import cv2


from gym_chrono.envs.driving.cobra_corridor_mefloor import cobra_corridor_mefloor

if __name__ == '__main__':
    env = cobra_corridor_mefloor()
    check_env(env)

    obs, _ = env.reset()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    # Hardcoded best agent: always go left!
    n_steps = 1000000
    for step in range(n_steps):
        print(f"Step {step + 1}")
        # obs, reward, terminated, truncated, info = env.step(
        #     env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step([0.0, 0.3])
        done = terminated or truncated
        print("obs=", obs, "reward=", reward, "done=", done)
        env.render(mode='rgb_array')
        frame = env.render(mode='rgb_array')
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Rotate the frame by 180 degrees
        rotated_frame = cv2.rotate(bgr_frame, cv2.ROTATE_180)
        cv2.imshow('Frame', rotated_frame)
        cv2.waitKey(1)  # Wait for a key press to close the window
        
        if done:
            print( "reward=", reward)
            break
