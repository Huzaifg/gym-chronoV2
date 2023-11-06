import gymnasium as gym
from stable_baselines3 import A2C, SAC, PPO, TD3
from gym_chrono.envs.driving.art_wpts import art_wpts
from gym_chrono.envs.driving.cobra_corridor import cobra_corridor
from gym_chrono.envs.driving.cobra_corridor_mefloor import cobra_corridor_mefloor
from stable_baselines3.common.evaluation import evaluate_policy
from gym_chrono.envs.utils.utils import CalcInitialPose, chVector_to_npArray, SetChronoDataDirectories
import cv2

import os


env = cobra_corridor_mefloor()


loaded_model = PPO.load("ppo_checkpoint24")

# i = 50
# checkpoint_dir = 'ppo_checkpoints'

# loaded_model = PPO.load(os.path.join(
#     checkpoint_dir, f"ppo_checkpoint{i}"), env)

# mean_reward, std_reward = evaluate_policy(
#     loaded_model, env, n_eval_episodes=5)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

sim_time = 50
timeStep = 0.1

totalSteps = int(sim_time / timeStep)

obs, _ = env.reset(seed=0)
for step in range(totalSteps):
    action, _states = loaded_model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, teriminated, truncated, info = env.step(action)
    print("Distance to goal: ", env._old_distance)
    print("reward=", reward, "done=", (teriminated or truncated))
    
    env.render(mode='rgb_array')
    frame = env.render(mode='rgb_array')
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Rotate the frame by 180 degrees
    rotated_frame = cv2.rotate(bgr_frame, cv2.ROTATE_180)
    cv2.imshow('Frame', rotated_frame)
    cv2.waitKey(1)  # Wait for a key press to close the window
    
    if (teriminated or truncated):
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        # print("Time Out", "reward=", reward)
        print("Goal is at: ", env.goal)
        print("Final position is: ", chVector_to_npArray(
            env.rover.GetChassis().GetPos()))
        print("Distance is: ", env._old_distance)
        obs, _ = env.reset(seed=0)
        break
