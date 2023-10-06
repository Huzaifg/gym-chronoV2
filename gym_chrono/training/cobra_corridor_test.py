import gymnasium as gym
from stable_baselines3 import A2C, SAC, PPO, TD3
from gym_chrono.envs.driving.art_wpts import art_wpts
from gym_chrono.envs.driving.cobra_corridor import cobra_corridor
from gym_chrono.envs.utils.utils import CalcInitialPose, chVector_to_npArray, SetChronoDataDirectories


env = cobra_corridor()

loaded_model = PPO.load("PPO_cobra")

sim_time = 30
timeStep = 0.1

totalSteps = int(sim_time / timeStep)

obs, _ = env.reset(seed=0)
for step in range(totalSteps):
    action, _states = loaded_model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, teriminated, truncated, info = env.step(action)
    print("obs=", obs, "reward=", reward, "done=", (teriminated or truncated))
    env.render()
    if (teriminated or truncated):
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Time Out", "reward=", reward)
        print("Goal is at: ", env.goal)
        print("Final position is: ", chVector_to_npArray(
            env.rover.GetChassis().GetPos()))
        print("Distance is: ", env._old_distance)
        obs, _ = env.reset(seed=0)
        # break
