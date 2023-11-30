import gymnasium as gym


from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


from gym_chrono.envs.driving.off_road_gator import off_road_gator

render = True
if __name__ == '__main__':
    # Add the agent POV as a render mode
    if render:
        env = off_road_gator(render_mode='agent_pov')
    else:
        env = off_road_gator()
    # check_env(env)
    # Set the mode to render for nice viz
    env.set_play_mode()
    obs, _ = env.reset()
    if render:
        env.render('follow')

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    # Hardcoded best agent: always go left!
    n_steps = 1000000
    for step in range(n_steps):
        print(f"Step {step + 1}")
        obs, reward, terminated, truncated, info = env.step([0.0, 0.7, 0.])
        print("Terminated=", terminated, "Truncated=", truncated)
        done = terminated or truncated
        # print("obs=", obs, "reward=", reward, "done=", done)
        if render:
            env.render('follow')
        if done:
            print("reward=", reward)
            break
