from stable_baselines3.common.env_checker import check_env
from costume_pybullet_env import RoboticArmEnv

env = RoboticArmEnv()
obs = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Random action for testing
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    if terminated:
        print("Task completed!")
        break

env.close()

