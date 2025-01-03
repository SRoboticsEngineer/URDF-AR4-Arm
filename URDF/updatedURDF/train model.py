from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from costume_pybullet_env import RoboticArmEnv

# Initialize and check environment
env = RoboticArmEnv()
check_env(env)

# Train PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the model
model.save("robotic_arm_pick_place")
