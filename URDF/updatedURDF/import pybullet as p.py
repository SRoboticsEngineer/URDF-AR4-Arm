import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math
import cv2
from stable_baselines3 import PPO
import time

# Define the Custom Gym Environment
class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        self.physics_client = p.connect(p.DIRECT)  # Use DIRECT for faster training
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load robot and environment objects
        self.robot_id = p.loadURDF(
            "/home/ubuntu/testRos2_ws/src/ar4_ros_driver/annin_ar4_description/urdf/TESTURDF.urdf",  # Path to your upgraded URDF file
    basePosition=[0, 0, 0],
    useFixedBase=True
        )
        p.loadURDF("plane.urdf")
        self.cube_id = p.loadURDF("cube.urdf", basePosition=[0.5, 0, 0.05], globalScaling=0.05)
        self.container_id = p.loadURDF("tray/tray.urdf", basePosition=[-0.5, 0, 0.05], globalScaling=0.5)

        # Define action and observation spaces
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_limits = [(-math.radians(170), math.radians(170)) for _ in range(self.num_joints)]

        # Observation space: Joint positions, velocities, and visual data (flattened image)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_joints * 2 + 64 * 64,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

    def reset(self):
        # Reset joint positions
        for joint_index in range(self.num_joints):
            p.resetJointState(self.robot_id, joint_index, targetValue=0.0)
        return self._get_observation()

    def step(self, action):
        # Scale actions to joint limits
        scaled_action = [
            np.clip(action[i], self.joint_limits[i][0], self.joint_limits[i][1])
            for i in range(len(action))
        ]

        # Apply actions to joints
        for joint_index, joint_position in enumerate(scaled_action):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_position,
                force=50
            )
        p.stepSimulation()

        # Compute reward and check termination
        observation = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        return observation, reward, done, {}

    def _get_observation(self):
        # Joint positions and velocities
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]

        # Simulated camera image
        width, height, rgb_img, _, _ = p.getCameraImage(
            width=64, height=64,
            viewMatrix=p.computeViewMatrix(
                cameraEyePosition=[0, 0, 1],
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=[0, 1, 0]
            ),
            projectionMatrix=p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=3.1
            )
        )
        # Convert image to grayscale and flatten
        grayscale_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).flatten()

        return np.concatenate([positions, velocities, grayscale_img])

    def _compute_reward(self):
        # Reward based on cube proximity to end effector and container
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        container_pos, _ = p.getBasePositionAndOrientation(self.container_id)
        end_effector_state = p.getLinkState(self.robot_id, self.num_joints - 1)
        end_effector_pos = np.array(end_effector_state[0])

        distance_to_cube = np.linalg.norm(end_effector_pos - np.array(cube_pos))
        cube_to_container = np.linalg.norm(np.array(cube_pos) - np.array(container_pos))

        reward = -distance_to_cube  # Penalize distance to cube
        if cube_to_container < 0.1:
            reward += 10  # Bonus for placing cube in container

        return reward

    def _check_done(self):
        # Example termination condition
        return False

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()

# Train and Deploy PPO
if __name__ == "__main__":
    # Initialize environment
    env = RobotEnv()

    # Train PPO agent
    print("Training PPO agent...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Save the trained model
    model.save("ppo_robot")
    print("Model trained and saved.")

    # Load the trained model for deployment
    print("Loading trained model...")
    model = PPO.load("ppo_robot")

    # Test the trained model
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.01)
        if done:
            obs = env.reset()

    env.close()
