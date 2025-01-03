import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np

class RoboticArmEnv(gym.Env):
    def __init__(self):
        super(RoboticArmEnv, self).__init__()
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load environment objects
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("/home/ubuntu/testRos2_ws/src/ar4_ros_driver/annin_ar4_description/urdf/TESTURDF.urdf", useFixedBase=True)
        self.object = p.loadURDF("cube.urdf", basePosition=[0.3, 0, 0.05], globalScaling=0.02)  # Smaller and closer object

        self.num_joints = p.getNumJoints(self.robot)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_joints,), dtype=np.float32)
        obs_dim = self.num_joints * 2 + 7  # Joints positions, velocities, object position & orientation
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("/home/ubuntu/testRos2_ws/src/ar4_ros_driver/annin_ar4_description/urdf/TESTURDF.urdf", useFixedBase=True)

        # Load a smaller object closer to the robot
        self.object = p.loadURDF("cube.urdf", basePosition=[0.3, 0, 0.05], globalScaling=0.02)  # Closer and smaller
        return self._get_observation(), {}

    def step(self, action):
        for i in range(len(action)):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=action[i])
        p.stepSimulation()

        # Compute observation, reward, and done flags
        observation = self._get_observation()
        reward, done = self._compute_reward_and_done()
        
        # Ensure `terminated` is a boolean
        terminated = bool(done)  # Task completion condition
        truncated = False  # No truncation logic

        return observation, reward, terminated, truncated, {}

    def _get_observation(self):
        # Get joint positions and velocities
        joint_positions = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        joint_velocities = [p.getJointState(self.robot, i)[1] for i in range(self.num_joints)]
        
        # Get the object's position and orientation
        object_pos, object_orientation = p.getBasePositionAndOrientation(self.object)

        # Return observation as float32
        observation = np.array(joint_positions + joint_velocities + list(object_pos) + list(object_orientation), dtype=np.float32)
        return observation

    def _compute_reward_and_done(self):
        # Reward based on distance to tray center
        object_pos, _ = p.getBasePositionAndOrientation(self.object)
        tray_center = [0.6, 0, 0.1]
        distance_to_tray = np.linalg.norm(np.array(object_pos[:2]) - np.array(tray_center[:2]))

        # Reward is negative distance; done if object is within threshold
        reward = -distance_to_tray
        done = distance_to_tray < 0.05  # Success if object is in the tray
        return reward, done

    def close(self):
        p.disconnect()
