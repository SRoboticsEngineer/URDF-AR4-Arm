import pybullet as p
import pybullet_data
import numpy as np
import time
from stable_baselines3 import PPO

# Initialize PyBullet Physics Server
physics_client = p.connect(p.GUI)  # Use GUI for visualization
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Path for PyBullet data

# Load Environment
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")

# Load Robot and Objects
robot_id = p.loadURDF("/home/ubuntu/testRos2_ws/src/ar4_ros_driver/annin_ar4_description/urdf/TESTURDF.urdf", basePosition=[0, 0, 0], useFixedBase=True)
cube_id = p.loadURDF("cube.urdf", basePosition=[0.5, 0, 0.05], globalScaling=0.05)
container_id = p.loadURDF("tray/tray.urdf", basePosition=[-0.5, 0, 0.05], globalScaling=0.5)

# Number of joints in the robot
num_joints = p.getNumJoints(robot_id)

# Load the trained model
print("Loading trained PPO model...")
model = PPO.load("ppo_robot")

# Helper function to reset the robot

def reset_robot():
    for joint_index in range(num_joints):
        p.resetJointState(robot_id, joint_index, targetValue=0.0)

# Helper function to perform actions

def perform_action(action):
    for joint_index, joint_position in enumerate(action):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_position,
            force=50
        )
    p.stepSimulation()

# Helper function to check task success

def is_task_success():
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    container_pos, _ = p.getBasePositionAndOrientation(container_id)
    distance = np.linalg.norm(np.array(cube_pos) - np.array(container_pos))
    return distance < 0.1  # Success if cube is within 10 cm of the container

# Reset the robot
reset_robot()

# Simulation loop
obs = np.zeros((num_joints * 2 + 64 * 64,))  # Mock observation
while True:
    action, _ = model.predict(obs)  # Get action from the trained model
    perform_action(action)  # Execute the action

    # Check for success
    if is_task_success():
        print("Task successfully completed!")
        break

    time.sleep(0.01)  # Slow down simulation for visualization

# Disconnect from PyBullet
p.disconnect()
