import pybullet as p
import pybullet_data
import time
import math

# Initialize PyBullet Physics Server
physicsClient = p.connect(p.GUI)  # Use p.GUI for visualization
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path for PyBullet data

# Load Ground Plane
p.loadURDF("plane.urdf")

# Set Gravity
p.setGravity(0, 0, -9.8)

# Load Robot URDF
robot_id = p.loadURDF(
    "/home/ubuntu/testRos2_ws/src/ar4_ros_driver/annin_ar4_description/urdf/ar4.urdf",  # Path to your upgraded URDF file
    basePosition=[0, 0, 0],
    useFixedBase=True
)

# Add Objects for Pick and Place
cube_id = p.loadURDF(
    "cube.urdf", 
    basePosition=[0.5, 0, 0.05],  # Adjusted for small size
    globalScaling=0.02  # Very small size
)
sphere_id = p.loadURDF(
    "sphere2.urdf", 
    basePosition=[-0.5, 0, 0.05],  # Adjusted for small size
    globalScaling=0.02  # Very small size
)

# Joint Limits (in degrees, converted to radians)
joint_limits = {
    0: (math.radians(-170), math.radians(170)),
    1: (math.radians(-42), math.radians(90)),
    2: (math.radians(-89), math.radians(52)),
    3: (math.radians(-165), math.radians(165)),
    4: (math.radians(-105), math.radians(105)),
    5: (math.radians(-155), math.radians(155)),
    6: (math.radians(0), math.radians(20)),  # Gripper Jaw 1
    7: (math.radians(-20), math.radians(0))   # Gripper Jaw 2
}

# Create a GUI slider for each joint
joint_sliders = {}
num_joints = p.getNumJoints(robot_id)
for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint_index)
    joint_name = f"Joint {joint_index + 1}"
    lower_limit, upper_limit = joint_limits.get(joint_index, (-3.14, 3.14))  # Default limits
    slider_id = p.addUserDebugParameter(joint_name, lower_limit, upper_limit, 0)
    joint_sliders[joint_index] = slider_id

# Function to move the robot to a target position
def move_robot_to_target(joint_positions):
    for joint_index, position in enumerate(joint_positions):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=position,
            force=50
        )
    p.stepSimulation()

# Pick and Place Function
def pick_and_place():
    try:
        # Move to pick position
        pick_position = [0, math.radians(45), math.radians(-45), math.radians(0), math.radians(90), 0, 0, 0]
        move_robot_to_target(pick_position)
        time.sleep(2)

        # Close gripper (jaw 1 and jaw 2)
        p.setJointMotorControl2(robot_id, 6, p.POSITION_CONTROL, targetPosition=0.02, force=10)  # Gripper Jaw 1
        p.setJointMotorControl2(robot_id, 7, p.POSITION_CONTROL, targetPosition=-0.02, force=10)  # Gripper Jaw 2
        p.stepSimulation()
        time.sleep(1)

        # Move to place position
        place_position = [math.radians(90), math.radians(-30), math.radians(45), math.radians(90), math.radians(-90), 0, 0, 0]
        move_robot_to_target(place_position)
        time.sleep(2)

        # Open gripper
        p.setJointMotorControl2(robot_id, 6, p.POSITION_CONTROL, targetPosition=0, force=10)  # Gripper Jaw 1
        p.setJointMotorControl2(robot_id, 7, p.POSITION_CONTROL, targetPosition=0, force=10)  # Gripper Jaw 2
        p.stepSimulation()
        time.sleep(1)

        print("Pick and Place operation completed.")
    except Exception as e:
        print(f"Error in pick and place: {e}")

# Control Loop
while True:
    # Read sliders and control the joints
    for joint_index, slider_id in joint_sliders.items():
        target_position = p.readUserDebugParameter(slider_id)
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_position,
            force=50
        )
    
    # Add a keypress (or condition) to perform pick and place
    keys = p.getKeyboardEvents()
    if ord('p') in keys:  # Press 'P' to initiate pick and place
        pick_and_place()

    p.stepSimulation()
    time.sleep(0.01)
