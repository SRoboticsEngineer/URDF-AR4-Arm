import pybullet as p

# Connect to PyBullet
p.connect(p.GUI)

# Load the URDF
robot_id = p.loadURDF("robot1.urdf")

# Run simulation
while True:
    p.stepSimulation()
