import pybullet as p

# Connect to PyBullet
p.connect(p.GUI)

# Load the URDF
robot_id = p.loadURDF("C:/Users/MSI/python code/AR4_MK3/robot22.urdf")

# Run simulation
while True:
    p.stepSimulation()
