import mujoco
import mujoco.viewer

# 1. Define the path to the scene file
# 'scene.xml' includes the robot plus the environment (floor, lights)
model_path = "ufactory_lite6/scene.xml"

try:
    # 2. Load the model and create a data object for simulation state
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 3. Launch the interactive viewer
    # This opens a window where you can click and drag the robot
    print("Launching MuJoCo Viewer... Close the window to stop.")
    mujoco.viewer.launch(model, data)

except ValueError as e:
    print(f"Error loading model: {e}")
    print("Make sure the 'ufactory_lite6' folder is in the same directory as this script.")