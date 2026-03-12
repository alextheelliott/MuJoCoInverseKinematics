import collections
import time
import os

import numpy as np

import mujoco
import mujoco.viewer

Point = collections.namedtuple('Point',['x','y','z'])
Quaternion = collections.namedtuple('Quaternion',['w','x','y','z'])

# --- CONFIGURATION ---
MODEL_PATH = "ufactory_lite6/scene.xml"
# Target Position: (X, Y, Z) in meters
TARGET_POINT = Point(0.4, 0.1, 0.4)
# Target Orientation: (W, X, Y, Z) Quaternion
# (1, 0, 0, 0) is the default orientation from the XML
TARGET_QUAT = Quaternion(0.707, 0.707, 0.0, 0.0)

STEP_GAIN = 0.5
DAMPING = 1e-3 #1e-4 default
# ---------------------

def compute_6dof_ik(model, data, site_id, target_pos: Point, target_quat: Quaternion):
    """Calculates joint velocity (dq) for a 6-DOF target pose."""
    # 1. Position Error
    current_pos = data.site_xpos[site_id]
    pos_error = np.array(target_pos) - current_pos

    # 2. Orientation Error
    current_quat = np.zeros(4)
    mujoco.mju_mat2Quat(current_quat, data.site_xmat[site_id])
    
    # Relative rotation: error = target * inv(current)
    neg_current_quat = np.zeros(4)
    mujoco.mju_negQuat(neg_current_quat, current_quat)
    error_quat = np.zeros(4)
    mujoco.mju_mulQuat(error_quat, np.array(target_quat), neg_current_quat)
    
    # Convert to 3D angular velocity vector
    ori_error = np.zeros(3)
    mujoco.mju_quat2Vel(ori_error, error_quat, 1.0)

    # 3. Full 6D Error
    full_error = np.concatenate([pos_error, ori_error])

    # 4. Jacobian (Linear + Angular)
    jac_pos = np.zeros((3, model.nv))
    jac_ori = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_pos, jac_ori, site_id)
    jac = np.vstack([jac_pos, jac_ori])

    # 5. Solve for the first 6 joints (the arm)
    jac_arm = jac[:, :6]
    n = jac_arm.shape[0]
    dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + DAMPING * np.eye(n)) @ full_error

    return dq

keycodes = {
    262: 'right',
    263: 'left',
    264: 'down',
    265: 'up',
    44: 'comma',
    46: 'period',
    39: 'apostrophe',
    59: 'semicolan',
}

def key_callback(keycode):    
    global TARGET_POINT, TARGET_QUAT

    key = keycodes[keycode]
    if (key == 'right'):
        TARGET_POINT = Point(TARGET_POINT.x+0.02, TARGET_POINT.y, TARGET_POINT.z)
    elif (key == 'left'):
        TARGET_POINT = Point(TARGET_POINT.x-0.02, TARGET_POINT.y, TARGET_POINT.z)
    elif (key == 'up'):
        TARGET_POINT = Point(TARGET_POINT.x, TARGET_POINT.y+0.02, TARGET_POINT.z)
    elif (key == 'down'):
        TARGET_POINT = Point(TARGET_POINT.x, TARGET_POINT.y-0.02, TARGET_POINT.z)
    elif (key == 'period'):
        TARGET_POINT = Point(TARGET_POINT.x, TARGET_POINT.y, TARGET_POINT.z+0.02)
    elif (key == 'comma'):
        TARGET_POINT = Point(TARGET_POINT.x, TARGET_POINT.y, TARGET_POINT.z-0.02)

    print(f"Moving to Position: {TARGET_POINT}")
    print(f"Moving to Orientation: {TARGET_QUAT}")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # Try to find the correct site name for the tip
    try:
        site_id = model.site('link_tcp').id
    except:
        site_id = model.site('end_effector').id

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        print(f"Moving to Position: {TARGET_POINT}")
        print(f"Moving to Orientation: {TARGET_QUAT}")

        while viewer.is_running():
            step_start = time.time()

            # Calculate the update
            dq = compute_6dof_ik(model, data, site_id, TARGET_POINT, TARGET_QUAT)

            # Apply update to the first 6 joints
            data.qpos[:6] += dq * STEP_GAIN
            
            # Set control to maintain the position (prevents gravity drop)
            if model.nu >= 6:
                data.ctrl[:6] = data.qpos[:6]

            # Step physics and update visuals
            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time control
            elapsed = time.time() - step_start
            if model.opt.timestep > elapsed:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()