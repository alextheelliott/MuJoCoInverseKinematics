import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# mjpython launch_robot.py

# --- CONFIGURATION ---
# Set your desired [X, Y, Z] target position here
TARGET_POS = np.array([1, 1, 1]) 
MODEL_PATH = "ufactory_lite6/scene.xml"
# ---------------------

def run_model():
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"File not found: {MODEL_PATH}")
        return

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    site_id = model.site('end_effector').id
    data = calc_ik(model, site_id)

    # Launch Viewer to see the result
    print(f"Final Position: {data.site_xpos[site_id]}")
    print(f"Joint Angles (Radians): {data.qpos[:6]}")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer is running. Press Ctrl+C in terminal to stop.")
        
        # Keep the window alive
        while viewer.is_running():
            step_start = time.time()

            # mj_step keeps the physics engine 'alive' (gravity, etc.)
            #mujoco.mj_step(model, data)

            # Sync the viewer with the data
            viewer.sync()

            # Maintain real-time sync (approx 60fps)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

def calc_ik(model, site_id):
    data = mujoco.MjData(model)

    # IK Parameters
    tol = 1e-4          # Accuracy tolerance
    max_steps = 500     # Maximum iterations
    step_size = 0.5     # How "bold" each step is
    damping = 1e-2      # Prevents erratic movement near singularities

    # The IK Loop (Differential Kinematics)
    # This solves: Δq = J⁺ * Δx
    for i in range(max_steps):
        # Forward kinematics to update current site position
        mujoco.mj_fwdPosition(model, data)
        
        # Calculate the error between current position and target
        current_pos = data.site_xpos[site_id]
        error = TARGET_POS - current_pos
        
        # Check if we are close enough
        if np.linalg.norm(error) < tol:
            print(f"Converged in {i} steps!")
            break

        # Get the Jacobian (the matrix relating joint velocity to tip velocity)
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, site_id)

        # Solve for joint velocities (Δq) using damped least squares
        # Δq = Jᵀ * (J Jᵀ + λ²I)⁻¹ * error
        n = jac.shape[1]
        hessian_inv = np.linalg.inv(jac @ jac.T + damping * np.identity(3))
        dq = jac.T @ hessian_inv @ error
        
        # Update joint positions: q = q + Δq
        # We only update the first 6 joints (the arm)
        data.qpos[:6] += dq[:6] * step_size
        
        # Keep joint positions within their legal limits
        mujoco.mj_normalizeQuat(model, data.qpos)

    return data

if __name__ == "__main__":
    run_model()