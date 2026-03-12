# MuJoCo Inverse Kinematics Demo

by **Alex Elliott** for **UBC Mech 464**

### Libraries:
Python 3.12
- Used to launch MuJoCo and manage the instance
- Used to add keyboard commands to move the end effector target locations
    - Right arrow: Jog +X
    - Left arrow: Jog -X
    - Up arrow: Jog +Y
    - Down arrow: Jog -Y
    - Period arrow: Jog +Z
    - Comma arrow: Jog -Z

[MuJoCo](https://github.com/google-deepmind/mujoco)
- Used to simulate the arm and physics

[UFactory Lite 6 Arm](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_lite6)
- Premade arm model, with functional joints