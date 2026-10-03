# Robotic Systems I Project (ECE_DK808)

**Perception-Guided Arm Control in Unstructured Scenes**

A 7-DOF Elephant Robotics myArm with a custom suction gripper autonomously finds red cubes on a table (including stacked ones), picks them up one by one and drops them off, using an Intel RealSense RGB-D camera for perception and ROS 2 for control.

## How it works

1. **Camera calibration** – an ArUco marker at a known pose gives the camera-to-robot transform.
2. **Cube detection** – red cubes are segmented in HSV and combined with depth to locate the centre of each top face in the robot frame.
3. **Pre-grasp** – QP-based inverse kinematics (OSQP, multi-start, joint limits) moves the suction cup above the target cube.
4. **Visual servoing** – the cube's top face and blue tape on the cup are tracked to correct the remaining XY misalignment before descending.
5. **Pick & drop** – the suction is switched on, the cube is lifted, carried to a drop pose and released.
6. **Grasp check** – the scene is re-scanned; if the cube count went down the grasp succeeded, otherwise it retries. Unreachable cubes are skipped.

## Hardware

- Elephant Robotics myArm (7-DOF) with its onboard Raspberry Pi
- Intel RealSense D435i, connected to the laptop
- Suction pump and valve driven through a relay module on the Pi's GPIO
- 3D-printed suction kit mount (`suction_kit_base_stl/`)

## License

[MIT](LICENSE)
