# Running Instructions

## What's in this folder

| Folder / File | What it is |
|---|---|
| `project3-team3_workspace/` | ROS 2 workspace (laptop) |
| `myarm_workspace/` | Two TCP servers that run on the MyArm Raspberry Pi |
| `suction_kit_base_stl/` | 3D-printed suction kit mount (STL files) |
| `project3-team3_report.pdf` | Project report |

---

## Laptop — Python dependencies

Any ROS 2 distro works (Humble, Iron, Jazzy, etc.). The arm is controlled over TCP, not ROS 2 inter-machine communication, so the ROS version on the Pi does not need to match.

```bash
pip install numpy opencv-contrib-python pyrealsense2 scipy qpsolvers[osqp]
```

Also install the ROS 2 Python client and vision packages (usually already present):
```bash
sudo apt install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-realsense2-camera
```

---

## How to run

### Step 1 — Start the TCP servers on the Pi

SSH into the Pi, then in two separate terminals:

```bash
python3 joint_tcp_server.py
python3 suction_tcp_server.py
```

Both servers will keep running and wait for connections from the laptop.

### Step 2 — Build and source the ROS 2 workspace on the laptop

```bash
cd project3-team3_workspace
colcon build
source install/setup.bash
```

### Step 3 — Run the calibration (once per session)

Place an ArUco marker at the known position on the table, then:

```bash
ros2 run perception_pkg aruco_extrinsic_calibrator
```

This writes `~/.ros/myarm_camera_extrinsic.json` and exits.

### Step 4 — Run the task automation

```bash
ros2 run control_pkg task_automation_node
```

This runs the full pick-and-place loop automatically.

---

## Notes

- The Intel RealSense D435i must be connected to the laptop via USB before launching any node.
- The Pi's IP address is configured inside the node parameters (default `192.168.0.103`). Change it if running on different robotic arm.
- The TCP servers run on ports **5017** (joints) and **5018** (suction).
