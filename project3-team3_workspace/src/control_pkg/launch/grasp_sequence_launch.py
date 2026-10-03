"""Full multi-cube grasp automation.

Run this AFTER the aruco extrinsic calibration (so ~/.ros/myarm_camera_extrinsic
.json exists) and with the Pi's joint TCP server (5017) and suction TCP server
(5018) running. It starts a single orchestrator node that finds every red cube,
indexes them by world Y (index 0 = most negative Y), and grasps them one by one
(pre-grasp QP IK -> visual servoing -> home -> count-based grasp check, with
retries and out-of-reach skipping) until none remain.

    ros2 launch control_pkg grasp_sequence_launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_host = LaunchConfiguration("robot_host")
    robot_port = LaunchConfiguration("robot_port")

    return LaunchDescription([
        DeclareLaunchArgument("robot_host", default_value="192.168.0.103",
                              description="Raspberry Pi IP for the joint + suction TCP servers"),
        DeclareLaunchArgument("robot_port", default_value="5017",
                              description="TCP port of the myArm joint server"),

        Node(
            package="control_pkg",
            executable="task_automation_node",
            name="task_automation_node",
            output="screen",
            parameters=[{
                "robot_host": robot_host,
                "robot_port": robot_port,
                "suction_host": robot_host,
                "suction_port": 5018,
                "enable_suction": True,
                "extrinsic_file": "~/.ros/myarm_camera_extrinsic.json",
                "show_visualization": True,  # live perception viewers

                # pre-grasp QP IK (5 cm above the cube top)
                "z_offset_m": 0.05,
                "position_tol_cm": 0.5,
                "z_axis_tol_deg": 0.5,

                # visual-servoing descend (extra offsets live in vs_move_xy_node)
                "move_z_cm": -3.0,

                # post-grasp: lift straight up, go to a fixed drop pose, release
                "lift_z_cm": 5.0,
                "drop_pose_deg": [52.96265554, 15.92218408, -68.92615251, -66.08121139,
                                  15.58929589, -107.42198129, -154.95087565],
                "valve_pulse_s": 0.3,
                "drop_release_margin_s": 1.0,
                # confirm the drop pose is reached (poll angles) before releasing
                "pose_reach_tol_deg": 3.0,
                "pose_reach_timeout_s": 12.0,

                # gross moves at 60; the VS micro-move (move_speed) stays at 20
                "pregrasp_speed": 60,
                "move_speed": 20,
                "drop_speed": 60,
                "home_speed": 60,

                # retry a cube up to this many times if the count doesn't drop
                "max_grasp_attempts": 3,

                # settle times (tune to your arm speed)
                "pregrasp_settle_s": 6.0,
                "move_settle_s": 3.0,
                "lift_settle_s": 2.0,
                "drop_settle_s": 1.0,
                "home_settle_s": 4.0,

                "home_angles_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }],
        ),
    ])
