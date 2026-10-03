"""Visual-servoing pipeline.

Run this AFTER the QP pre-grasp (qp_suction_pipeline_launch.py) has placed the
suction cup ~5 cm above the cube. It brings up:

  * perception_pkg/claude_visual_servoing_node - watches the cube top face and
    the blue tape on the cup, averages the world XY misalignment over a few
    seconds, and publishes it once (cm) on /visual_servoing/move_cm.

  * control_pkg/vs_move_xy_node - on that message, reads the robot's current
    angles, solves IK for the XY correction plus a fixed -3 cm descend while
    HOLDING the current tool orientation, and sends the joint target over TCP.

    ros2 launch control_pkg visual_servoing_launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_host = LaunchConfiguration("robot_host")
    robot_port = LaunchConfiguration("robot_port")
    move_topic = "/visual_servoing/move_cm"

    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_host",
            default_value="192.168.0.103",
            description="Raspberry Pi IP address for the myArm TCP server",
        ),
        DeclareLaunchArgument(
            "robot_port",
            default_value="5017",
            description="TCP port for the myArm TCP server",
        ),

        Node(
            package="perception_pkg",
            executable="visual_servoing_perception_node",
            name="visual_servoing_perception_node",
            output="screen",
            parameters=[{
                "output_topic": move_topic,
                "status_topic": "/visual_servoing/status",
                "extrinsic_file": "~/.ros/myarm_camera_extrinsic.json",
                "warmup_s": 2.0,
                "gather_duration_s": 3.0,
                "min_samples": 15,
                "max_std_cm": 1.0,
                "publish_once": True,
                "locked_republish_count": 5,
                "shutdown_after_publish": False,
                "show_visualization": True,
                "ee_offset_x_m": -0.004,
            }],
        ),

        Node(
            package="control_pkg",
            executable="visual_servoing_motion_node",
            name="visual_servoing_motion_node",
            output="screen",
            parameters=[{
                "move_topic": move_topic,
                "robot_host": robot_host,
                "robot_port": robot_port,
                "speed": 20,
                "move_z_cm": -3.0,          # fixed downward descend for VS.
                "move_settle_s": 3.0,
                "hold_orientation": True,   # keep current tool z-axis.
                "handle_once": True,
                "dry_run": False,
                "max_joint_delta_deg": 60.0,
                # After the grasp: lift straight up, go to drop pose, release.
                "lift_z_cm": 5.0,
                "drop_pose_deg": [52.96265554, 15.92218408, -68.92615251, -66.08121139,
                                  15.58929589, -107.42198129, -154.95087565],
                "drop_speed": 60,
                "lift_settle_s": 2.0,
                "drop_settle_s": 1.0,
                "drop_release_margin_s": 1.0,
                # confirm the drop pose is reached (poll angles) before releasing
                "pose_reach_tol_deg": 3.0,
                "pose_reach_timeout_s": 12.0,
                "home_after_release": True,
                "home_speed": 60,
                "home_settle_s": 3.0,
                "home_angles_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # Suction: pump ON at node start; at the drop pose pump OFF + valve pulse.
                "enable_suction": True,
                "suction_host": robot_host,
                "suction_port": 5018,
                "valve_pulse_s": 0.3,
            }],
        ),
    ])
