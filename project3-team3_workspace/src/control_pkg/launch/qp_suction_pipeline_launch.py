from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_host = LaunchConfiguration("robot_host")
    robot_port = LaunchConfiguration("robot_port")

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
            executable="top_face_center_node",
            name="top_face_center_node",
            output="screen",
            parameters=[{
                "output_topic": "/perception/suction_point_world",
                "status_topic": "/perception/suction_point_status",
                "extrinsic_file": "~/.ros/myarm_camera_extrinsic.json",
                "sample_duration_s": 2.0,
                "max_std_m": 0.02,
                "publish_once": True,
                "shutdown_after_publish": True,
                "locked_republish_count": 3,
                "show_visualization": True,
                "color_width": 1920,
                "color_height": 1080,
                "depth_width": 1280,
                "depth_height": 720,
                "fps": 30,
                "use_depth_filters": True,
                # Top-face detector knobs (see perception_pkg/claude_top_face_mask.py).
                "cube_edge_m": 0.024,
                "min_depth_m": 0.10,
                "max_depth_m": 1.20,
                "min_component_area_px": 340,
                "hsv_s_min": 90,
                "hsv_v_min": 60,
                "band_below_m": 0.006,
                "band_above_m": 0.008,
                "top_core_tol_m": 0.003,
                "table_margin_px": 120,
                "snap_to_known": True,
            }],
        ),

        Node(
            package="control_pkg",
            executable="qp_ik_node",
            name="qp_ik_node",
            output="screen",
            parameters=[{
                "target_topic": "/perception/suction_point_world",
                "joint_target_topic": "/myarm/joint_targets",
                "solve_once": True,
                "z_offset_m": 0.05,
            }],
        ),

        Node(
            package="control_pkg",
            executable="joint_target_tcp_bridge_node",
            name="joint_target_tcp_bridge_node",
            output="screen",
            parameters=[{
                "topic_name": "/myarm/joint_targets",
                "robot_host": robot_host,
                "robot_port": robot_port,
                "speed": 60,
                "send_repeats": 3,
            }],
        ),
    ])
