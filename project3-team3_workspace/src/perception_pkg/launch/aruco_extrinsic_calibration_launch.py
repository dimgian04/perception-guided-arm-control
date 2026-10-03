from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    realsense_launch = PathJoinSubstitution([
        FindPackageShare("realsense2_camera"),
        "launch",
        "rs_launch.py",
    ])

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch),
            launch_arguments={
                "enable_color": "true",
                "enable_depth": "true",
                "align_depth.enable": "true",
                "pointcloud.enable": "false",
                "rgb_camera.color_profile": "1280x720x30",
                "depth_module.depth_profile": "848x480x30",
            }.items(),
        ),

        Node(
            package="perception_pkg",
            executable="aruco_extrinsic_calibrator",
            name="aruco_extrinsic_calibrator",
            output="screen",
            parameters=[{
                "marker_length_m": 0.10,
                # -1 means "use the first visible marker".
                "marker_id": -1,
                # Placeholder marker pose in myarm_base_frame. Update these.
                "marker_world_xyz_m": [0.0, -0.156, -0.015],
                "marker_world_rpy_rad": [0.000, 0.000, 0.000],
                "sample_duration_s": 6.0,
                "min_samples": 20,
                "output_file": "~/.ros/myarm_camera_extrinsic.json",
                "show_visualization": True,
                "pose_log_period_s": 0.5,
            }],
        ),
    ])
