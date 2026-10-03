import json
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

# qp_ik_solver.py and utils.py use plain (non-package) imports, so add this dir to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qp_ik_solver import solve_ik_multistart
from utils import M, S, fk_poe_world, q_max_rad, q_min_rad


JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]


class QPSuctionControlNode(Node):
    def __init__(self):
        super().__init__("qp_ik_node")

        self.declare_parameter("target_topic", "/perception/suction_point_world")
        self.declare_parameter("joint_target_topic", "/myarm/joint_targets")
        self.declare_parameter("solve_once", True)
        self.declare_parameter("z_offset_m", 0.02)
        self.declare_parameter("pregrasp_target_file", "~/.ros/myarm_last_pregrasp_target.json")

        self.target_sub = self.create_subscription(
            PointStamped,
            self.get_parameter("target_topic").value,
            self.target_callback,
            10,
        )
        self.joint_pub = self.create_publisher(
            JointState,
            self.get_parameter("joint_target_topic").value,
            10,
        )

        self.solved_once = False
        self.get_logger().info(
            "QP suction control node ready. "
            f"Waiting for suction point on {self.get_parameter('target_topic').value}"
        )

    def target_callback(self, msg):
        if bool(self.get_parameter("solve_once").value) and self.solved_once:
            return

        z_offset_m = float(self.get_parameter("z_offset_m").value)
        p_des = np.array([msg.point.x, msg.point.y, msg.point.z + z_offset_m], dtype=float) * 100.0 # make it cm
        self.get_logger().info(f"p_des from suction point: {np.round(p_des, 3).tolist()}")

        result = solve_ik_multistart(
            p_des=p_des,
            sample_count=96,
            try_count=8,
            seed=None,
            joint_limit_margin_deg=15.0,
            joint_limit_score_weight=0.5,
            dt=0.25,
            max_iters=500,
            position_tol=0.5,  # cm -> 5 mm
            z_axis_tol=np.deg2rad(0.1),  # 0.1 degrees
            position_weight=2.0,
            z_axis_weight=1.0,
            damping=1e-3,
        )

        self.get_logger().info(f"success: {result.success}")
        self.get_logger().info(f"message: {result.message}")
        self.get_logger().info(f"iterations: {result.iterations}")
        self.get_logger().info(f"position_error: {result.position_error}")
        self.get_logger().info(f"z_axis_error: {result.z_axis_error}")
        self.get_logger().info(f"q_rad: {result.q}")
        self.get_logger().info(f"q_deg: {np.rad2deg(result.q)}")
        self.get_logger().info(
            f"within_limits: {np.all(result.q >= q_min_rad) and np.all(result.q <= q_max_rad)}"
        )

        T = fk_poe_world(M, S, result.q)
        self.get_logger().info("Validation")
        self.get_logger().info(f"Forward Kinematics of calculated joint angles:\n{T}")
        self.get_logger().info(f"Position: {T[:3, 3]}")
        self.get_logger().info(f"z-axis: {T[:3, 2]}")
        self.get_logger().info(f"Position error: {p_des - T[:3, 3]}")

        if not result.success:
            self.get_logger().error("IK failed; joint angles were not published")
            return

        p_des_m = p_des / 100.0
        self.save_pregrasp_target(msg, p_des_m, z_offset_m, T)

        msg_out = JointState()
        msg_out.header.stamp = self.get_clock().now().to_msg()
        msg_out.name = JOINT_NAMES
        msg_out.position = [float(q) for q in result.q]
        self.joint_pub.publish(msg_out)
        self.solved_once = True
        self.get_logger().info(
            f"Published joint target on {self.get_parameter('joint_target_topic').value}"
        )

    def save_pregrasp_target(self, msg, p_des_m, z_offset_m, solved_transform):
        path = Path(str(self.get_parameter("pregrasp_target_file").value)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "frame_id": msg.header.frame_id or "base",
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nanosec": int(msg.header.stamp.nanosec),
            "suction_point_world_m": [
                float(msg.point.x),
                float(msg.point.y),
                float(msg.point.z),
            ],
            "z_offset_m": float(z_offset_m),
            "pregrasp_target_world_m": [
                float(p_des_m[0]),
                float(p_des_m[1]),
                float(p_des_m[2]),
            ],
            "pregrasp_rotation_world": solved_transform[:3, :3].tolist(),
            "pregrasp_z_axis_world": solved_transform[:3, 2].tolist(),
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        self.get_logger().info(
            f"Saved pre-grasp target to {path}: "
            f"{np.round(np.asarray(data['pregrasp_target_world_m']), 4).tolist()} m"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = QPSuctionControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
