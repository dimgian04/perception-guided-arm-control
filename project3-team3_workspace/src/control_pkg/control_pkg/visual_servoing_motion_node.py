import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))

import move_xyz as mxy
from nr_ik import Q_MAX, Q_MIN


# Extra offsets applied on top of the visual-servoing XY correction every move.
# EXTRA_Z_CM goes on top of the fixed -3 cm descend (move_z_cm parameter).
EXTRA_X_CM = 0.0
EXTRA_Y_CM = -1.0
EXTRA_Z_CM = -2.0


class VisualServoMoveNode(Node):
    def __init__(self):
        super().__init__("visual_servoing_motion_node")

        self.declare_parameter("move_topic", "/visual_servoing/move_cm")
        self.declare_parameter("robot_host", "192.168.0.103")
        self.declare_parameter("robot_port", 5017)
        self.declare_parameter("timeout", 3.0)
        self.declare_parameter("speed", 20)

        self.declare_parameter("move_z_cm", -3.0)
        self.declare_parameter("move_settle_s", 3.0)
        self.declare_parameter("hold_orientation", True)
        self.declare_parameter("handle_once", True)
        self.declare_parameter("dry_run", False)

        self.declare_parameter("lift_z_cm", 5.0)
        self.declare_parameter("drop_pose_deg", [
            52.96265554, 15.92218408, -68.92615251, -66.08121139,
            15.58929589, -107.42198129, -154.95087565,
        ])
        self.declare_parameter("drop_speed", 60)
        self.declare_parameter("lift_settle_s", 2.0)
        self.declare_parameter("drop_settle_s", 1.0)
        self.declare_parameter("drop_release_margin_s", 1.0)
        self.declare_parameter("pose_reach_tol_deg", 3.0)
        self.declare_parameter("pose_reach_timeout_s", 12.0)
        self.declare_parameter("home_after_release", True)
        self.declare_parameter("home_speed", 60)
        self.declare_parameter("home_settle_s", 3.0)
        self.declare_parameter("home_angles_deg", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.declare_parameter("enable_suction", True)
        self.declare_parameter("suction_host", "192.168.0.103")
        self.declare_parameter("suction_port", 5018)
        self.declare_parameter("valve_pulse_s", 0.3)

        self.declare_parameter("max_iters", 1000)
        self.declare_parameter("error_tol", 1e-5)
        self.declare_parameter("position_tol_m", 0.0001)
        self.declare_parameter("z_axis_tol_deg", 0.1)
        self.declare_parameter("damping", 0.05)
        self.declare_parameter("max_step_deg", 3.0)
        self.declare_parameter("joint_center_gain", 0.0)
        self.declare_parameter("max_joint_delta_deg", 45.0)
        self.declare_parameter("allow_large_joint_delta", False)

        self.handled = False
        self.busy = False

        self.sub = self.create_subscription(
            Vector3,
            self.get_parameter("move_topic").value,
            self.move_callback,
            10,
        )
        self.get_logger().info(
            "Visual-servoing move node ready. Waiting for XY correction (cm) on "
            f"{self.get_parameter('move_topic').value}; z is fixed at "
            f"{float(self.get_parameter('move_z_cm').value):.1f} cm, "
            f"hold_orientation={bool(self.get_parameter('hold_orientation').value)}"
        )

        self.set_pump(True)

    def set_pump(self, on):
        if not bool(self.get_parameter("enable_suction").value):
            return
        host = str(self.get_parameter("suction_host").value)
        port = int(self.get_parameter("suction_port").value)
        timeout = float(self.get_parameter("timeout").value)
        try:
            response = mxy.tcp_request(host, port, {"type": "set", "pump": bool(on)}, timeout)
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Failed to set pump over suction TCP ({host}:{port}): {exc}")
            return
        self.get_logger().info(
            f"Suction pump {'ON (pin 20 LOW)' if on else 'OFF (pin 20 HIGH)'} | "
            f"server response={response}"
        )

    def move_callback(self, msg):
        if self.busy:
            return
        if bool(self.get_parameter("handle_once").value) and self.handled:
            return

        self.busy = True
        try:
            self.handle_move(float(msg.x), float(msg.y))
        finally:
            self.busy = False

    def handle_move(self, move_x_cm, move_y_cm):
        host = str(self.get_parameter("robot_host").value)
        port = int(self.get_parameter("robot_port").value)
        timeout = float(self.get_parameter("timeout").value)
        hold_orientation = bool(self.get_parameter("hold_orientation").value)

        move_x_cm = move_x_cm + EXTRA_X_CM
        move_y_cm = move_y_cm + EXTRA_Y_CM
        move_z_cm = float(self.get_parameter("move_z_cm").value) + EXTRA_Z_CM

        self.get_logger().info(
            f"Final move (incl. hardcoded EXTRA_X_CM={EXTRA_X_CM:+.2f}, "
            f"EXTRA_Y_CM={EXTRA_Y_CM:+.2f}, EXTRA_Z_CM={EXTRA_Z_CM:+.2f}): "
            f"x={move_x_cm:+.2f} cm y={move_y_cm:+.2f} cm z={move_z_cm:+.2f} cm"
        )

        try:
            q0_deg = mxy.request_current_angles_deg(host, port, timeout)
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Failed to read current angles over TCP: {exc}")
            return
        q0_rad = np.clip(np.deg2rad(q0_deg), Q_MIN, Q_MAX)

        pregrasp_rotation = None
        if not hold_orientation:
            try:
                pregrasp_rotation, _ = mxy.load_pregrasp_rotation(
                    "~/.ros/myarm_last_pregrasp_target.json"
                )
            except (OSError, RuntimeError, ValueError) as exc:
                self.get_logger().error(f"Failed to load pregrasp rotation: {exc}")
                return

        q_sol, success, iterations, error_norm, current_t, target_t, solved_t = mxy.solve_xyz(
            q0_rad=q0_rad,
            move_x_cm=move_x_cm,
            move_y_cm=move_y_cm,
            move_z_cm=move_z_cm,
            max_iters=int(self.get_parameter("max_iters").value),
            error_tol=float(self.get_parameter("error_tol").value),
            position_tol_m=float(self.get_parameter("position_tol_m").value),
            z_axis_tol_deg=float(self.get_parameter("z_axis_tol_deg").value),
            damping=float(self.get_parameter("damping").value),
            max_step_deg=float(self.get_parameter("max_step_deg").value),
            joint_center_gain=float(self.get_parameter("joint_center_gain").value),
            pregrasp_rotation=pregrasp_rotation,
            hold_orientation=hold_orientation,
        )

        q0_deg_arr = np.asarray(q0_deg, dtype=float)
        q_sol_deg = np.rad2deg(q_sol)
        delta_deg = q_sol_deg - q0_deg_arr
        max_delta = float(np.max(np.abs(delta_deg)))

        self.get_logger().info(
            f"IK success={success} iters={iterations} err={error_norm:.3g} | "
            f"target_pos_m={np.round(target_t[:3, 3], 4).tolist()} "
            f"solved_pos_m={np.round(solved_t[:3, 3], 4).tolist()} | "
            f"max_joint_delta_deg={max_delta:.2f}"
        )

        if not success:
            self.get_logger().error("IK did not converge; no joint target sent.")
            return

        max_allowed = float(self.get_parameter("max_joint_delta_deg").value)
        if max_delta > max_allowed and not bool(self.get_parameter("allow_large_joint_delta").value):
            self.get_logger().error(
                f"Joint delta {max_delta:.2f} deg exceeds limit {max_allowed:.2f} deg; "
                "no joint target sent (set allow_large_joint_delta to override)."
            )
            return

        if bool(self.get_parameter("dry_run").value):
            self.handled = True
            self.get_logger().info(
                f"DRY RUN: would send q_deg={np.round(q_sol_deg, 3).tolist()}"
            )
            return

        try:
            response = mxy.send_joint_target_deg(
                host, port, q_sol_deg, int(self.get_parameter("speed").value), timeout
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Failed to send joint target over TCP: {exc}")
            return

        self.handled = True
        self.get_logger().info(
            f"Sent descend move q_deg={np.round(q_sol_deg, 3).tolist()} | TCP response={response}"
        )
        time.sleep(float(self.get_parameter("move_settle_s").value))
        self.post_grasp_sequence()

    def post_grasp_sequence(self):
        self.post_grasp_lift()
        self.go_to_drop_pose()
        self.release_at_drop()
        if bool(self.get_parameter("home_after_release").value):
            self.go_home()

    def robot_tcp(self):
        return (str(self.get_parameter("robot_host").value),
                int(self.get_parameter("robot_port").value),
                float(self.get_parameter("timeout").value))

    def post_grasp_lift(self):
        lift_cm = float(self.get_parameter("lift_z_cm").value)
        host, port, timeout = self.robot_tcp()
        try:
            q0_deg = mxy.request_current_angles_deg(host, port, timeout)
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Lift: read angles failed: {exc}")
            return
        q0 = np.clip(np.deg2rad(q0_deg), Q_MIN, Q_MAX)
        self.get_logger().info(f"Post-grasp lift: +{lift_cm:.1f} cm straight up on Z")
        q_sol, success, *_rest = mxy.solve_xyz(
            q0_rad=q0, move_x_cm=0.0, move_y_cm=0.0, move_z_cm=lift_cm,
            max_iters=int(self.get_parameter("max_iters").value),
            error_tol=float(self.get_parameter("error_tol").value),
            position_tol_m=float(self.get_parameter("position_tol_m").value),
            z_axis_tol_deg=float(self.get_parameter("z_axis_tol_deg").value),
            damping=float(self.get_parameter("damping").value),
            max_step_deg=float(self.get_parameter("max_step_deg").value),
            joint_center_gain=float(self.get_parameter("joint_center_gain").value),
            pregrasp_rotation=None, hold_orientation=True,
        )
        if not success:
            self.get_logger().error("Lift IK did not converge; skipping lift.")
            return
        try:
            mxy.send_joint_target_deg(host, port, np.rad2deg(q_sol).tolist(),
                                      int(self.get_parameter("speed").value), timeout)
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Lift send failed: {exc}")
            return
        time.sleep(float(self.get_parameter("lift_settle_s").value))

    def go_to_drop_pose(self):
        host, port, timeout = self.robot_tcp()
        drop = [float(a) for a in self.get_parameter("drop_pose_deg").value]
        self.get_logger().info(f"Moving to drop pose q_deg={np.round(drop, 2).tolist()}")
        try:
            mxy.send_joint_target_deg(host, port, drop,
                                      int(self.get_parameter("drop_speed").value), timeout)
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Drop pose send failed: {exc}")
        self.wait_until_at_pose(drop)
        time.sleep(float(self.get_parameter("drop_settle_s").value))

    def wait_until_at_pose(self, target_deg):
        host, port, timeout = self.robot_tcp()
        tol = float(self.get_parameter("pose_reach_tol_deg").value)
        deadline = time.time() + float(self.get_parameter("pose_reach_timeout_s").value)
        last_err = None
        while time.time() < deadline:
            try:
                cur = mxy.request_current_angles_deg(host, port, timeout)
            except (OSError, RuntimeError, ValueError):
                time.sleep(0.3)
                continue
            last_err = max(abs(float(c) - float(t)) for c, t in zip(cur, target_deg))
            if last_err <= tol:
                self.get_logger().info(f"Reached drop pose (max joint err {last_err:.2f} deg)")
                return True
            time.sleep(0.3)
        self.get_logger().warn(f"Drop-pose reach timeout (max joint err {last_err} deg); proceeding")
        return False

    def release_at_drop(self):
        if not bool(self.get_parameter("enable_suction").value):
            return
        pulse_s = float(self.get_parameter("valve_pulse_s").value)
        host = str(self.get_parameter("suction_host").value)
        port = int(self.get_parameter("suction_port").value)
        try:
            resp = mxy.tcp_request(host, port, {"type": "release", "pulse_s": pulse_s},
                                   float(self.get_parameter("timeout").value))
            self.get_logger().info(f"Release at drop: pump OFF + valve pulse {pulse_s:.2f}s -> {resp}")
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Release failed: {exc}")
        time.sleep(pulse_s + float(self.get_parameter("drop_release_margin_s").value))

    def go_home(self):
        host, port, timeout = self.robot_tcp()
        home = [float(a) for a in self.get_parameter("home_angles_deg").value]
        self.get_logger().info(f"Returning home q_deg={home}")
        try:
            mxy.send_joint_target_deg(host, port, home,
                                      int(self.get_parameter("home_speed").value), timeout)
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Home send failed: {exc}")
        time.sleep(float(self.get_parameter("home_settle_s").value))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VisualServoMoveNode()
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
