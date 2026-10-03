import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node

from perception_pkg import top_face_center_mask as ctf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import move_xyz as mxy
from nr_ik import Q_MAX, Q_MIN
from qp_ik_solver import solve_ik_multistart
from visual_servoing_motion_node import EXTRA_X_CM, EXTRA_Y_CM, EXTRA_Z_CM


WIN_MAIN = "grasp sequence"
WIN_MASK = "grasp sequence masks"


class GraspSequenceNode(Node):
    def __init__(self):
        super().__init__("task_automation_node")

        self.declare_parameter("robot_host", "192.168.0.103")
        self.declare_parameter("robot_port", 5017)
        self.declare_parameter("suction_host", "192.168.0.103")
        self.declare_parameter("suction_port", 5018)
        self.declare_parameter("timeout", 3.0)
        self.declare_parameter("enable_suction", True)

        self.declare_parameter("extrinsic_file", ctf.EXTRINSIC_FILE)

        self.declare_parameter("pregrasp_speed", 60)
        self.declare_parameter("move_speed", 20)
        self.declare_parameter("home_speed", 60)
        self.declare_parameter("drop_speed", 60)
        self.declare_parameter("pregrasp_settle_s", 6.0)
        self.declare_parameter("move_settle_s", 3.0)
        self.declare_parameter("home_settle_s", 4.0)
        self.declare_parameter("lift_settle_s", 2.0)
        self.declare_parameter("drop_settle_s", 4.0)
        self.declare_parameter("deposit_settle_s", 1.5)
        self.declare_parameter("home_angles_deg", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.declare_parameter("lift_z_cm", 5.0)
        self.declare_parameter("drop_pose_deg", [
            52.96265554, 15.92218408, -68.92615251, -66.08121139,
            15.58929589, -107.42198129, -154.95087565,
        ])
        self.declare_parameter("valve_pulse_s", 0.3)
        self.declare_parameter("drop_release_margin_s", 1.0)
        self.declare_parameter("pose_reach_tol_deg", 3.0)
        self.declare_parameter("pose_reach_timeout_s", 12.0)

        self.declare_parameter("z_offset_m", 0.05)
        self.declare_parameter("position_tol_cm", 0.5)
        self.declare_parameter("z_axis_tol_deg", 0.5)

        self.declare_parameter("move_z_cm", -3.0)
        self.declare_parameter("ik_max_iters", 1000)
        self.declare_parameter("ik_damping", 0.05)
        self.declare_parameter("ik_position_tol_m", 0.0001)
        self.declare_parameter("ik_z_axis_tol_deg", 0.1)
        self.declare_parameter("ik_max_step_deg", 3.0)
        self.declare_parameter("max_joint_delta_deg", 60.0)

        self.declare_parameter("warmup_s", 2.0)
        self.declare_parameter("gather_duration_s", 3.0)
        self.declare_parameter("min_samples", 15)
        self.declare_parameter("max_std_cm", 1.5)
        self.declare_parameter("gather_timeout_s", 15.0)
        self.declare_parameter("ee_offset_x_m", -0.004)

        self.declare_parameter("red_s_min", ctf.RED_HSV_RANGES[0][0][1])
        self.declare_parameter("red_v_min", ctf.RED_HSV_RANGES[0][0][2])
        self.declare_parameter("band_below_m", ctf.TOP_BAND_BELOW_M)
        self.declare_parameter("band_above_m", ctf.TOP_BAND_ABOVE_M)
        self.declare_parameter("detect_frames", 12)
        self.declare_parameter("detect_support_frac", 0.4)
        self.declare_parameter("cube_match_radius_m", 0.02)

        self.declare_parameter("blue_h_lo", 94)
        self.declare_parameter("blue_h_hi", 105)
        self.declare_parameter("blue_s_lo", 55)
        self.declare_parameter("blue_s_hi", 140)
        self.declare_parameter("blue_v_lo", 95)
        self.declare_parameter("blue_v_hi", 155)
        self.declare_parameter("blue_min_area_px", 340)

        self.declare_parameter("max_grasp_attempts", 3)
        self.declare_parameter("skip_radius_m", 0.03)

        self.declare_parameter("show_visualization", True)
        self.viz = bool(self.get_parameter("show_visualization").value)

        self.t_wc = self.load_t_wc()
        if self.t_wc is None:
            raise RuntimeError("No camera extrinsic; run aruco calibration first.")
        self.rotation = self.t_wc[:3, :3]
        self.translation = self.t_wc[:3, 3]
        self.up_cam = ctf.world_up_in_camera(self.t_wc)

        if self.viz:
            cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
            cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)

        self.pipe, self.align, self.depth_scale = ctf.start_realsense()
        self.depth_filters = ctf.create_depth_filters()
        for _ in range(15):
            self.grab_frame()

    def grab_frame(self):
        frames = self.align.process(self.pipe.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None
        for depth_filter in self.depth_filters:
            depth_frame = depth_filter.process(depth_frame)
        intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        depth_m = depth.astype(np.float32) * float(self.depth_scale)
        return color, depth_m, intrinsics

    def detect_params(self):
        return {
            "band_below_m": float(self.get_parameter("band_below_m").value),
            "band_above_m": float(self.get_parameter("band_above_m").value),
        }

    def _panel(self, display, lines):
        x0, y0 = 12, 12
        h = 26 * len(lines) + 14
        overlay = display.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + 620, y0 + h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0.0, dst=display)
        for i, line in enumerate(lines):
            cv2.putText(display, line, (x0 + 12, y0 + 24 + 24 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        2 if i == 0 else 1, cv2.LINE_AA)

    def _fitted_quad(self, corners_cam, intrinsics):
        if corners_cam is None:
            return None
        pts = []
        for corner in corners_cam:
            px = ctf.project_point(corner, intrinsics)
            if px is None:
                return None
            pts.append(px)
        return np.array(pts, dtype=np.int32)

    def _draw_square(self, display, mask_vis, corners_cam, intrinsics):
        quad = self._fitted_quad(corners_cam, intrinsics)
        if quad is None:
            return
        fitted = np.zeros(display.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(fitted, quad, 255)
        ctf.overlay_mask(display, fitted, (0, 255, 0), 0.40)
        ctf.draw_contours(display, fitted, (0, 255, 0), 2)
        if mask_vis is not None:
            mask_vis[fitted > 0] = (0, 255, 0)

    def show_detection(self, color, cubes, intrinsics, phase):
        if not self.viz:
            return
        display = color.copy()
        mask_vis = np.zeros_like(color)
        for cube in cubes:
            self._draw_square(display, mask_vis, cube.get("corners_cam"), intrinsics)
            px = ctf.project_point(cube["center_camera"], intrinsics)
            if px is not None:
                p = (int(round(px[0])), int(round(px[1])))
                cv2.drawMarker(display, p, (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
                cv2.putText(display, f"#{cube['index']}", (p[0] + 10, p[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        self._panel(display, [phase, f"cubes detected this frame: {len(cubes)}"])
        cv2.imshow(WIN_MAIN, display)
        cv2.imshow(WIN_MASK, mask_vis)
        cv2.waitKey(1)

    def show_vs(self, color, intrinsics, top_corners_cam, top_world, blue_world, blue_mask,
                ee_world, move, samples):
        if not self.viz:
            return
        display = color.copy()
        mask_vis = np.zeros_like(color)
        self._draw_square(display, mask_vis, top_corners_cam, intrinsics)
        ctf.overlay_mask(display, blue_mask, (255, 60, 0), 0.30)
        ctf.draw_contours(display, blue_mask, (255, 120, 0), 2)
        if blue_mask is not None:
            mask_vis[blue_mask > 0] = (255, 120, 0)
        for world, color_bgr, label in (
            (top_world, (0, 255, 0), "CUBE TOP"),
            (blue_world, (255, 150, 0), "BLUE"),
            (ee_world, (0, 255, 255), "EE CENTER"),
        ):
            if world is None:
                continue
            px = ctf.project_point(self.rotation.T @ (np.asarray(world) - self.translation), intrinsics)
            if px is None:
                continue
            p = (int(round(px[0])), int(round(px[1])))
            cv2.drawMarker(display, p, color_bgr, cv2.MARKER_CROSS, 22, 2)
            cv2.circle(display, p, 4, color_bgr, -1)
            cv2.putText(display, label, (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)
        move_txt = "move: waiting" if move is None else f"move x={move[0]:+.2f} y={move[1]:+.2f} cm"
        self._panel(display, ["VISUAL SERVOING", move_txt, f"samples={samples}"])
        cv2.imshow(WIN_MAIN, display)
        cv2.imshow(WIN_MASK, mask_vis)
        cv2.waitKey(1)

    def sleep_with_view(self, seconds, label):
        if not self.viz:
            time.sleep(max(0.0, seconds))
            return
        end = time.time() + max(0.0, seconds)
        while time.time() < end:
            grab = self.grab_frame()
            if grab is None:
                cv2.waitKey(1)
                continue
            display = grab[0].copy()
            self._panel(display, [label, "(robot moving / settling)"])
            cv2.imshow(WIN_MAIN, display)
            cv2.waitKey(1)

    def detect_cubes_stable(self):
        frames = max(1, int(self.get_parameter("detect_frames").value))
        s_min = int(self.get_parameter("red_s_min").value)
        v_min = int(self.get_parameter("red_v_min").value)
        radius = float(self.get_parameter("cube_match_radius_m").value)
        params = self.detect_params()

        # Cluster detections across frames to reject single-frame noise.
        clusters = []
        for _ in range(frames):
            grab = self.grab_frame()
            if grab is None:
                continue
            color, depth_m, intr = grab
            cubes = ctf.detect_indexed_cubes(color, depth_m, intr, self.t_wc, self.up_cam,
                                             params, s_min, v_min)
            self.show_detection(color, cubes, intr, "DETECTING CUBES")
            for cube in cubes:
                p = cube["center_world"]
                placed = False
                for cl in clusters:
                    if float(np.linalg.norm(p[:2] - cl["mean"][:2])) < radius:
                        cl["pts"].append(p)
                        cl["mean"] = np.mean(cl["pts"], axis=0)
                        placed = True
                        break
                if not placed:
                    clusters.append({"pts": [p], "mean": p.copy()})

        support = max(1, int(frames * float(self.get_parameter("detect_support_frac").value)))
        centres = [cl["mean"] for cl in clusters if len(cl["pts"]) >= support]
        centres.sort(key=ctf.cube_sort_key)
        return centres

    def count_cubes(self, exclude):
        centres = self.detect_cubes_stable()
        centres = [c for c in centres if not self.near_any(c, exclude)]
        return len(centres), centres

    def near_any(self, centre, others):
        radius = float(self.get_parameter("skip_radius_m").value)
        return any(float(np.linalg.norm(centre[:2] - o[:2])) < radius for o in others)

    def robot_host_port(self):
        return str(self.get_parameter("robot_host").value), int(self.get_parameter("robot_port").value)

    def get_angles_deg(self):
        host, port = self.robot_host_port()
        return mxy.request_current_angles_deg(host, port, float(self.get_parameter("timeout").value))

    def send_angles_deg(self, q_deg, speed):
        host, port = self.robot_host_port()
        return mxy.send_joint_target_deg(host, port, [float(a) for a in q_deg], int(speed),
                                         float(self.get_parameter("timeout").value))

    def go_home(self):
        home = [float(a) for a in self.get_parameter("home_angles_deg").value]
        self.get_logger().info(f"Returning home q_deg={home}")
        try:
            self.send_angles_deg(home, int(self.get_parameter("home_speed").value))
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Home command failed: {exc}")
        self.wait_until_at_pose(home, "home")
        self.sleep_with_view(float(self.get_parameter("home_settle_s").value), "HOMING")

    def set_pump(self, on):
        if not bool(self.get_parameter("enable_suction").value):
            return
        host = str(self.get_parameter("suction_host").value)
        port = int(self.get_parameter("suction_port").value)
        try:
            resp = mxy.tcp_request(host, port, {"type": "set", "pump": bool(on)},
                                   float(self.get_parameter("timeout").value))
            self.get_logger().info(f"Pump {'ON (pin 20 LOW)' if on else 'OFF (pin 20 HIGH)'}: {resp}")
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Suction command failed: {exc}")

    def pregrasp(self, centre_world):
        z_off = float(self.get_parameter("z_offset_m").value)
        p_des = np.array([centre_world[0], centre_world[1], centre_world[2] + z_off]) * 100.0
        result = solve_ik_multistart(
            p_des=p_des, sample_count=96, try_count=8, seed=None,
            joint_limit_margin_deg=15.0, joint_limit_score_weight=0.5, dt=0.25, max_iters=500,
            position_tol=float(self.get_parameter("position_tol_cm").value),
            z_axis_tol=np.deg2rad(float(self.get_parameter("z_axis_tol_deg").value)),
            position_weight=2.0, z_axis_weight=1.0, damping=1e-3,
        )
        if not result.success:
            self.get_logger().warn(f"QP IK failed: {result.message}")
            return False
        q_deg = np.rad2deg(result.q).tolist()
        self.get_logger().info(f"Pre-grasp IK ok; sending q_deg={np.round(q_deg, 2).tolist()}")
        try:
            self.send_angles_deg(q_deg, int(self.get_parameter("pregrasp_speed").value))
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Pre-grasp send failed: {exc}")
            return False
        self.sleep_with_view(float(self.get_parameter("pregrasp_settle_s").value), "PRE-GRASP move")
        return True

    def pregrasp_with_retry(self, centre_world):
        if self.pregrasp(centre_world):
            return True
        self.get_logger().warn("Pre-grasp IK failed; retrying once.")
        return self.pregrasp(centre_world)

    def blue_world(self, hsv, depth_m, intrinsics, ref_px):
        lo = np.array((int(self.get_parameter("blue_h_lo").value),
                       int(self.get_parameter("blue_s_lo").value),
                       int(self.get_parameter("blue_v_lo").value)), np.uint8)
        hi = np.array((int(self.get_parameter("blue_h_hi").value),
                       int(self.get_parameter("blue_s_hi").value),
                       int(self.get_parameter("blue_v_hi").value)), np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = int(self.get_parameter("blue_min_area_px").value)
        cands = [i for i in range(1, count) if int(stats[i, cv2.CC_STAT_AREA]) >= min_area]
        if not cands:
            return None, None
        if ref_px is None:
            best = max(cands, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
        else:
            best = min(cands, key=lambda i: (centroids[i][0] - ref_px[0]) ** 2 + (centroids[i][1] - ref_px[1]) ** 2)
        blue_mask = (labels == best).astype(np.uint8) * 255

        moments = cv2.moments(blue_mask, binaryImage=True)
        if moments["m00"] <= 0:
            return None, None
        cu = moments["m10"] / moments["m00"]
        cv_ = moments["m01"] / moments["m00"]
        cu_i, cv_i = int(round(cu)), int(round(cv_))
        h, w = depth_m.shape
        y0, y1 = max(0, cv_i - 15), min(h, cv_i + 16)
        x0, x1 = max(0, cu_i - 15), min(w, cu_i + 16)
        win = depth_m[y0:y1, x0:x1]
        mwin = blue_mask[y0:y1, x0:x1] > 0
        valid = mwin & np.isfinite(win) & (win >= ctf.MIN_DEPTH_M) & (win <= ctf.MAX_DEPTH_M)
        samples = win[valid]
        if samples.size < 5:
            return None, blue_mask
        z = float(np.median(samples))
        x = (cu - float(intrinsics.ppx)) * z / float(intrinsics.fx)
        y = (cv_ - float(intrinsics.ppy)) * z / float(intrinsics.fy)
        return self.rotation @ np.array([x, y, z]) + self.translation, blue_mask

    def nearest_cube_center(self, cubes, target):
        radius = float(self.get_parameter("cube_match_radius_m").value) * 3.0
        best = None
        best_d = radius
        for cube in cubes:
            d = float(np.linalg.norm(cube["center_world"][:2] - target[:2]))
            if d < best_d:
                best_d = d
                best = cube
        return best

    def gather_move(self, target_centre):
        warmup_s = float(self.get_parameter("warmup_s").value)
        gather_s = float(self.get_parameter("gather_duration_s").value)
        min_samples = int(self.get_parameter("min_samples").value)
        max_std = float(self.get_parameter("max_std_cm").value)
        timeout_s = float(self.get_parameter("gather_timeout_s").value)
        ee_off = np.array([float(self.get_parameter("ee_offset_x_m").value), 0.0, 0.0])
        params = self.detect_params()
        s_min = int(self.get_parameter("red_s_min").value)
        v_min = int(self.get_parameter("red_v_min").value)

        start = time.time()
        detect_start = None
        sampling_start = None
        samples = []
        while time.time() - start < timeout_s:
            grab = self.grab_frame()
            if grab is None:
                continue
            color, depth_m, intr = grab
            cubes = ctf.detect_indexed_cubes(color, depth_m, intr, self.t_wc, self.up_cam,
                                             params, s_min, v_min)
            cube = self.nearest_cube_center(cubes, target_centre)
            if cube is None:
                self.show_vs(color, intr, None, None, None, None, None, None, len(samples))
                continue
            top_world = cube["center_world"]
            top_corners = cube["corners_cam"]
            top_cam = self.rotation.T @ (top_world - self.translation)
            ref_px = ctf.project_point(top_cam, intr)
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            blue, blue_mask = self.blue_world(hsv, depth_m, intr, ref_px)
            if blue is None:
                self.show_vs(color, intr, top_corners, top_world, None, blue_mask, None, None, len(samples))
                continue
            ee = blue + ee_off
            move = (float((top_world[0] - ee[0]) * 100.0), float((top_world[1] - ee[1]) * 100.0))
            self.show_vs(color, intr, top_corners, top_world, blue, blue_mask, ee, move, len(samples))

            now = time.time()
            if detect_start is None:
                detect_start = now
                continue
            if now - detect_start < warmup_s:
                continue
            if sampling_start is None:
                sampling_start = now
                samples = []
            samples.append(np.array(move))
            if now - sampling_start >= gather_s and len(samples) >= min_samples:
                arr = np.array(samples)
                mean = np.mean(arr, axis=0)
                std = np.std(arr, axis=0)
                if float(np.max(std)) > max_std:
                    self.get_logger().warn(f"VS jittery (std={np.round(std,2).tolist()}); restarting window")
                    sampling_start = None
                    samples = []
                    continue
                self.get_logger().info(
                    f"VS correction x={mean[0]:+.2f} y={mean[1]:+.2f} cm (std={np.round(std,2).tolist()}, n={len(samples)})"
                )
                return float(mean[0]), float(mean[1])
        self.get_logger().warn("VS gather timed out; no stable correction.")
        return None

    def execute_move(self, dx_cm, dy_cm):
        try:
            q0_deg = self.get_angles_deg()
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Read angles failed: {exc}")
            return False
        q0 = np.clip(np.deg2rad(q0_deg), Q_MIN, Q_MAX)
        move_x = dx_cm + EXTRA_X_CM
        move_y = dy_cm + EXTRA_Y_CM
        move_z = float(self.get_parameter("move_z_cm").value) + EXTRA_Z_CM
        self.get_logger().info(
            f"VS move (incl. extras): x={move_x:+.2f} y={move_y:+.2f} z={move_z:+.2f} cm"
        )
        q_sol, success, iters, err, _ct, _tt, _st = mxy.solve_xyz(
            q0_rad=q0, move_x_cm=move_x, move_y_cm=move_y, move_z_cm=move_z,
            max_iters=int(self.get_parameter("ik_max_iters").value), error_tol=1e-5,
            position_tol_m=float(self.get_parameter("ik_position_tol_m").value),
            z_axis_tol_deg=float(self.get_parameter("ik_z_axis_tol_deg").value),
            damping=float(self.get_parameter("ik_damping").value),
            max_step_deg=float(self.get_parameter("ik_max_step_deg").value),
            joint_center_gain=0.0, pregrasp_rotation=None, hold_orientation=True,
        )
        if not success:
            self.get_logger().error("VS move IK did not converge; skipping move.")
            return False
        q_sol_deg = np.rad2deg(q_sol)
        max_delta = float(np.max(np.abs(q_sol_deg - np.asarray(q0_deg))))
        if max_delta > float(self.get_parameter("max_joint_delta_deg").value):
            self.get_logger().error(f"VS move joint delta {max_delta:.1f} deg too large; skipping move.")
            return False
        try:
            self.send_angles_deg(q_sol_deg.tolist(), int(self.get_parameter("move_speed").value))
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"VS move send failed: {exc}")
            return False
        self.sleep_with_view(float(self.get_parameter("move_settle_s").value), "VS descend")
        return True

    def post_grasp_lift(self):
        lift_cm = float(self.get_parameter("lift_z_cm").value)
        try:
            q0_deg = self.get_angles_deg()
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Post-grasp lift: read angles failed: {exc}")
            return
        q0 = np.clip(np.deg2rad(q0_deg), Q_MIN, Q_MAX)
        self.get_logger().info(f"Post-grasp lift: +{lift_cm:.1f} cm straight up on Z")
        q_sol, success, _it, _err, _ct, _tt, _st = mxy.solve_xyz(
            q0_rad=q0, move_x_cm=0.0, move_y_cm=0.0, move_z_cm=lift_cm,
            max_iters=int(self.get_parameter("ik_max_iters").value), error_tol=1e-5,
            position_tol_m=float(self.get_parameter("ik_position_tol_m").value),
            z_axis_tol_deg=float(self.get_parameter("ik_z_axis_tol_deg").value),
            damping=float(self.get_parameter("ik_damping").value),
            max_step_deg=float(self.get_parameter("ik_max_step_deg").value),
            joint_center_gain=0.0, pregrasp_rotation=None, hold_orientation=True,
        )
        if not success:
            self.get_logger().error("Post-grasp lift IK did not converge; skipping lift.")
            return
        try:
            self.send_angles_deg(np.rad2deg(q_sol).tolist(), int(self.get_parameter("move_speed").value))
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Post-grasp lift send failed: {exc}")
            return
        self.sleep_with_view(float(self.get_parameter("lift_settle_s").value), "LIFT +Z")

    def go_to_drop_pose(self):
        drop = [float(a) for a in self.get_parameter("drop_pose_deg").value]
        self.get_logger().info(f"Moving to drop pose q_deg={np.round(drop, 2).tolist()}")
        try:
            self.send_angles_deg(drop, int(self.get_parameter("drop_speed").value))
        except (OSError, RuntimeError, ValueError) as exc:
            self.get_logger().error(f"Drop pose send failed: {exc}")
        self.wait_until_at_pose(drop, "drop pose")
        self.sleep_with_view(float(self.get_parameter("drop_settle_s").value), "AT DROP POSE")

    def wait_until_at_pose(self, target_deg, name="pose"):
        tol = float(self.get_parameter("pose_reach_tol_deg").value)
        deadline = time.time() + float(self.get_parameter("pose_reach_timeout_s").value)
        last_err = None
        while time.time() < deadline:
            try:
                cur = self.get_angles_deg()
            except (OSError, RuntimeError, ValueError):
                time.sleep(0.3)
                continue
            last_err = max(abs(float(c) - float(t)) for c, t in zip(cur, target_deg))
            if last_err <= tol:
                self.get_logger().info(f"Reached {name} (max joint err {last_err:.2f} deg)")
                return True
            self.sleep_with_view(0.3, f"MOVING TO {name.upper()} (err {last_err:.1f} deg)")
        self.get_logger().warn(f"{name} reach timeout (max joint err {last_err} deg); proceeding")
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
            self.get_logger().error(f"Release command failed: {exc}")
        self.sleep_with_view(pulse_s + float(self.get_parameter("drop_release_margin_s").value),
                             "RELEASING (valve pulse)")

    def pick_attempt(self, target_centre):
        if not self.pregrasp_with_retry(target_centre):
            return False

        self.set_pump(True)
        move = self.gather_move(target_centre)
        if move is not None:
            self.execute_move(*move)
        else:
            self.get_logger().warn("No VS correction; descending with zero XY.")
            self.execute_move(0.0, 0.0)

        self.post_grasp_lift()
        self.go_to_drop_pose()
        self.release_at_drop()
        return True

    def run(self):
        self.go_home()
        self.get_logger().info("=== Grasp sequence started ===")
        max_attempts = int(self.get_parameter("max_grasp_attempts").value)

        centres = self.detect_cubes_stable()
        self.get_logger().info(f"Found {len(centres)} red cube(s) at start.")
        skipped = []

        while rclpy.ok():
            remaining = [c for c in centres if not self.near_any(c, skipped)]
            if not remaining:
                self.get_logger().info("No red cubes left to grasp.")
                break
            target = remaining[0]
            self.get_logger().info(
                f"Targeting leftmost cube at {np.round(target, 3).tolist()} "
                f"({len(remaining)} remaining)"
            )

            grasped = False
            skip_ik = False
            for attempt in range(max_attempts):
                count_before = len(centres)
                self.get_logger().info(f"--- Attempt {attempt + 1}/{max_attempts} (cubes seen: {count_before}) ---")
                if not self.pick_attempt(target):
                    skip_ik = True
                    break
                # Redetect at the drop pose: serves as both grasp check (count dropped?)
                # and the next round's target list.
                centres = self.detect_cubes_stable()
                count_after = len(centres)
                if count_after < count_before:
                    self.get_logger().info(f"Grasp confirmed by count: {count_before} -> {count_after}")
                    grasped = True
                    break
                self.get_logger().warn(f"Grasp NOT confirmed by count: {count_before} -> {count_after}; retrying")
                remaining = [c for c in centres if not self.near_any(c, skipped)]
                if not remaining:
                    break
                target = remaining[0]

            if skip_ik:
                self.get_logger().warn("Pre-grasp IK failed twice; skipping cube (out of reach).")
                skipped.append(np.asarray(target))
            elif not grasped:
                self.get_logger().warn(f"Grasp failed after {max_attempts} attempts; skipping cube.")
                skipped.append(np.asarray(target))

        self.go_home()
        self.get_logger().info("=== Grasp sequence finished ===")

    def load_t_wc(self):
        import json
        import os
        path = os.path.expanduser(self.get_parameter("extrinsic_file").value)
        if not os.path.exists(path):
            self.get_logger().error(f"Camera extrinsic not found: {path}")
            return None
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            transform = np.array(data["transform_matrix"], dtype=float)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(f"Failed to load extrinsic {path}: {exc}")
            return None
        if transform.shape != (4, 4):
            return None
        self.get_logger().info(f"Loaded T_WC from {path}")
        return transform

    def destroy_node(self):
        try:
            self.set_pump(False)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.pipe.stop()
        except RuntimeError:
            pass
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GraspSequenceNode()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
