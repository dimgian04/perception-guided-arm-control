import json
import os

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from std_msgs.msg import String

from perception_pkg import top_face_center_mask as ctf


WINDOW_NAME = "visual servoing node"
MASK_WINDOW_NAME = "vs masks (green=top, blue=paper)"


class ClaudeVisualServoingNode(Node):
    def __init__(self):
        super().__init__("visual_servoing_perception_node")

        self.declare_parameter("output_topic", "/visual_servoing/move_cm")
        self.declare_parameter("status_topic", "/visual_servoing/status")
        self.declare_parameter("extrinsic_file", ctf.EXTRINSIC_FILE)

        # Warmup: discard the first seconds after the target appears (the blue
        # mask is briefly wrong while it settles), then average.
        self.declare_parameter("warmup_s", 2.0)
        # Gather/average window (the per-frame XY estimate is noisy).
        self.declare_parameter("gather_duration_s", 3.0)
        self.declare_parameter("min_samples", 15)
        self.declare_parameter("max_std_cm", 1.0)
        self.declare_parameter("publish_once", True)
        self.declare_parameter("locked_republish_count", 5)
        self.declare_parameter("shutdown_after_publish", False)
        self.declare_parameter("show_visualization", True)
        self.declare_parameter("fps", ctf.FPS)

        # End-effector geometry: blue tape sits on the cylindrical cup surface,
        # the true center is offset in world X (matches the standalone viewer).
        self.declare_parameter("ee_offset_x_m", -0.004)

        # Red top-face detector knobs.
        self.declare_parameter("min_depth_m", ctf.MIN_DEPTH_M)
        self.declare_parameter("max_depth_m", ctf.MAX_DEPTH_M)
        self.declare_parameter("red_s_min", ctf.RED_HSV_RANGES[0][0][1])
        self.declare_parameter("red_v_min", ctf.RED_HSV_RANGES[0][0][2])
        self.declare_parameter("band_below_m", ctf.TOP_BAND_BELOW_M)
        self.declare_parameter("band_above_m", ctf.TOP_BAND_ABOVE_M)

        # Blue-tape bounded HSV band (copied from ibvs.py defaults).
        self.declare_parameter("blue_h_lo", 94)
        self.declare_parameter("blue_h_hi", 105)
        self.declare_parameter("blue_s_lo", 55)
        self.declare_parameter("blue_s_hi", 140)
        self.declare_parameter("blue_v_lo", 95)
        self.declare_parameter("blue_v_hi", 155)
        # Pixel thresholds scaled for 1920x1080 (1.5x linear / 2.25x area).
        self.declare_parameter("blue_morph_kernel", 5)
        self.declare_parameter("blue_min_area_px", 340)
        self.declare_parameter("blue_depth_window_px", 31)

        ctf.MIN_DEPTH_M = float(self.get_parameter("min_depth_m").value)
        ctf.MAX_DEPTH_M = float(self.get_parameter("max_depth_m").value)

        self.output_pub = self.create_publisher(
            Vector3, self.get_parameter("output_topic").value, 10
        )
        self.status_pub = self.create_publisher(
            String, self.get_parameter("status_topic").value, 10
        )

        self.t_wc = self.load_t_wc()
        if self.t_wc is not None:
            self.rotation = self.t_wc[:3, :3]
            self.translation = self.t_wc[:3, 3]
            self.up_cam = ctf.world_up_in_camera(self.t_wc)

        self.pipe = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.depth_scale = None
        self.depth_filters = []

        self.samples = []          # list of (dx_cm, dy_cm)
        self.detect_start = None   # when the target first appeared (warmup clock)
        self.sampling_start = None
        self.locked_move = None    # (dx, dy) once published
        self.locked_count = 0
        self.locked_timer = None
        self.shutdown_requested = False

        self.start_camera()
        self.depth_filters = ctf.create_depth_filters()

        if bool(self.get_parameter("show_visualization").value):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.namedWindow(MASK_WINDOW_NAME, cv2.WINDOW_NORMAL)

        fps = max(1, int(self.get_parameter("fps").value))
        self.timer = self.create_timer(1.0 / fps, self.process_frame)
        self.get_logger().info(
            "Visual-servoing perception node ready. Averaging XY correction over "
            f"{float(self.get_parameter('gather_duration_s').value):.1f} s, "
            f"publishing cm on {self.get_parameter('output_topic').value}"
        )

    # ------------------------------------------------------------
    # Camera.
    # ------------------------------------------------------------

    def start_camera(self):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, ctf.COLOR_WIDTH, ctf.COLOR_HEIGHT, rs.format.bgr8, ctf.FPS)
        cfg.enable_stream(rs.stream.depth, ctf.DEPTH_WIDTH, ctf.DEPTH_HEIGHT, rs.format.z16, ctf.FPS)
        profile = self.pipe.start(cfg)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.get_logger().info(f"Started RealSense. depth_scale={self.depth_scale}")

    # ------------------------------------------------------------
    # Per-frame detection.
    # ------------------------------------------------------------

    def process_frame(self):
        if self.shutdown_requested or self.t_wc is None:
            return
        try:
            frames = self.align.process(self.pipe.wait_for_frames(1000))
        except RuntimeError as exc:
            self.get_logger().warn(f"RealSense frame wait failed: {exc}")
            return
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return
        for depth_filter in self.depth_filters:
            depth_frame = depth_filter.process(depth_frame)

        intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_m = depth_image.astype(np.float32) * float(self.depth_scale)
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        top_world, top_px, top_mask, top_corners_cam = self.detect_top(color_image, depth_m, intrinsics)
        blue_world, blue_mask = self.detect_blue(hsv, depth_m, intrinsics, top_px)

        ee_world = None
        move = None
        if blue_world is not None:
            ee_world = blue_world + np.array([float(self.get_parameter("ee_offset_x_m").value), 0.0, 0.0])
            if top_world is not None:
                move = (float((top_world[0] - ee_world[0]) * 100.0),
                        float((top_world[1] - ee_world[1]) * 100.0))

        self.accumulate(move)
        self.publish_status(top_world, blue_world, ee_world, move)

        if bool(self.get_parameter("show_visualization").value):
            self.show(color_image, intrinsics, top_mask, top_corners_cam, blue_mask,
                      top_world, blue_world, ee_world, move)

    def detect_top(self, color_image, depth_m, intrinsics):
        red_mask = ctf.make_hsv_mask(
            color_image,
            int(self.get_parameter("red_s_min").value),
            int(self.get_parameter("red_v_min").value),
        )
        red_mask = ctf.keep_largest_component(red_mask, ctf.MIN_COMPONENT_AREA_PX)
        params = {
            "band_below_m": float(self.get_parameter("band_below_m").value),
            "band_above_m": float(self.get_parameter("band_above_m").value),
        }
        top_mask, fit_points, plane, _info = ctf.detect_top_face(
            red_mask, depth_m, intrinsics, self.t_wc, self.up_cam, params
        )
        top_world = None
        top_px = None
        top_corners_cam = None
        if plane is not None and fit_points is not None:
            top_corners_cam, _ = ctf.fit_square_corners(plane, fit_points, self.up_cam, snap_to_known=True)
            if top_corners_cam is not None:
                corners_world = (self.rotation @ top_corners_cam.T).T + self.translation
                top_world = np.mean(corners_world, axis=0)
                top_px = self.world_to_pixel(top_world, intrinsics)
        return top_world, top_px, top_mask, top_corners_cam

    def detect_blue(self, hsv, depth_m, intrinsics, top_px):
        lower = np.array((int(self.get_parameter("blue_h_lo").value),
                          int(self.get_parameter("blue_s_lo").value),
                          int(self.get_parameter("blue_v_lo").value)), dtype=np.uint8)
        upper = np.array((int(self.get_parameter("blue_h_hi").value),
                          int(self.get_parameter("blue_s_hi").value),
                          int(self.get_parameter("blue_v_hi").value)), dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        k = max(1, int(self.get_parameter("blue_morph_kernel").value))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        min_area = int(self.get_parameter("blue_min_area_px").value)
        ref_px = None if top_px is None else (top_px[0], top_px[1])
        blue_mask = self.select_nearest_component(mask, ref_px, min_area)
        blue_world = self.blue_center_world(blue_mask, depth_m, intrinsics)
        return blue_world, blue_mask

    def select_nearest_component(self, mask, ref_px, min_area_px):
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return np.zeros_like(mask)
        cands = [i for i in range(1, count) if int(stats[i, cv2.CC_STAT_AREA]) >= min_area_px]
        if not cands:
            return np.zeros_like(mask)
        if ref_px is None:
            best = max(cands, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
        else:
            rx, ry = ref_px
            best = min(cands, key=lambda i: (centroids[i][0] - rx) ** 2 + (centroids[i][1] - ry) ** 2)
        return (labels == best).astype(np.uint8) * 255

    def blue_center_world(self, blue_mask, depth_m, intrinsics):
        moments = cv2.moments(blue_mask, binaryImage=True)
        if moments["m00"] <= 0.0:
            return None
        cu = moments["m10"] / moments["m00"]
        cv_ = moments["m01"] / moments["m00"]
        cu_i, cv_i = int(round(cu)), int(round(cv_))

        half = int(self.get_parameter("blue_depth_window_px").value) // 2
        h, w = depth_m.shape
        y0, y1 = max(0, cv_i - half), min(h, cv_i + half + 1)
        x0, x1 = max(0, cu_i - half), min(w, cu_i + half + 1)
        depth_win = depth_m[y0:y1, x0:x1]
        mask_win = blue_mask[y0:y1, x0:x1] > 0
        valid = mask_win & np.isfinite(depth_win) & (depth_win >= ctf.MIN_DEPTH_M) & (depth_win <= ctf.MAX_DEPTH_M)
        samples = depth_win[valid]
        if samples.size < 5:
            all_valid = (blue_mask > 0) & np.isfinite(depth_m) & (depth_m >= ctf.MIN_DEPTH_M) & (depth_m <= ctf.MAX_DEPTH_M)
            samples = depth_m[all_valid]
            if samples.size < 5:
                return None
        z = float(np.median(samples))
        x = (cu - float(intrinsics.ppx)) * z / float(intrinsics.fx)
        y = (cv_ - float(intrinsics.ppy)) * z / float(intrinsics.fy)
        return self.rotation @ np.array([x, y, z]) + self.translation

    def world_to_pixel(self, point_world, intrinsics):
        point_cam = self.rotation.T @ (np.asarray(point_world, dtype=float) - self.translation)
        return ctf.project_point(point_cam, intrinsics)

    # ------------------------------------------------------------
    # Gather / average / publish-once.
    # ------------------------------------------------------------

    def accumulate(self, move):
        if self.locked_move is not None:
            return
        if move is None:
            self.samples.clear()
            self.sampling_start = None
            self.detect_start = None
            return

        now = self.get_clock().now()
        warmup_s = float(self.get_parameter("warmup_s").value)

        # Start (or hold during) the warmup window before averaging.
        if self.detect_start is None:
            self.detect_start = now
            self.get_logger().info(
                f"Cube top + blue paper visible; warming up {warmup_s:.1f} s "
                "before averaging (letting the blue mask settle)"
            )
            return
        if (now - self.detect_start).nanoseconds * 1e-9 < warmup_s:
            return  # still settling; discard these early samples.

        if self.sampling_start is None:
            self.sampling_start = now
            self.samples.clear()
            self.get_logger().info("Warmup done; gathering XY correction samples")
        self.samples.append(np.array(move, dtype=float))

        elapsed = (now - self.sampling_start).nanoseconds * 1e-9
        if elapsed < float(self.get_parameter("gather_duration_s").value):
            return
        if len(self.samples) < int(self.get_parameter("min_samples").value):
            return

        arr = np.array(self.samples, dtype=float)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        if float(np.max(std)) > float(self.get_parameter("max_std_cm").value):
            self.get_logger().warn(
                f"XY correction too jittery (std={np.round(std, 2).tolist()} cm); restarting window"
            )
            self.samples.clear()
            self.sampling_start = None
            return

        self.publish_move(mean)
        self.locked_move = mean if bool(self.get_parameter("publish_once").value) else None
        self.locked_count = 1
        self.get_logger().info(
            f"Published XY correction: x={mean[0]:+.2f} cm y={mean[1]:+.2f} cm "
            f"(std={np.round(std, 2).tolist()} cm, n={len(self.samples)})"
        )
        if self.locked_move is not None:
            self.locked_timer = self.create_timer(0.2, self.republish_locked)
        else:
            self.samples.clear()
            self.sampling_start = None

    def publish_move(self, move):
        msg = Vector3()
        msg.x = float(move[0])
        msg.y = float(move[1])
        msg.z = 0.0
        self.output_pub.publish(msg)

    def republish_locked(self):
        if self.locked_move is None or self.shutdown_requested:
            return
        if self.locked_count >= int(self.get_parameter("locked_republish_count").value):
            if self.locked_timer is not None:
                self.locked_timer.cancel()
                self.locked_timer = None
            if bool(self.get_parameter("shutdown_after_publish").value):
                self.shutdown_requested = True
                self.get_logger().info("XY correction published; shutting down VS perception node")
                rclpy.shutdown()
            return
        self.publish_move(self.locked_move)
        self.locked_count += 1

    def publish_status(self, top_world, blue_world, ee_world, move):
        data = {
            "has_top": top_world is not None,
            "has_blue": blue_world is not None,
            "cube_top_world_m": None if top_world is None else [float(v) for v in top_world],
            "blue_world_m": None if blue_world is None else [float(v) for v in blue_world],
            "ee_center_world_m": None if ee_world is None else [float(v) for v in ee_world],
            "move_cm": None if move is None else [float(move[0]), float(move[1])],
            "samples": len(self.samples),
            "published": self.locked_move is not None,
        }
        msg = String()
        msg.data = json.dumps(data)
        self.status_pub.publish(msg)

    # ------------------------------------------------------------
    # Visualization + extrinsic.
    # ------------------------------------------------------------

    def show(self, color_image, intrinsics, top_mask, top_corners_cam, blue_mask,
             top_world, blue_world, ee_world, move):
        display = color_image.copy()
        ctf.overlay_mask(display, blue_mask, (255, 60, 0), 0.30)
        ctf.draw_contours(display, blue_mask, (255, 120, 0), 2)

        fitted_mask = None
        if top_corners_cam is not None:
            pts, ok = [], True
            for corner in top_corners_cam:
                px = ctf.project_point(corner, intrinsics)
                if px is None:
                    ok = False
                    break
                pts.append(px)
            if ok:
                fitted_mask = np.zeros(blue_mask.shape, dtype=np.uint8)
                cv2.fillConvexPoly(fitted_mask, np.array(pts, dtype=np.int32), 255)
        draw_top = fitted_mask if fitted_mask is not None else top_mask
        ctf.overlay_mask(display, draw_top, (0, 255, 0), 0.40)
        ctf.draw_contours(display, draw_top, (0, 255, 0), 2)

        for world, color, label in (
            (top_world, (0, 255, 0), "CUBE TOP"),
            (blue_world, (255, 150, 0), "BLUE"),
            (ee_world, (0, 255, 255), "EE CENTER"),
        ):
            if world is None:
                continue
            px = self.world_to_pixel(world, intrinsics)
            if px is None:
                continue
            px_i = (int(round(px[0])), int(round(px[1])))
            cv2.drawMarker(display, px_i, color, cv2.MARKER_CROSS, 22, 2)
            cv2.circle(display, px_i, 4, color, -1)
            cv2.putText(display, label, (px_i[0] + 8, px_i[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        lines = ["VISUAL SERVOING NODE"]
        if move is not None:
            lines.append(f"move x={move[0]:+.2f} cm  y={move[1]:+.2f} cm")
        else:
            lines.append("move: waiting for cube top + blue paper")
        lines.append(f"samples={len(self.samples)} published={self.locked_move is not None}")
        x0, y0 = 12, 12
        overlay = display.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + 520, y0 + 26 * len(lines) + 16), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.72, display, 0.28, 0.0, dst=display)
        for idx, line in enumerate(lines):
            cv2.putText(display, line, (x0 + 14, y0 + 26 + 24 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255),
                        2 if idx == 0 else 1, cv2.LINE_AA)

        masks_vis = np.zeros_like(color_image)
        if draw_top is not None:
            masks_vis[draw_top > 0] = (0, 255, 0)
        masks_vis[blue_mask > 0] = (255, 120, 0)
        cv2.imshow(WINDOW_NAME, display)
        cv2.imshow(MASK_WINDOW_NAME, masks_vis)
        cv2.waitKey(1)

    def load_t_wc(self):
        extrinsic_file = os.path.expanduser(self.get_parameter("extrinsic_file").value)
        if not os.path.exists(extrinsic_file):
            self.get_logger().error(f"Camera extrinsic file not found: {extrinsic_file}")
            return None
        try:
            with open(extrinsic_file, "r", encoding="utf-8") as file:
                data = json.load(file)
            transform = np.array(data["transform_matrix"], dtype=float)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"Failed to load camera extrinsic {extrinsic_file}: {exc}")
            return None
        if transform.shape != (4, 4):
            self.get_logger().error(f"Camera extrinsic must be 4x4, got {transform.shape}")
            return None
        self.get_logger().info(
            f"Loaded T_WC from {extrinsic_file}: t={np.round(transform[:3, 3], 4).tolist()} m"
        )
        return transform

    def destroy_node(self):
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
        node = ClaudeVisualServoingNode()
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
