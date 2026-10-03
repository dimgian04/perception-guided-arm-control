import json
import os

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String

from perception_pkg import top_face_center_mask as ctf


WINDOW_NAME = "claude top face"
MASK_WINDOW_NAME = "claude top face mask"


class ClaudeTopFaceCenterNode(Node):
    def __init__(self):
        super().__init__("top_face_center_node")

        # Publish / lifecycle contract (matches suction_points_node).
        self.declare_parameter("output_topic", "/perception/suction_point_world")
        self.declare_parameter("status_topic", "/perception/suction_point_status")
        self.declare_parameter("extrinsic_file", ctf.EXTRINSIC_FILE)
        self.declare_parameter("sample_duration_s", 2.0)
        self.declare_parameter("max_std_m", 0.02)
        self.declare_parameter("publish_once", True)
        self.declare_parameter("shutdown_after_publish", True)
        self.declare_parameter("locked_republish_count", 3)
        self.declare_parameter("show_visualization", True)

        # Camera config.
        self.declare_parameter("color_width", ctf.COLOR_WIDTH)
        self.declare_parameter("color_height", ctf.COLOR_HEIGHT)
        self.declare_parameter("depth_width", ctf.DEPTH_WIDTH)
        self.declare_parameter("depth_height", ctf.DEPTH_HEIGHT)
        self.declare_parameter("fps", ctf.FPS)
        self.declare_parameter("use_depth_filters", True)

        # Detection knobs (forwarded into the exact detector module).
        self.declare_parameter("cube_edge_m", ctf.CUBE_EDGE_M)
        self.declare_parameter("min_depth_m", ctf.MIN_DEPTH_M)
        self.declare_parameter("max_depth_m", ctf.MAX_DEPTH_M)
        self.declare_parameter("min_component_area_px", ctf.MIN_COMPONENT_AREA_PX)
        self.declare_parameter("hsv_s_min", ctf.RED_HSV_RANGES[0][0][1])
        self.declare_parameter("hsv_v_min", ctf.RED_HSV_RANGES[0][0][2])
        self.declare_parameter("band_below_m", ctf.TOP_BAND_BELOW_M)
        self.declare_parameter("band_above_m", ctf.TOP_BAND_ABOVE_M)
        self.declare_parameter("top_core_tol_m", ctf.TOP_CORE_TOL_M)
        self.declare_parameter("table_margin_px", ctf.TABLE_MARGIN_PX)
        self.declare_parameter("snap_to_known", True)

        # Push the tunables into the detector module so its functions stay exact.
        ctf.CUBE_EDGE_M = float(self.get_parameter("cube_edge_m").value)
        ctf.MIN_DEPTH_M = float(self.get_parameter("min_depth_m").value)
        ctf.MAX_DEPTH_M = float(self.get_parameter("max_depth_m").value)
        ctf.MIN_COMPONENT_AREA_PX = int(self.get_parameter("min_component_area_px").value)
        ctf.TOP_CORE_TOL_M = float(self.get_parameter("top_core_tol_m").value)
        ctf.TABLE_MARGIN_PX = int(self.get_parameter("table_margin_px").value)

        self.s_min = int(self.get_parameter("hsv_s_min").value)
        self.v_min = int(self.get_parameter("hsv_v_min").value)
        self.band_below_m = float(self.get_parameter("band_below_m").value)
        self.band_above_m = float(self.get_parameter("band_above_m").value)
        self.snap_to_known = bool(self.get_parameter("snap_to_known").value)

        self.output_pub = self.create_publisher(
            PointStamped, self.get_parameter("output_topic").value, 10
        )
        self.status_pub = self.create_publisher(
            String, self.get_parameter("status_topic").value, 10
        )

        self.t_wc = self.load_t_wc()
        if self.t_wc is None:
            self.get_logger().error(
                "No camera extrinsic (T_WC). Run the aruco calibration first; "
                "this node needs world +Z to find the table and the top face."
            )
        else:
            self.rotation = self.t_wc[:3, :3]
            self.translation = self.t_wc[:3, 3]
            self.up_cam = ctf.world_up_in_camera(self.t_wc)

        self.pipe = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.depth_scale = None
        self.depth_filters = []

        self.samples_world = []
        self.samples_camera = []
        self.sampling_start_time = None
        self.locked_world_point = None
        self.locked_publish_count = 0
        self.locked_timer = None
        self.shutdown_requested = False

        self.start_camera()
        if bool(self.get_parameter("use_depth_filters").value):
            self.depth_filters = ctf.create_depth_filters()

        if bool(self.get_parameter("show_visualization").value):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.namedWindow(MASK_WINDOW_NAME, cv2.WINDOW_NORMAL)

        fps = max(1, int(self.get_parameter("fps").value))
        self.frame_timer = self.create_timer(1.0 / fps, self.process_frame)
        self.get_logger().info(
            "Claude top-face center node ready. Publishing base-frame top-face "
            f"center on {self.get_parameter('output_topic').value}"
        )

    # ------------------------------------------------------------
    # Camera.
    # ------------------------------------------------------------

    def start_camera(self):
        cfg = rs.config()
        fps = int(self.get_parameter("fps").value)
        cfg.enable_stream(
            rs.stream.color,
            int(self.get_parameter("color_width").value),
            int(self.get_parameter("color_height").value),
            rs.format.bgr8,
            fps,
        )
        cfg.enable_stream(
            rs.stream.depth,
            int(self.get_parameter("depth_width").value),
            int(self.get_parameter("depth_height").value),
            rs.format.z16,
            fps,
        )
        profile = self.pipe.start(cfg)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.get_logger().info(
            "Started RealSense: "
            f"color={self.get_parameter('color_width').value}x{self.get_parameter('color_height').value}, "
            f"depth={self.get_parameter('depth_width').value}x{self.get_parameter('depth_height').value}, "
            f"depth_scale={self.depth_scale}"
        )

    # ------------------------------------------------------------
    # Per-frame top-face center via the exact detector.
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

        (hsv_mask, top_mask, corners_cam, p_camera, p_world, info,
         edge_len) = self.compute_center(color_image, depth_m, intrinsics)

        self.publish_status(p_camera, p_world, info, edge_len)
        self.handle_candidate(p_camera, p_world)

        if bool(self.get_parameter("show_visualization").value):
            self.show(color_image, intrinsics, hsv_mask, top_mask, corners_cam,
                      p_camera, p_world, info, edge_len)

    def compute_center(self, color_image, depth_m, intrinsics):
        hsv_mask = ctf.make_hsv_mask(color_image, self.s_min, self.v_min)
        hsv_mask = ctf.keep_largest_component(hsv_mask, ctf.MIN_COMPONENT_AREA_PX)

        params = {"band_below_m": self.band_below_m, "band_above_m": self.band_above_m}
        top_mask, fit_points, plane, info = ctf.detect_top_face(
            hsv_mask, depth_m, intrinsics, self.t_wc, self.up_cam, params
        )

        corners_cam = None
        p_camera = None
        p_world = None
        edge_len = None
        if plane is not None and fit_points is not None:
            corners_cam, edge_len = ctf.fit_square_corners(
                plane, fit_points, self.up_cam, self.snap_to_known
            )
            if corners_cam is not None:
                corners_world = (self.rotation @ corners_cam.T).T + self.translation
                p_world = np.mean(corners_world, axis=0)
                p_camera = self.rotation.T @ (p_world - self.translation)

        return hsv_mask, top_mask, corners_cam, p_camera, p_world, info, edge_len

    # ------------------------------------------------------------
    # Sample / average / publish-once contract.
    # ------------------------------------------------------------

    def handle_candidate(self, p_camera, p_world):
        if self.locked_world_point is not None:
            return

        if p_camera is None or p_world is None:
            self.samples_camera.clear()
            self.samples_world.clear()
            self.sampling_start_time = None
            return

        now = self.get_clock().now()
        if self.sampling_start_time is None:
            self.sampling_start_time = now
            self.samples_camera.clear()
            self.samples_world.clear()
            self.get_logger().info("Detected red cube top face; collecting center samples")

        self.samples_camera.append(np.array(p_camera, dtype=float))
        self.samples_world.append(np.array(p_world, dtype=float))

        elapsed = (now - self.sampling_start_time).nanoseconds * 1e-9
        if elapsed < float(self.get_parameter("sample_duration_s").value):
            return

        world_samples = np.array(self.samples_world, dtype=float)
        camera_samples = np.array(self.samples_camera, dtype=float)
        p_world_mean = np.mean(world_samples, axis=0)
        p_camera_mean = np.mean(camera_samples, axis=0)
        p_world_std = np.std(world_samples, axis=0)

        if np.max(p_world_std) > float(self.get_parameter("max_std_m").value):
            self.get_logger().warn(
                "Top-face center too jittery; restarting sample window. "
                f"world_std={np.round(p_world_std, 4).tolist()} m"
            )
            self.samples_camera.clear()
            self.samples_world.clear()
            self.sampling_start_time = None
            return

        self.publish_world_point(p_world_mean)
        self.locked_world_point = p_world_mean if bool(self.get_parameter("publish_once").value) else None
        self.locked_publish_count = 1
        self.get_logger().info(
            "Published top-face center: "
            f"camera={np.round(p_camera_mean, 4).tolist()} m, "
            f"world={np.round(p_world_mean, 4).tolist()} m, "
            f"world_std={np.round(p_world_std, 4).tolist()} m"
        )

        if self.locked_world_point is not None:
            self.locked_timer = self.create_timer(0.2, self.republish_locked_point)
        else:
            self.samples_camera.clear()
            self.samples_world.clear()
            self.sampling_start_time = None

    def publish_world_point(self, p_world):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "myarm_base_frame"
        msg.point.x = float(p_world[0])
        msg.point.y = float(p_world[1])
        msg.point.z = float(p_world[2])
        self.output_pub.publish(msg)

    def republish_locked_point(self):
        if self.locked_world_point is None or self.shutdown_requested:
            return

        if self.locked_publish_count >= int(self.get_parameter("locked_republish_count").value):
            if self.locked_timer is not None:
                self.locked_timer.cancel()
                self.locked_timer = None
            if bool(self.get_parameter("shutdown_after_publish").value):
                self.shutdown_requested = True
                self.get_logger().info("Top-face center published; shutting down perception node")
                rclpy.shutdown()
            return

        self.publish_world_point(self.locked_world_point)
        self.locked_publish_count += 1

    def publish_status(self, p_camera, p_world, info, edge_len):
        data = {
            "target_color": "red",
            "has_target": p_world is not None,
            "camera_m": None if p_camera is None else [float(v) for v in p_camera],
            "world_m": None if p_world is None else [float(v) for v in p_world],
            "z_table_m": info.get("z_table"),
            "z_top_m": info.get("z_top"),
            "measured_edge_m": None if edge_len is None else float(edge_len),
            "red_px": int(info.get("red_px", 0)),
            "top_px": int(info.get("top_px", 0)),
        }
        msg = String()
        msg.data = json.dumps(data)
        self.status_pub.publish(msg)

    # ------------------------------------------------------------
    # Visualization + extrinsic loading.
    # ------------------------------------------------------------

    def show(self, color_image, intrinsics, hsv_mask, top_mask, corners_cam,
             p_camera, p_world, info, edge_len):
        display = color_image.copy()
        ctf.overlay_mask(display, hsv_mask, (0, 0, 160), 0.12)

        fitted_mask = None
        if corners_cam is not None:
            pixel_corners = []
            ok = True
            for corner in corners_cam:
                px = ctf.project_point(corner, intrinsics)
                if px is None:
                    ok = False
                    break
                pixel_corners.append(px)
            if ok:
                fitted_mask = np.zeros(hsv_mask.shape, dtype=np.uint8)
                cv2.fillConvexPoly(fitted_mask, np.array(pixel_corners, dtype=np.int32), 255)

        draw_mask = fitted_mask if fitted_mask is not None else top_mask
        ctf.overlay_mask(display, draw_mask, (0, 255, 0), 0.45)
        ctf.draw_contours(display, draw_mask, (0, 255, 0), 2)

        if p_camera is not None:
            cpx = ctf.project_point(p_camera, intrinsics)
            if cpx is not None:
                cpx_i = (int(round(cpx[0])), int(round(cpx[1])))
                cv2.drawMarker(display, cpx_i, (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
                cv2.circle(display, cpx_i, 4, (0, 255, 0), -1)

        z_table = info.get("z_table")
        z_top = info.get("z_top")
        lines = [
            f"red_px={info.get('red_px', 0)} top_px={info.get('top_px', 0)}",
            (f"z_table={z_table*100:.1f}cm z_top={z_top*100:.1f}cm"
             if z_table is not None else "z_table=?? (no table depth)"),
        ]
        if edge_len is not None:
            lines.append(f"measured_edge={edge_len*100:.2f}cm (cube={ctf.CUBE_EDGE_M*100:.1f}cm)")
        if p_world is not None:
            cm = np.array(p_world) * 100.0
            lines.append(f"top_center_world=({cm[0]:+.1f}, {cm[1]:+.1f}, {cm[2]:+.1f}) cm")
        else:
            lines.append("top_center_world: none")
        ctf.draw_panel(display, lines)

        cv2.imshow(WINDOW_NAME, display)
        cv2.imshow(MASK_WINDOW_NAME, draw_mask if draw_mask is not None else hsv_mask)
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
        node = ClaudeTopFaceCenterNode()
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
