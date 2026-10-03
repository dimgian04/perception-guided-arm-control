import json
import math
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


# Set these to the ArUco marker's known pose in the robot base/world frame.
MARKER_WORLD_XYZ_M = [-0.015, -0.200, -0.010]
MARKER_WORLD_RPY_RAD = [0.000, 0.000, 0.000]

MARKER_LENGTH_M = 0.100
MARKER_ID = -1
ARUCO_DICT = cv2.aruco.DICT_4X4_50
DEFAULT_OUTPUT_FILE = "~/.ros/myarm_camera_extrinsic.json"


def rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_rot(roll, pitch, yaw):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def make_transform(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def project_to_rotation(rotation):
    # SVD-based projection: nearest valid rotation matrix, handles numerical
    # drift that can accumulate when averaging many rotation matrices directly.
    u, _, vt = np.linalg.svd(rotation)
    projected = u @ vt
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


class ArucoExtrinsicCalibrator(Node):
    def __init__(self):
        super().__init__("aruco_extrinsic_calibrator")

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("marker_length_m", MARKER_LENGTH_M)
        self.declare_parameter("marker_id", MARKER_ID)
        self.declare_parameter("marker_world_xyz_m", MARKER_WORLD_XYZ_M)
        self.declare_parameter("marker_world_rpy_rad", MARKER_WORLD_RPY_RAD)
        self.declare_parameter("sample_duration_s", 2.0)
        self.declare_parameter("min_samples", 20)
        self.declare_parameter("output_file", DEFAULT_OUTPUT_FILE)
        self.declare_parameter("show_visualization", False)
        self.declare_parameter("pose_log_period_s", 0.5)

        self.bridge = CvBridge()
        self.camera_info = None
        self.samples = []
        self.start_time = None
        self.last_waiting_log_time = None
        self.last_pose_log_time = None
        self.window_ready = False
        self.done = False

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.detector_params = self.create_detector_params()

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter("color_topic").value,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "Looking for ArUco marker to calibrate camera extrinsic. "
            "Update MARKER_WORLD_XYZ_M / MARKER_WORLD_RPY_RAD before trusting the result."
        )
        self.get_logger().info(
            "Active marker pose from ROS parameters: "
            f"xyz_m={list(self.get_parameter('marker_world_xyz_m').value)}, "
            f"rpy_rad={list(self.get_parameter('marker_world_rpy_rad').value)}"
        )

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def image_callback(self, msg):
        if self.done:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.camera_info is None:
            self.show_image(frame, None, None)
            self.log_waiting("Waiting for CameraInfo intrinsics")
            return

        target_id = int(self.get_parameter("marker_id").value)
        corners, ids = self.detect_markers(frame, target_id)
        if ids is None:
            self.show_image(frame, None, None)
            self.log_waiting("Waiting to see ArUco marker")
            return

        flat_ids = ids.flatten()
        if target_id < 0:
            marker_index = 0
            target_id = int(flat_ids[marker_index])
        else:
            matches = np.where(flat_ids == target_id)[0]
            if matches.size == 0:
                self.show_image(frame, corners, ids)
                self.log_waiting(f"Saw marker IDs {flat_ids.tolist()}, waiting for ID {target_id}")
                return
            marker_index = int(matches[0])
        marker_corners = [corners[marker_index]]
        rvec, tvec = self.estimate_marker_pose(marker_corners[0])
        if rvec is None:
            self.log_waiting("Marker visible, but pose estimation failed")
            return

        r_camera_marker, _ = cv2.Rodrigues(rvec)
        t_camera_marker = np.asarray(tvec, dtype=float)
        t_world_camera = self.compute_world_camera(r_camera_marker, t_camera_marker)
        self.samples.append(t_world_camera)
        self.log_marker_pose(target_id, r_camera_marker, t_camera_marker)
        self.show_image(frame, marker_corners, np.array([[target_id]], dtype=np.int32), rvec, tvec)

        now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = now
            self.get_logger().info("Marker detected; collecting calibration samples")
            return

        elapsed = (now - self.start_time).nanoseconds * 1e-9
        sample_duration = float(self.get_parameter("sample_duration_s").value)
        min_samples = int(self.get_parameter("min_samples").value)
        if elapsed < sample_duration or len(self.samples) < min_samples:
            return

        self.save_average_transform()
        self.done = True
        rclpy.shutdown()

    def create_detector_params(self):
        # OpenCV 4.6 exposes DetectorParameters(), but passing it to the old
        # detectMarkers API can segfault; the factory method is the safe path.
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            return cv2.aruco.DetectorParameters_create()
        return cv2.aruco.DetectorParameters()

    def detect_markers(self, frame, target_id):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.detector_params,
        )
        return corners, ids

    def estimate_marker_pose(self, marker_corners):
        length = float(self.get_parameter("marker_length_m").value)
        half = length / 2.0
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )
        image_points = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix(),
            self.dist_coeffs(),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix(),
                self.dist_coeffs(),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        if not ok:
            return None, None
        return rvec.reshape(3), tvec.reshape(3)

    def camera_matrix(self):
        k = self.camera_info.k
        return np.array(
            [[k[0], k[1], k[2]], [k[3], k[4], k[5]], [k[6], k[7], k[8]]],
            dtype=np.float64,
        )

    def dist_coeffs(self):
        if not self.camera_info.d:
            return np.zeros(5, dtype=np.float64)
        return np.array(self.camera_info.d, dtype=np.float64)

    def marker_world_transform(self):
        xyz = np.array(self.get_parameter("marker_world_xyz_m").value, dtype=float)
        rpy = np.array(self.get_parameter("marker_world_rpy_rad").value, dtype=float)
        return make_transform(rpy_to_rot(rpy[0], rpy[1], rpy[2]), xyz)

    def compute_world_camera(self, r_camera_marker, t_camera_marker):
        t_camera_marker_tf = make_transform(r_camera_marker, t_camera_marker)
        return self.marker_world_transform() @ np.linalg.inv(t_camera_marker_tf)

    def save_average_transform(self):
        transforms = np.array(self.samples)
        translation = np.mean(transforms[:, :3, 3], axis=0)
        rotation = project_to_rotation(np.mean(transforms[:, :3, :3], axis=0))
        transform = make_transform(rotation, translation)
        inverse_transform = np.linalg.inv(transform)

        output_file = os.path.expanduser(self.get_parameter("output_file").value)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        data = {
            "frame_id": "myarm_base_frame",
            "child_frame_id": "camera_color_optical_frame",
            "translation_m": translation.tolist(),
            "rotation_matrix": rotation.tolist(),
            "transform_matrix": transform.tolist(),
            "inverse_transform_matrix": inverse_transform.tolist(),
            "sample_count": len(self.samples),
            "translation_std_m": np.std(transforms[:, :3, 3], axis=0).tolist(),
            "marker_length_m": float(self.get_parameter("marker_length_m").value),
            "marker_world_xyz_m": list(self.get_parameter("marker_world_xyz_m").value),
            "marker_world_rpy_rad": list(self.get_parameter("marker_world_rpy_rad").value),
        }
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        self.get_logger().info(
            "Saved camera extrinsic: "
            f"T_WC translation={np.round(translation, 4).tolist()} m, "
            f"samples={len(self.samples)}, file={output_file}"
        )
        self.get_logger().info(
            "T_WC maps camera-frame points into world/base frame: p_W = T_WC * p_C\n"
            f"{np.array2string(transform, precision=5, suppress_small=True)}"
        )
        self.get_logger().info(
            "T_CW maps world/base-frame points into camera frame: p_C = T_CW * p_W\n"
            f"{np.array2string(inverse_transform, precision=5, suppress_small=True)}"
        )

    def show_image(self, frame, corners, ids, rvec=None, tvec=None):
        if not bool(self.get_parameter("show_visualization").value):
            return

        display = frame.copy()
        if corners is not None and ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                points = marker_corners.reshape(4, 2).astype(int)
                cv2.polylines(display, [points], True, (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"id={int(marker_id)}",
                    tuple(points[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            if rvec is not None and tvec is not None:
                self.draw_labeled_axes(
                    display,
                    rvec,
                    tvec,
                    float(self.get_parameter("marker_length_m").value) * 0.5,
                )

        window_name = "aruco extrinsic calibration"
        if not self.window_ready:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, display.shape[1], display.shape[0])
            self.window_ready = True

        cv2.imshow(window_name, display)
        cv2.waitKey(1)

    def draw_labeled_axes(self, display, rvec, tvec, axis_length):
        axis_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_length, 0.0, 0.0],
                [0.0, axis_length, 0.0],
                [0.0, 0.0, axis_length],
            ],
            dtype=np.float64,
        )
        image_points, _ = cv2.projectPoints(
            axis_points,
            rvec,
            tvec,
            self.camera_matrix(),
            self.dist_coeffs(),
        )
        origin, x_end, y_end, z_end = image_points.reshape(-1, 2).astype(int)

        axes = [
            (x_end, "X", (0, 0, 255)),
            (y_end, "Y", (0, 255, 0)),
            (z_end, "Z", (255, 0, 0)),
        ]
        for end_point, label, color in axes:
            cv2.line(display, tuple(origin), tuple(end_point), color, 3)
            cv2.putText(
                display,
                label,
                tuple(end_point),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.circle(display, tuple(origin), 5, (255, 255, 255), -1)

    def log_marker_pose(self, marker_id, r_camera_marker, t_camera_marker):
        now = self.get_clock().now()
        if self.last_pose_log_time is not None:
            elapsed = (now - self.last_pose_log_time).nanoseconds * 1e-9
            if elapsed < float(self.get_parameter("pose_log_period_s").value):
                return
        self.last_pose_log_time = now

        marker_x_in_camera = r_camera_marker[:, 0]
        marker_y_in_camera = r_camera_marker[:, 1]
        marker_z_in_camera = r_camera_marker[:, 2]
        t_cm_cm = t_camera_marker * 100.0
        self.get_logger().info(
            f"OpenCV marker {marker_id}: "
            f"t_CM_cm={np.round(t_cm_cm, 2).tolist()}, "
            f"+X_M_in_C={np.round(marker_x_in_camera, 3).tolist()}, "
            f"+Y_M_in_C={np.round(marker_y_in_camera, 3).tolist()}, "
            f"+Z_M_in_C={np.round(marker_z_in_camera, 3).tolist()}"
        )

    def log_waiting(self, text):
        now = self.get_clock().now()
        if self.last_waiting_log_time is not None:
            elapsed = (now - self.last_waiting_log_time).nanoseconds * 1e-9
            if elapsed < 1.0:
                return
        self.last_waiting_log_time = now
        self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ArucoExtrinsicCalibrator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
