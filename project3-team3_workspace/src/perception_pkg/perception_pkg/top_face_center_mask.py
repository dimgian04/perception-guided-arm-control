import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs


CUBE_EDGE_M = 0.024

COLOR_WIDTH = 1920
COLOR_HEIGHT = 1080
DEPTH_WIDTH = 1280
DEPTH_HEIGHT = 720
FPS = 30

MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 1.20
EXTRINSIC_FILE = "~/.ros/myarm_camera_extrinsic.json"

# Red wraps around 0/179 in OpenCV HSV (H=0..179).
RED_HSV_RANGES = [
    ((0, 112, 73), (12, 255, 255)),
    ((170, 112, 73), (179, 255, 255)),
]

MORPH_KERNEL_SIZE = 5
MIN_COMPONENT_AREA_PX = 340

TABLE_MARGIN_PX = 120
TABLE_MIN_SAMPLES = 400
TABLE_HIST_BIN_M = 0.002
TABLE_REFINE_BAND_M = 0.006

TOP_BAND_BELOW_M = 0.006
TOP_BAND_ABOVE_M = 0.008
# Tight core band for square fitting: excludes side-face rim pixels that would
# inflate the rectangle and bias the center estimate toward the sides.
TOP_CORE_TOL_M = 0.003

# Stacked cube support: snap detected height to z_table + k*CUBE_EDGE_M.
TOP_HEIGHT_PERCENTILE = 95.0
MAX_STACK_LEVELS = 3
Z_GROUP_TOL_M = 0.012

CORNER_SMOOTHING_ALPHA = 0.35

WINDOW_NAME = "top face mask"
MASK_WINDOW_NAME = "final mask"
CONTROLS_WINDOW = "controls"

MASK_SOURCES = ("fitted-square", "top-pixels", "intersection")


def start_realsense():
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)

    profile = pipe.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"Started RealSense. depth_scale={depth_scale}")
    return pipe, align, depth_scale


def create_depth_filters():
    return [
        rs.spatial_filter(),
        rs.temporal_filter(),
        rs.hole_filling_filter(),
    ]


def apply_depth_filters(depth_frame, depth_filters):
    filtered = depth_frame
    for depth_filter in depth_filters:
        filtered = depth_filter.process(filtered)
    return filtered


def load_t_wc():
    extrinsic_file = os.path.expanduser(EXTRINSIC_FILE)
    if not os.path.exists(extrinsic_file):
        print(f"No extrinsic file found at {extrinsic_file}")
        print("This top-face detector needs T_WC so it knows world +Z (up).")
        return None

    try:
        with open(extrinsic_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        transform = np.array(data["transform_matrix"], dtype=float)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to load extrinsic from {extrinsic_file}: {exc}")
        return None

    if transform.shape != (4, 4):
        print(f"Bad extrinsic shape {transform.shape}; expected 4x4.")
        return None

    print(f"Loaded T_WC from {extrinsic_file}")
    print(f"  camera position in world = {np.round(transform[:3, 3], 4).tolist()} m")
    return transform


def world_up_in_camera(t_wc):
    up_cam = t_wc[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    return up_cam / float(np.linalg.norm(up_cam))


def make_odd(value):
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def make_hsv_mask(color_image, s_min, v_min):
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for (h_lo, _s, _v), (h_hi, s_hi, v_hi) in RED_HSV_RANGES:
        lower = np.array((h_lo, s_min, v_min), dtype=np.uint8)
        upper = np.array((h_hi, s_hi, v_hi), dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    kernel_size = make_odd(MORPH_KERNEL_SIZE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def keep_largest_component(mask, min_area_px):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask

    valid_labels = [
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area_px)
    ]
    if not valid_labels:
        return np.zeros_like(mask)

    best_label = max(valid_labels, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    return (labels == best_label).astype(np.uint8) * 255


def world_height_of_pixels(rows, cols, depth_m, intrinsics, up_cam, t_z):
    z = depth_m[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - float(intrinsics.ppx)) * z / float(intrinsics.fx)
    y = (rows.astype(np.float64) - float(intrinsics.ppy)) * z / float(intrinsics.fy)
    points_cam = np.column_stack((x, y, z))
    heights = points_cam @ up_cam + t_z
    return heights, points_cam


def estimate_local_table_height(depth_m, red_mask, cube_bbox, intrinsics, up_cam, t_z):
    height, width = depth_m.shape
    x0, y0, w, h = cube_bbox
    xa = max(0, x0 - TABLE_MARGIN_PX)
    ya = max(0, y0 - TABLE_MARGIN_PX)
    xb = min(width, x0 + w + TABLE_MARGIN_PX)
    yb = min(height, y0 + h + TABLE_MARGIN_PX)

    sub_depth = depth_m[ya:yb, xa:xb]
    sub_red = red_mask[ya:yb, xa:xb]
    valid = (
        np.isfinite(sub_depth)
        & (sub_depth >= MIN_DEPTH_M)
        & (sub_depth <= MAX_DEPTH_M)
        & (sub_red == 0)
    )
    rows_local, cols_local = np.nonzero(valid)
    if rows_local.size < TABLE_MIN_SAMPLES:
        return None

    heights, _ = world_height_of_pixels(
        rows_local + ya, cols_local + xa, depth_m, intrinsics, up_cam, t_z
    )
    heights = heights[np.isfinite(heights)]
    if heights.size < TABLE_MIN_SAMPLES:
        return None

    # Histogram peak finds the dominant flat surface (the table), then a median
    # over the narrow peak band removes depth outliers.
    z_lo, z_hi = np.percentile(heights, [1.0, 99.0])
    if z_hi - z_lo < 1e-4:
        return float(np.median(heights))
    bins = max(8, int((z_hi - z_lo) / TABLE_HIST_BIN_M))
    hist, edges = np.histogram(heights, bins=bins, range=(z_lo, z_hi))
    peak = int(np.argmax(hist))
    z_peak = 0.5 * (edges[peak] + edges[peak + 1])
    near = heights[np.abs(heights - z_peak) <= TABLE_REFINE_BAND_M]
    if near.size == 0:
        return float(z_peak)
    return float(np.median(near))


def raycast_pixels_to_plane(rows, cols, intrinsics, n_cam, d0):
    """Intersect each pixel ray with the camera-frame plane n_cam . p = d0."""
    dx = (cols.astype(np.float64) - float(intrinsics.ppx)) / float(intrinsics.fx)
    dy = (rows.astype(np.float64) - float(intrinsics.ppy)) / float(intrinsics.fy)
    dirs = np.column_stack((dx, dy, np.ones_like(dx)))
    denom = dirs @ n_cam
    safe = np.abs(denom) > 1e-9
    s = np.zeros_like(denom)
    s[safe] = d0 / denom[safe]
    valid = safe & (s > 0)
    points_cam = s[:, None] * dirs
    return points_cam[valid], valid


def detect_top_face(red_mask, depth_m, intrinsics, t_wc, up_cam, params):
    info = {"red_px": int(cv2.countNonZero(red_mask)), "top_px": 0,
            "z_table": None, "z_top": None}

    rows, cols = np.nonzero(red_mask > 0)
    if rows.size < MIN_COMPONENT_AREA_PX:
        return None, None, None, info

    x0, x1 = int(cols.min()), int(cols.max())
    y0, y1 = int(rows.min()), int(rows.max())
    cube_bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    t_z = float(t_wc[2, 3])
    z_table = estimate_local_table_height(depth_m, red_mask, cube_bbox, intrinsics, up_cam, t_z)
    if z_table is None:
        return None, None, None, info
    info["z_table"] = z_table

    z = depth_m[rows, cols]
    has_depth = np.isfinite(z) & (z >= MIN_DEPTH_M) & (z <= MAX_DEPTH_M)
    rows_d = rows[has_depth]
    cols_d = cols[has_depth]
    if rows_d.size < MIN_COMPONENT_AREA_PX:
        return None, None, None, info

    heights, _ = world_height_of_pixels(rows_d, cols_d, depth_m, intrinsics, up_cam, t_z)

    # Snap to the nearest stack level k so stacked cubes get their true top face.
    red_top_z = float(np.percentile(heights, TOP_HEIGHT_PERCENTILE))
    level = int(round((red_top_z - z_table) / CUBE_EDGE_M))
    level = max(1, min(level, MAX_STACK_LEVELS))
    z_top = z_table + level * CUBE_EDGE_M
    info["z_top"] = z_top
    info["level"] = level

    top_sel = (heights >= z_top - params["band_below_m"]) & (heights <= z_top + params["band_above_m"])
    if int(np.count_nonzero(top_sel)) < MIN_COMPONENT_AREA_PX:
        return None, None, None, info

    top_mask = np.zeros_like(red_mask)
    top_mask[rows_d[top_sel], cols_d[top_sel]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, kernel)
    top_mask = keep_largest_component(top_mask, MIN_COMPONENT_AREA_PX)
    top_mask = cv2.bitwise_and(top_mask, red_mask)

    top_rows, top_cols = np.nonzero(top_mask > 0)
    info["top_px"] = int(top_rows.size)
    if top_rows.size < MIN_COMPONENT_AREA_PX:
        return top_mask, None, None, info

    core_in_mask = top_mask[top_rows, top_cols] > 0
    core_heights, _ = world_height_of_pixels(top_rows, top_cols, depth_m, intrinsics, up_cam, t_z)
    core_sel = core_in_mask & (np.abs(core_heights - z_top) <= TOP_CORE_TOL_M)
    fit_rows = top_rows[core_sel] if int(np.count_nonzero(core_sel)) >= MIN_COMPONENT_AREA_PX else top_rows
    fit_cols = top_cols[core_sel] if int(np.count_nonzero(core_sel)) >= MIN_COMPONENT_AREA_PX else top_cols

    # Ray-cast onto the known top plane for bias-free 3D point positions.
    d0 = z_top - t_z
    fit_points, _ = raycast_pixels_to_plane(fit_rows, fit_cols, intrinsics, up_cam, d0)
    if fit_points.shape[0] < MIN_COMPONENT_AREA_PX:
        return top_mask, None, None, info

    center = np.mean(fit_points, axis=0)
    plane = (center, up_cam.copy())
    return top_mask, fit_points, plane, info


def plane_basis(plane, up_cam):
    center, normal = plane
    seed = up_cam - (up_cam @ normal) * normal
    if float(np.linalg.norm(seed)) < 1e-6:
        seed = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
    u = seed / float(np.linalg.norm(seed))
    v = np.cross(normal, u)
    v /= float(np.linalg.norm(v))
    return center, normal, u, v


def fit_square_corners(plane, top_points, up_cam, snap_to_known):
    if top_points is None or top_points.shape[0] < 3:
        return None, None

    center, normal, u, v = plane_basis(plane, up_cam)
    rel = top_points - center
    coords = np.column_stack((rel @ u, rel @ v)).astype(np.float32)

    rect = cv2.minAreaRect(coords)
    (cx, cy), (w, h), angle = rect
    box = cv2.boxPoints(rect).astype(np.float64)

    if snap_to_known:
        # Override measured size with known edge length to remove depth-noise bias.
        theta = np.deg2rad(angle)
        ax = np.array([np.cos(theta), np.sin(theta)])
        ay = np.array([-np.sin(theta), np.cos(theta)])
        half = 0.5 * CUBE_EDGE_M
        centre2d = np.array([cx, cy])
        box = np.array([
            centre2d - half * ax - half * ay,
            centre2d + half * ax - half * ay,
            centre2d + half * ax + half * ay,
            centre2d - half * ax + half * ay,
        ])

    corners_cam = center[None, :] + box[:, 0:1] * u[None, :] + box[:, 1:2] * v[None, :]
    edge_len = float(0.5 * (w + h))
    return corners_cam, edge_len


def cube_sort_key(center_world):
    z = float(center_world[2])
    y = float(center_world[1])
    return (-round(z / Z_GROUP_TOL_M), y)


def detect_indexed_cubes(color_image, depth_m, intrinsics, t_wc, up_cam, params,
                         s_min, v_min, min_area_px=None):
    if min_area_px is None:
        min_area_px = MIN_COMPONENT_AREA_PX
    rotation = t_wc[:3, :3]
    translation = t_wc[:3, 3]

    full_red = make_hsv_mask(color_image, s_min, v_min)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(full_red, connectivity=8)

    cubes = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < int(min_area_px):
            continue
        component = (labels == label).astype(np.uint8) * 255
        top_mask, fit_points, plane, info = detect_top_face(
            component, depth_m, intrinsics, t_wc, up_cam, params
        )
        if plane is None or fit_points is None:
            continue
        corners_cam, edge_len = fit_square_corners(plane, fit_points, up_cam, snap_to_known=True)
        if corners_cam is None:
            continue
        corners_world = (rotation @ corners_cam.T).T + translation
        center_world = np.mean(corners_world, axis=0)
        cubes.append({
            "center_world": center_world,
            "center_camera": rotation.T @ (center_world - translation),
            "top_mask": top_mask,
            "corners_cam": corners_cam,
            "edge_cm": None if edge_len is None else float(edge_len * 100.0),
            "info": info,
        })

    cubes.sort(key=lambda cube: cube_sort_key(cube["center_world"]))
    for idx, cube in enumerate(cubes):
        cube["index"] = idx
    return cubes


def order_corners_world(corners_world):
    centroid = np.mean(corners_world, axis=0)
    angles = np.arctan2(corners_world[:, 1] - centroid[1], corners_world[:, 0] - centroid[0])
    return corners_world[np.argsort(angles)]


def smooth_corners(new_world, state):
    if new_world is None:
        return None
    ordered = order_corners_world(new_world)
    prev = state.get("corners")
    if prev is None:
        state["corners"] = ordered
        return ordered

    # Try all 4 cyclic rotations of corner ordering to find the best alignment
    # with the previous frame — avoids EMA artifacts from label permutations.
    best, best_cost = ordered, np.inf
    for shift in range(4):
        candidate = np.roll(ordered, shift, axis=0)
        cost = float(np.sum((candidate - prev) ** 2))
        if cost < best_cost:
            best_cost, best = cost, candidate

    if best_cost > (3.0 * CUBE_EDGE_M) ** 2:
        state["corners"] = ordered
        return ordered
    alpha = CORNER_SMOOTHING_ALPHA
    state["corners"] = (1.0 - alpha) * prev + alpha * best
    return state["corners"]


def project_point(point_camera, intrinsics):
    x, y, z = point_camera
    if z <= 1e-6:
        return None
    u = float(intrinsics.fx) * x / z + float(intrinsics.ppx)
    v = float(intrinsics.fy) * y / z + float(intrinsics.ppy)
    return u, v


def overlay_mask(image, mask, color, alpha):
    if mask is None:
        return
    overlay = np.zeros_like(image)
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, alpha, image, 1.0, 0.0, dst=image)


def draw_contours(image, mask, color, thickness):
    if mask is None:
        return
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)


def draw_panel(display, lines):
    x0, y0 = 12, 12
    width, height = 580, 26 * len(lines) + 16
    overlay = display.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, display, 0.28, 0.0, dst=display)
    cv2.rectangle(display, (x0, y0), (x0 + width, y0 + height), (220, 220, 220), 1)
    for idx, line in enumerate(lines):
        cv2.putText(
            display, line, (x0 + 14, y0 + 26 + 24 * idx),
            cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255),
            2 if idx == 0 else 1, cv2.LINE_AA,
        )


def create_controls():
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 420, 200)
    cv2.createTrackbar("S min", CONTROLS_WINDOW, RED_HSV_RANGES[0][0][1], 255, lambda v: None)
    cv2.createTrackbar("V min", CONTROLS_WINDOW, RED_HSV_RANGES[0][0][2], 255, lambda v: None)
    cv2.createTrackbar("band below mm", CONTROLS_WINDOW, int(TOP_BAND_BELOW_M * 1000), 24, lambda v: None)
    cv2.createTrackbar("band above mm", CONTROLS_WINDOW, int(TOP_BAND_ABOVE_M * 1000), 24, lambda v: None)


def read_controls(controls_open):
    if not controls_open:
        return {
            "s_min": RED_HSV_RANGES[0][0][1],
            "v_min": RED_HSV_RANGES[0][0][2],
            "band_below_m": TOP_BAND_BELOW_M,
            "band_above_m": TOP_BAND_ABOVE_M,
        }
    return {
        "s_min": cv2.getTrackbarPos("S min", CONTROLS_WINDOW),
        "v_min": cv2.getTrackbarPos("V min", CONTROLS_WINDOW),
        "band_below_m": max(1, cv2.getTrackbarPos("band below mm", CONTROLS_WINDOW)) / 1000.0,
        "band_above_m": max(1, cv2.getTrackbarPos("band above mm", CONTROLS_WINDOW)) / 1000.0,
    }


def main():
    t_wc = load_t_wc()
    if t_wc is None:
        return
    rotation = t_wc[:3, :3]
    translation = t_wc[:3, 3]
    up_cam = world_up_in_camera(t_wc)

    pipe, align, depth_scale = start_realsense()
    depth_filters = create_depth_filters()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW_NAME, cv2.WINDOW_NORMAL)
    controls_open = True
    create_controls()

    snap_to_known = True
    mask_source_idx = 0
    corner_state = {"corners": None}

    try:
        while True:
            frames = align.process(pipe.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_frame = apply_depth_filters(depth_frame, depth_filters)
            intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_m = depth_image.astype(np.float32) * float(depth_scale)

            params = read_controls(controls_open)
            hsv_mask = make_hsv_mask(color_image, params["s_min"], params["v_min"])
            hsv_mask = keep_largest_component(hsv_mask, MIN_COMPONENT_AREA_PX)

            top_mask, top_points, plane, info = detect_top_face(
                hsv_mask, depth_m, intrinsics, t_wc, up_cam, params
            )

            fitted_mask = None
            center_world = None
            edge_len = None
            corners_cam = None

            if plane is not None and top_points is not None:
                corners_cam, edge_len = fit_square_corners(plane, top_points, up_cam, snap_to_known)
                if corners_cam is not None:
                    corners_world = (rotation @ corners_cam.T).T + translation
                    corners_world = smooth_corners(corners_world, corner_state)
                    corners_cam = (rotation.T @ (corners_world - translation).T).T

                    pixel_corners, ok = [], True
                    for corner in corners_cam:
                        px = project_point(corner, intrinsics)
                        if px is None:
                            ok = False
                            break
                        pixel_corners.append(px)
                    if ok:
                        poly = np.array(pixel_corners, dtype=np.int32)
                        fitted_mask = np.zeros(hsv_mask.shape, dtype=np.uint8)
                        cv2.fillConvexPoly(fitted_mask, poly, 255)
                        center_world = np.mean(corners_world, axis=0)
            else:
                corner_state["corners"] = None

            source = MASK_SOURCES[mask_source_idx]
            if source == "fitted-square":
                final_mask = fitted_mask
            elif source == "top-pixels":
                final_mask = top_mask
            else:
                final_mask = cv2.bitwise_and(fitted_mask, hsv_mask) if fitted_mask is not None else top_mask

            display = color_image.copy()
            overlay_mask(display, hsv_mask, (0, 0, 160), 0.12)
            overlay_mask(display, final_mask, (0, 255, 0), 0.45)
            draw_contours(display, final_mask, (0, 255, 0), 2)
            if center_world is not None:
                center_cam = rotation.T @ (center_world - translation)
                cpx = project_point(center_cam, intrinsics)
                if cpx is not None:
                    cpx_i = (int(round(cpx[0])), int(round(cpx[1])))
                    cv2.drawMarker(display, cpx_i, (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
                    cv2.circle(display, cpx_i, 4, (0, 255, 0), -1)

            final_px = 0 if final_mask is None else int(cv2.countNonZero(final_mask))
            z_table = info.get("z_table")
            z_top = info.get("z_top")
            lines = [
                f"{COLOR_WIDTH}x{COLOR_HEIGHT} | source={source} | snap_known={snap_to_known}",
                f"red_px={info.get('red_px', 0)} top_px={info.get('top_px', 0)} final_px={final_px}",
                (f"z_table={z_table*100:.1f}cm z_top={z_top*100:.1f}cm"
                 if z_table is not None else "z_table=?? (no table depth)"),
            ]
            if edge_len is not None:
                lines.append(f"measured_edge={edge_len*100:.2f}cm (cube=2.40cm)")
            if center_world is not None:
                cm = center_world * 100.0
                lines.append(f"top_center_world=({cm[0]:+.1f}, {cm[1]:+.1f}, {cm[2]:+.1f}) cm")
            else:
                lines.append("top_center_world: none")
            lines.append("keys: q quit | k snap | m source | h hsv | s save")
            draw_panel(display, lines)

            show_mask = final_mask if final_mask is not None else np.zeros(hsv_mask.shape, np.uint8)
            cv2.imshow(WINDOW_NAME, display)
            cv2.imshow(MASK_WINDOW_NAME, show_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("k"):
                snap_to_known = not snap_to_known
            if key == ord("m"):
                mask_source_idx = (mask_source_idx + 1) % len(MASK_SOURCES)
            if key == ord("h"):
                controls_open = not controls_open
                if controls_open:
                    create_controls()
                else:
                    cv2.destroyWindow(CONTROLS_WINDOW)
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"top_face_color_{stamp}.png", color_image)
                cv2.imwrite(f"top_face_mask_{stamp}.png", show_mask)
                print(f"Saved top_face_color_{stamp}.png and top_face_mask_{stamp}.png")

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
