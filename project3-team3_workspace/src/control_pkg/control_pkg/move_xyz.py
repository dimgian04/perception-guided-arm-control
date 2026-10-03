import argparse
import json
import math
import os
import socket
import time

import numpy as np

from nr_ik import Q_MAX, Q_MIN, jacobian_space, poe_fk, skew_mat


JOINT_COUNT = 7
DEFAULT_HOST = "192.168.0.103"
DEFAULT_PORT = 5017
DEFAULT_PREGRASP_TARGET_FILE = "~/.ros/myarm_last_pregrasp_target.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone IK test: read current myArm angles, move the "
            "end-effector pose in world X/Y/Z, solve with Jacobian "
            "pseudo-inverse IK from q0, and send the result."
        )
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--move-x-cm", type=float, default=0.0)
    parser.add_argument("--move-y-cm", type=float, default=0.0)
    parser.add_argument("--move-z-cm", type=float, default=0.0)
    parser.add_argument(
        "--pregrasp-target-file",
        default=DEFAULT_PREGRASP_TARGET_FILE,
    )
    parser.add_argument("--error-tol", type=float, default=1e-5)
    parser.add_argument("--position-tol-m", type=float, default=0.0001)
    parser.add_argument("--z-axis-tol-deg", type=float, default=0.1)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--max-step-deg", type=float, default=3.0)
    parser.add_argument("--joint-center-gain", type=float, default=0.0)
    parser.add_argument(
        "--current-deg",
        nargs=JOINT_COUNT,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    parser.add_argument("--max-joint-delta-deg", type=float, default=45.0)
    parser.add_argument("--allow-large-joint-delta", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--hold-orientation",
        action="store_true",
        help=(
            "Target the CURRENT tool z-axis instead of the saved pregrasp z-axis. "
            "Use for XY visual-servoing nudges to avoid rotating the wrist."
        ),
    )
    return parser.parse_args()


def read_line(sock):
    chunks = []
    while True:
        chunk = sock.recv(1)
        if not chunk:
            break
        if chunk == b"\n":
            break
        chunks.append(chunk)
    if not chunks:
        raise RuntimeError("Empty TCP response")
    return b"".join(chunks).decode("utf-8")


def tcp_request(host, port, command, timeout):
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall((json.dumps(command) + "\n").encode("utf-8"))
        return json.loads(read_line(sock))


def validate_joint_count(angles):
    if len(angles) != JOINT_COUNT:
        raise RuntimeError(f"Expected {JOINT_COUNT} joints, got {len(angles)}")


def extract_angles_deg(response):
    for key in ("angles_deg", "current_angles_deg", "current_deg", "q_deg"):
        if key in response:
            angles = [float(value) for value in response[key]]
            validate_joint_count(angles)
            return angles

    for key in ("angles_rad", "current_angles_rad", "current_rad", "q_rad"):
        if key in response:
            angles = [math.degrees(float(value)) for value in response[key]]
            validate_joint_count(angles)
            return angles

    raise RuntimeError(f"TCP response does not contain joint angles: {response}")


def request_current_angles_deg(host, port, timeout):
    command = {
        "type": "get_angles",
        "seq": int(time.time() * 1000),
    }
    response = tcp_request(host, port, command, timeout)
    if not response.get("ok", False):
        raise RuntimeError(f"get_angles rejected by TCP server: {response}")
    return extract_angles_deg(response)


def send_joint_target_deg(host, port, angles_deg, speed, timeout):
    command = {
        "type": "joint_target",
        "seq": int(time.time() * 1000),
        "angles_deg": [float(angle) for angle in angles_deg],
        "speed": int(speed),
    }
    return tcp_request(host, port, command, timeout)


def get_q0_deg(args):
    if args.current_deg is not None:
        angles = [float(value) for value in args.current_deg]
        validate_joint_count(angles)
        return angles
    return request_current_angles_deg(args.host, args.port, args.timeout)


def load_pregrasp_rotation(path):
    expanded_path = os.path.expanduser(path)
    with open(expanded_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    rotation = np.asarray(data.get("pregrasp_rotation_world"), dtype=float)
    if rotation.shape != (3, 3):
        raise RuntimeError(
            f"Bad or missing pregrasp_rotation_world in {expanded_path}; "
            f"expected 3x3, got {rotation.shape}"
        )

    return rotation, expanded_path


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise RuntimeError(f"Cannot normalize near-zero vector: {vector}")
    return vector / norm


def z_axis_angle(current_z, desired_z):
    current_z = normalize(current_z)
    desired_z = normalize(desired_z)
    return float(np.arccos(np.clip(float(current_z @ desired_z), -1.0, 1.0)))


def poe_ik_position_z_axis_from_q0(
    q0,
    target_position,
    target_z_axis,
    max_iter=1000,
    error_tol=1e-5,
    position_tol_m=0.002,
    z_axis_tol_rad=np.deg2rad(2.0),
    damping=0.05,
    max_step_rad=np.deg2rad(3.0),
    joint_center_gain=0.01,
):
    q = np.clip(np.asarray(q0, dtype=float).reshape(JOINT_COUNT), Q_MIN, Q_MAX)
    target_position = np.asarray(target_position, dtype=float).reshape(3)
    target_z_axis = normalize(target_z_axis)
    q_mid = 0.5 * (Q_MIN + Q_MAX)

    last_error_norm = np.inf
    for iteration in range(max_iter):
        current_transform = poe_fk(q, frame_ref="space")
        current_position = current_transform[:3, 3]
        current_z_axis = normalize(current_transform[:3, 2])

        position_error = target_position - current_position
        z_axis_error = target_z_axis - current_z_axis
        z_angle = z_axis_angle(current_z_axis, target_z_axis)
        last_error_norm = float(np.sqrt(np.linalg.norm(position_error) ** 2 + z_angle**2))

        if (
            np.linalg.norm(position_error) <= position_tol_m
            and z_angle <= z_axis_tol_rad
        ) or last_error_norm <= error_tol:
            return q, True, iteration, last_error_norm

        jacobian = jacobian_space(q, frame_ref="space")
        angular_jacobian = jacobian[:3, :]
        linear_twist_jacobian = jacobian[3:, :]

        # Convert space-twist linear component to TCP origin velocity:
        # p_dot = omega x p + v = -skew(p) @ omega + v
        position_jacobian = -skew_mat(current_position) @ angular_jacobian + linear_twist_jacobian

        # z_dot = omega x z = -skew(z) @ omega. Rank 2: yaw about tool z stays free,
        # so the wrist doesn't spin to match a yaw that the QP pre-grasp already set.
        z_axis_jacobian = -skew_mat(current_z_axis) @ angular_jacobian

        task_jacobian = np.vstack((position_jacobian, z_axis_jacobian))
        task_error = np.concatenate((position_error, z_axis_error))

        jj_t = task_jacobian @ task_jacobian.T
        damping_matrix = float(damping) ** 2 * np.eye(jj_t.shape[0])
        task_jacobian_pinv = task_jacobian.T @ np.linalg.inv(jj_t + damping_matrix)
        dq = task_jacobian_pinv @ task_error

        if joint_center_gain > 0.0:
            nullspace = np.eye(JOINT_COUNT) - task_jacobian_pinv @ task_jacobian
            dq += nullspace @ (-float(joint_center_gain) * (q - q_mid))

        max_abs_step = float(np.max(np.abs(dq)))
        if max_step_rad > 0.0 and max_abs_step > max_step_rad:
            dq *= float(max_step_rad) / max_abs_step

        q = np.clip(q + dq, Q_MIN, Q_MAX)

    return q, False, max_iter, last_error_norm


def solve_xyz(
    q0_rad,
    move_x_cm,
    move_y_cm,
    move_z_cm,
    max_iters,
    error_tol,
    position_tol_m,
    z_axis_tol_deg,
    damping,
    max_step_deg,
    joint_center_gain,
    pregrasp_rotation,
    hold_orientation=False,
):
    current_transform = poe_fk(q0_rad, frame_ref="space")
    # hold_orientation keeps the current wrist angle during XY nudges so a pure
    # XY move doesn't also rotate the wrist back to the saved pre-grasp pose.
    if hold_orientation or pregrasp_rotation is None:
        target_rotation = current_transform[:3, :3]
    else:
        target_rotation = pregrasp_rotation
    target_transform = np.eye(4)
    target_transform[:3, :3] = target_rotation
    target_transform[:3, 3] = current_transform[:3, 3]
    target_transform[0, 3] += float(move_x_cm) / 100.0
    target_transform[1, 3] += float(move_y_cm) / 100.0
    target_transform[2, 3] += float(move_z_cm) / 100.0

    q_sol, success, iterations, error_norm = poe_ik_position_z_axis_from_q0(
        q0=q0_rad,
        target_position=target_transform[:3, 3],
        target_z_axis=target_transform[:3, 2],
        max_iter=max_iters,
        error_tol=error_tol,
        position_tol_m=position_tol_m,
        z_axis_tol_rad=np.deg2rad(z_axis_tol_deg),
        damping=damping,
        max_step_rad=np.deg2rad(max_step_deg),
        joint_center_gain=joint_center_gain,
    )
    solved_transform = poe_fk(q_sol, frame_ref="space")
    return q_sol, success, iterations, error_norm, current_transform, target_transform, solved_transform


def print_report(q0_deg, q_sol, success, iterations, error_norm, current_transform, target_transform, solved_transform):
    q_sol_deg = np.rad2deg(q_sol)
    delta_deg = q_sol_deg - np.asarray(q0_deg, dtype=float)
    position_error = target_transform[:3, 3] - solved_transform[:3, 3]
    target_z_axis = normalize(target_transform[:3, 2])
    solved_z_axis = normalize(solved_transform[:3, 2])
    z_axis_error = target_z_axis - solved_z_axis
    z_axis_angle_deg = np.rad2deg(z_axis_angle(solved_z_axis, target_z_axis))
    lower_margin_deg = np.rad2deg(q_sol - Q_MIN)
    upper_margin_deg = np.rad2deg(Q_MAX - q_sol)

    print("\nCurrent q deg:")
    print(np.array2string(np.asarray(q0_deg), precision=3, suppress_small=False))
    print("Current EE position m:")
    print(np.array2string(current_transform[:3, 3], precision=4, suppress_small=False))
    print("Target EE position m:")
    print(np.array2string(target_transform[:3, 3], precision=4, suppress_small=False))
    print("Solved EE position m:")
    print(np.array2string(solved_transform[:3, 3], precision=4, suppress_small=False))

    print("\nIK result:")
    print(f"success: {success}")
    print(f"iterations: {iterations}")
    print(f"solver_task_error_norm: {error_norm:.6g}")
    print("Target-solved position error m:")
    print(np.array2string(position_error, precision=6, suppress_small=False))
    print("Target z-axis:")
    print(np.array2string(target_z_axis, precision=6, suppress_small=False))
    print("Solved z-axis:")
    print(np.array2string(solved_z_axis, precision=6, suppress_small=False))
    print("Target-solved z-axis error:")
    print(np.array2string(z_axis_error, precision=6, suppress_small=False))
    print(f"z_axis_angle_error_deg: {z_axis_angle_deg:.4f}")

    print("\nSolved q deg:")
    print(np.array2string(q_sol_deg, precision=3, suppress_small=False))
    print("Joint delta deg:")
    print(np.array2string(delta_deg, precision=3, suppress_small=False))
    print(f"max_abs_joint_delta_deg: {float(np.max(np.abs(delta_deg))):.3f}")
    print("Distance to lower joint limits deg:")
    print(np.array2string(lower_margin_deg, precision=2, suppress_small=False))
    print("Distance to upper joint limits deg:")
    print(np.array2string(upper_margin_deg, precision=2, suppress_small=False))
    return q_sol_deg, delta_deg


def print_jacobian_report(q0_rad):
    jacobian = jacobian_space(q0_rad, frame_ref="space")
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 1e-12 else np.inf

    print("\nCurrent full Jacobian singular values:")
    print(np.array2string(singular_values, precision=5, suppress_small=False))
    print(f"Current full Jacobian condition number: {condition:.5g}")

    for axis_name, twist in (
        ("+X", np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0])),
        ("+Y", np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.0])),
        ("+Z", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01])),
    ):
        dq = np.linalg.pinv(jacobian) @ twist
        print(
            f"Instant {axis_name} 1cm equivalent joint step deg: "
            f"max={float(np.max(np.abs(np.rad2deg(dq)))):.3f}, "
            f"dq={np.array2string(np.rad2deg(dq), precision=3, suppress_small=False)}"
        )


def main():
    args = parse_args()
    q0_deg = get_q0_deg(args)
    q0_rad = np.clip(np.deg2rad(q0_deg), Q_MIN, Q_MAX)
    print_jacobian_report(q0_rad)
    if args.hold_orientation:
        pregrasp_rotation = None
        current_z_axis = poe_fk(q0_rad, frame_ref="space")[:3, 2]
        print("\nHold-orientation mode: targeting the CURRENT tool z-axis.")
        print("Current tool z-axis:")
        print(np.array2string(current_z_axis, precision=5, suppress_small=False))
    else:
        pregrasp_rotation, pregrasp_file = load_pregrasp_rotation(args.pregrasp_target_file)
        print(f"\nLoaded target rotation from: {pregrasp_file}")
        print("Target rotation matrix:")
        print(np.array2string(pregrasp_rotation, precision=5, suppress_small=False))
        print("Target z-axis:")
        print(np.array2string(pregrasp_rotation[:3, 2], precision=5, suppress_small=False))

    q_sol, success, iterations, error_norm, current_transform, target_transform, solved_transform = solve_xyz(
        q0_rad=q0_rad,
        move_x_cm=args.move_x_cm,
        move_y_cm=args.move_y_cm,
        move_z_cm=args.move_z_cm,
        max_iters=int(args.max_iters),
        error_tol=float(args.error_tol),
        position_tol_m=float(args.position_tol_m),
        z_axis_tol_deg=float(args.z_axis_tol_deg),
        damping=float(args.damping),
        max_step_deg=float(args.max_step_deg),
        joint_center_gain=float(args.joint_center_gain),
        pregrasp_rotation=pregrasp_rotation,
        hold_orientation=bool(args.hold_orientation),
    )
    q_sol_deg, delta_deg = print_report(
        q0_deg,
        q_sol,
        success,
        iterations,
        error_norm,
        current_transform,
        target_transform,
        solved_transform,
    )

    if not success:
        print("\nABORT: IK did not converge. No target was sent.")
        return

    max_delta = float(np.max(np.abs(delta_deg)))
    if max_delta > float(args.max_joint_delta_deg) and not args.allow_large_joint_delta:
        print(
            "\nABORT: IK solution asks for a large joint movement "
            f"({max_delta:.2f} deg > {args.max_joint_delta_deg:.2f} deg)."
        )
        print("No target was sent.")
        return

    if args.dry_run:
        print("\nDRY RUN: target was not sent.")
        return

    response = send_joint_target_deg(args.host, args.port, q_sol_deg, args.speed, args.timeout)
    print("\nTCP send response:")
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
