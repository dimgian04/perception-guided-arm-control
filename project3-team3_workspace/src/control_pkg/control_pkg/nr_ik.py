import numpy as np


TOL = 1e-7
N = 7

QP_HOME_Z_M = 47.88 / 100.0
TOOL_LENGTH_M = 14.6 / 100.0

S1 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
S2 = np.array([0.0, 1.0, 0.0, -0.1695, 0.0, 0.0])
S3 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
S4 = np.array([0.0, -1.0, 0.0, 0.285, 0.0, 0.0])
S5 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
S6 = np.array([0.0, -1.0, 0.0, 0.41283, 0.0, 0.0])
S7 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
SPACE_SCREWS = [S1, S2, S3, S4, S5, S6, S7]

T_SPACE_BODY_HOME = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, QP_HOME_Z_M + TOOL_LENGTH_M],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

Q_MIN = np.deg2rad([-160, -70, -170, -113, -170, -115, -180])
Q_MAX = np.deg2rad([160, 115, 170, 75, 170, 115, 180])
T_WORLD_SPACE = np.eye(4)


def skew_mat(v):
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def hat(tau):
    omega = tau[:3]
    v = tau[3:]
    return np.block([[skew_mat(omega), v.reshape(-1, 1)], [np.zeros((1, 4))]])


def exp(tau_hat):
    rho = tau_hat[:3, 3]
    v = np.array([tau_hat[2, 1], tau_hat[0, 2], tau_hat[1, 0]])
    theta = np.linalg.norm(v)
    u = v / theta if abs(theta) >= TOL else np.array([0.0, 0.0, 1.0])
    u_hat = skew_mat(u)
    exp_theta = np.eye(3) + np.sin(theta) * u_hat + (1.0 - np.cos(theta)) * u_hat @ u_hat
    if abs(theta) >= TOL:
        v_matrix = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta * u_hat
            + (theta - np.sin(theta)) / theta * u_hat @ u_hat
        )
    else:
        v_matrix = np.eye(3)
    return np.block([[exp_theta, v_matrix @ rho.reshape(-1, 1)], [np.zeros((1, 3)), 1.0]])


def log_se3(transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    theta = np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    if theta < TOL:
        omega = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / 2.0,
                (rotation[0, 2] - rotation[2, 0]) / 2.0,
                (rotation[1, 0] - rotation[0, 1]) / 2.0,
            ]
        )
        v_inv = np.eye(3)
    else:
        omega_hat = theta * (rotation - rotation.T) / (2.0 * np.sin(theta))
        omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])
        v_matrix = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta**2 * omega_hat
            + (theta - np.sin(theta)) / theta**3 * (omega_hat @ omega_hat)
        )
        v_inv = np.linalg.inv(v_matrix)
    rho = v_inv @ translation
    return np.block([omega, rho])


def adjoint(transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return np.block(
        [
            [rotation, np.zeros((3, 3))],
            [skew_mat(translation) @ rotation, rotation],
        ]
    )


def poe_fk(q, frame_ref="space"):
    q = np.clip(np.asarray(q, dtype=float).reshape(N), Q_MIN, Q_MAX)
    transform = T_SPACE_BODY_HOME.copy()
    for index in range(N - 1, -1, -1):
        transform = exp(hat(SPACE_SCREWS[index]) * q[index]) @ transform

    if frame_ref == "world":
        return T_WORLD_SPACE @ transform
    if frame_ref == "space":
        return transform
    raise ValueError(f"Unsupported frame_ref: {frame_ref}")


def jacobian_space(q, frame_ref="space"):
    q = np.clip(np.asarray(q, dtype=float).reshape(N), Q_MIN, Q_MAX)
    jac_adjoints = [np.eye(6)]
    for index in range(N - 1):
        jac_adjoints.append(
            jac_adjoints[-1] @ adjoint(exp(hat(SPACE_SCREWS[index]) * q[index]))
        )

    jac = np.zeros((6, N))
    for index, jac_adjoint in enumerate(jac_adjoints):
        jac[:, index] = jac_adjoint @ SPACE_SCREWS[index]

    if frame_ref == "world":
        return adjoint(T_WORLD_SPACE) @ jac
    if frame_ref == "space":
        return jac
    raise ValueError(f"Unsupported frame_ref: {frame_ref}")


def poe_ik_from_q0(q0, target_transform, frame_ref="space", max_iter=100, error_tol=1e-5):
    q = np.clip(np.asarray(q0, dtype=float).reshape(N), Q_MIN, Q_MAX)
    target_transform = np.asarray(target_transform, dtype=float).reshape(4, 4)

    last_error_norm = np.inf
    for iteration in range(max_iter):
        current_transform = poe_fk(q, frame_ref=frame_ref)
        error = log_se3(target_transform @ np.linalg.inv(current_transform))
        last_error_norm = float(np.linalg.norm(error))
        if last_error_norm < error_tol:
            return q, True, iteration, last_error_norm

        dq = np.linalg.pinv(jacobian_space(q, frame_ref=frame_ref)) @ error
        q = np.clip(q + dq, Q_MIN, Q_MAX)

    return q, False, max_iter, last_error_norm
