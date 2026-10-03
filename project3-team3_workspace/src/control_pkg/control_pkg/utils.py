import numpy as np
import time

q_min_deg = np.array([-160, -70, -170, -113, -170, -115, -180])
q_max_deg = np.array([160, 115, 170, 75, 170, 115, 180])
q_min_rad = np.deg2rad(q_min_deg)
q_max_rad = np.deg2rad(q_max_deg)
qdot_min_rad = np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2])
qdot_max_rad = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2])

def RotX(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.eye(3, 3)
    R[1, 1] = ct
    R[1, 2] = -st
    R[2, 1] = st
    R[2, 2] = ct
    return R

def RotY(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.eye(3, 3)
    R[0, 0] = ct
    R[0, 2] = st
    R[2, 0] = -st
    R[2, 2] = ct
    return R

def RotZ(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.eye(3, 3)
    R[0, 0] = ct
    R[0, 1] = -st
    R[1, 0] = st
    R[1, 1] = ct
    return R

def homogeneous(R, p = np.zeros((3, 1))):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = p.reshape(-1, 1)

    return T

def hom_inv(T):
    R = T[:3, :3]
    t = T[:3, 3:]

    Tn = np.eye(4)
    Tn[:3, :3] = R.T
    Tn[:3, 3:] = -R.T @ t

    return Tn

def vec3_to_skew_symmetric_matrix(v):
    v_hat = np.zeros((3, 3))
    v_hat[0, 1] = -v[2]
    v_hat[0, 2] = v[1]
    v_hat[1, 0] = v[2]
    v_hat[1, 2] = -v[0]
    v_hat[2, 0] = -v[1]
    v_hat[2, 1] = v[0]

    return v_hat

def skew_symmetric_matrix_to_vec3(v_hat):
    v = np.array([v_hat[2, 1], v_hat[0, 2], v_hat[1, 0]])
    return v

def vec6_to_skew_symmetric_matrix(V):
    w = V[0:3]
    v = V[3:6]
    w_hat = vec3_to_skew_symmetric_matrix(w)
    V_hat = np.zeros((4, 4))
    V_hat[:3, :3] = w_hat
    V_hat[:3, 3] = v

    return V_hat

def skew_symmetric_matrix_to_vec6(V_hat):
    w_hat = V_hat[:3, :3]
    v = V_hat[:3, 3]
    w = skew_symmetric_matrix_to_vec3(w_hat)

    return np.concatenate((w, v))

def adjoint_matrix(T):
    R = T[:3, :3]
    p = T[:3, 3]
    p_hat = vec3_to_skew_symmetric_matrix(p)
    adjoint = np.zeros((6, 6))
    adjoint[:3, :3] = R
    adjoint[3:, 3:] = R
    adjoint[3:, :3] = p_hat @ R

    return adjoint

def matrix_exponential_3(v_hat):
    v = skew_symmetric_matrix_to_vec3(v_hat)
    q = np.linalg.norm(v)
    w_hat = v_hat / q
    R = np.eye(3) + np.sin(q) * w_hat + (1 - np.cos(q)) * (w_hat @ w_hat)

    return R

def matrix_exponential_6(V_hat):
    w_hatq = V_hat[:3, :3]
    V = skew_symmetric_matrix_to_vec6(V_hat)
    wq = V[0:3]
    vq = V[3:6]

    q = np.linalg.norm(wq)

    if q < 1e-6:
        R = np.eye(3)
        p = vq
        return homogeneous(R, p)
    else:
        w_hat = w_hatq / q
        v = vq / q
        R = matrix_exponential_3(w_hatq)
        p = (np.eye(3) * q + (1 - np.cos(q)) * w_hat + (q - np.sin(q)) * (w_hat @ w_hat)) @ v
        return homogeneous(R, p)

def geometric_jacobians_world(S, q):
    n = S.shape[1]
    J_W = S.copy()
    T = np.eye(4)

    for i in range(1, n):
        S_hatq_i = vec6_to_skew_symmetric_matrix(S[:, i-1] * q[i-1])
        T = T @ matrix_exponential_6(S_hatq_i)
        J_W[:, i] = (adjoint_matrix(T) @ S[:, i])

    return J_W

def fk_poe_world(M, S, q):
    n = S.shape[1]
    T = np.array(M)
    for i in reversed(range(n)):
        S_hatq_i = vec6_to_skew_symmetric_matrix(S[:, i] * q[i])
        T = matrix_exponential_6(S_hatq_i) @ T

    return T

def position_task(p, p_des, dt):
    p = np.asarray(p, dtype=float).reshape(3, 1)
    p_des = np.asarray(p_des, dtype=float).reshape(3, 1)
    position_error = np.linalg.norm(p_des - p)
    t_pos = (p_des - p) / dt
    return t_pos, position_error

def z_axis_task(z_ee, z_des, dt):
    z_ee_vec = z_ee.reshape(3)
    z_des_vec = z_des.reshape(3)
    dot = np.clip(float(z_ee_vec @ z_des_vec), -1.0, 1.0)
    angle_error = np.arccos(dot)

    if angle_error < 1e-9:
        return np.zeros((3, 1)), 0.0

    rotation_axis = np.cross(z_ee_vec, z_des_vec)
    axis_norm = np.linalg.norm(rotation_axis)
    if axis_norm < 1e-9:
        reference = np.array([1.0, 0.0, 0.0])
        if abs(z_ee_vec @ reference) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        rotation_axis = np.cross(z_ee_vec, reference)
        axis_norm = np.linalg.norm(rotation_axis)

    rotation_axis = rotation_axis / axis_norm
    omega_des = (angle_error / dt) * rotation_axis
    t_z = np.cross(omega_des, z_ee_vec).reshape(3, 1)
    return t_z, angle_error

def rank_initial_guesses(
    p_des,
    sample_count=256,
    try_count=16,
    seed=None,
    z_des=np.array([0.0, 0.0, -1.0]),
    z_axis_rank_weight=10.0,
    q_min=q_min_rad,
    q_max=q_max_rad,
):
    p_des = np.asarray(p_des, dtype=float).reshape(3, 1)
    z_des = np.asarray(z_des, dtype=float).reshape(3, 1)
    z_des = z_des / np.linalg.norm(z_des)

    rng = np.random.default_rng(seed)
    q_samples = rng.uniform(q_min, q_max, size=(sample_count, len(q_min)))
    q_samples = np.vstack([np.zeros(len(q_min)), q_samples])

    scores = []
    for q in q_samples:
        T = fk_poe_world(M, S, q)
        p = T[:3, 3].reshape(3, 1)
        z_ee = T[:3, 2].reshape(3, 1)
        z_ee = z_ee / np.linalg.norm(z_ee)

        position_error = np.linalg.norm(p_des - p)
        z_axis_error = np.arccos(np.clip((z_des.reshape(3) @ z_ee.reshape(3)), -1.0, 1.0))
        scores.append(position_error + z_axis_rank_weight * z_axis_error)

    best_indices = np.argsort(scores)[:try_count]
    return q_samples[best_indices]

base_length = 14.6
p_M = np.array([[0.0, 0.0, 47.88 + base_length]]).T
M = homogeneous(np.eye(3), p_M)

S = np.array([
    [0., 0., 1., 0., 0., 0.],
    [0., 1., 0., -16.95, 0., 0.],
    [0., 0., 1., 0., 0., 0.],
    [0., -1., 0., 28.5, 0., 0.],
    [0., 0., 1., 0., 0., 0.],
    [0., -1., 0., 41.283, 0., 0.],
    [0., 0., 1., 0., 0., 0.],
]).T
