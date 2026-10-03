from dataclasses import dataclass

import numpy as np
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix
from utils import (
    M,
    S,
    fk_poe_world,
    geometric_jacobians_world,
    position_task,
    q_max_rad,
    q_min_rad,
    qdot_max_rad,
    qdot_min_rad,
    rank_initial_guesses,
    vec3_to_skew_symmetric_matrix,
    z_axis_task,
)


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int
    position_error: float
    z_axis_error: float
    message: str


DEFAULT_JOINT_LIMIT_MARGIN_DEG = 15.0


def apply_joint_limit_margin(q_min, q_max, margin_deg):
    q_min = np.asarray(q_min, dtype=float).reshape(-1)
    q_max = np.asarray(q_max, dtype=float).reshape(-1)
    margin = np.deg2rad(float(margin_deg))
    if margin <= 0.0:
        return q_min.copy(), q_max.copy()

    q_min_safe = q_min + margin
    q_max_safe = q_max - margin
    if np.any(q_min_safe >= q_max_safe):
        raise ValueError(
            f"Joint limit margin {margin_deg} deg is too large for these limits."
        )
    return q_min_safe, q_max_safe


def joint_limit_penalty(q, q_min=q_min_rad, q_max=q_max_rad):
    q = np.asarray(q, dtype=float).reshape(-1)
    q_min = np.asarray(q_min, dtype=float).reshape(-1)
    q_max = np.asarray(q_max, dtype=float).reshape(-1)
    distance_to_limit = np.minimum(q - q_min, q_max - q)
    half_range = 0.5 * (q_max - q_min)
    normalized_margin = np.clip(distance_to_limit / half_range, 0.0, 1.0)
    return float(np.mean((1.0 - normalized_margin) ** 2))


def min_joint_margin_deg(q, q_min=q_min_rad, q_max=q_max_rad):
    q = np.asarray(q, dtype=float).reshape(-1)
    q_min = np.asarray(q_min, dtype=float).reshape(-1)
    q_max = np.asarray(q_max, dtype=float).reshape(-1)
    distance_to_limit = np.minimum(q - q_min, q_max - q)
    return float(np.rad2deg(np.min(distance_to_limit)))


def solution_score(
    position_error,
    z_axis_error,
    z_axis_weight,
    q=None,
    joint_limit_weight=0.0,
    q_min=q_min_rad,
    q_max=q_max_rad,
):
    score = position_error + z_axis_weight * z_axis_error
    if q is not None and joint_limit_weight > 0.0:
        score += joint_limit_weight * joint_limit_penalty(q, q_min=q_min, q_max=q_max)
    return score


def build_joint_limit_constraints(
    q,
    dt,
    q_min=q_min_rad,
    q_max=q_max_rad,
    qdot_min=qdot_min_rad,
    qdot_max=qdot_max_rad,
):
    q = np.asarray(q, dtype=float).reshape(-1)
    q_min = np.asarray(q_min, dtype=float).reshape(-1)
    q_max = np.asarray(q_max, dtype=float).reshape(-1)
    qdot_min = np.asarray(qdot_min, dtype=float).reshape(-1)
    qdot_max = np.asarray(qdot_max, dtype=float).reshape(-1)

    n = q.size
    eye = np.eye(n)

    C = np.vstack([
        -dt * eye,
        dt * eye,
        -eye,
        eye,
    ])
    d = np.vstack([
        (q - q_min).reshape(-1, 1),
        (q_max - q).reshape(-1, 1),
        (-qdot_min).reshape(-1, 1),
        qdot_max.reshape(-1, 1),
    ])

    return C, d


def build_qp(
    q,
    p_des,
    dt=1.0,
    z_des=np.array([0.0, 0.0, -1.0]),
    position_weight=1.0,
    z_axis_weight=1.0,
    damping=1e-3,
    q_min=q_min_rad,
    q_max=q_max_rad,
    qdot_min=qdot_min_rad,
    qdot_max=qdot_max_rad,
):
    q = np.asarray(q, dtype=float).reshape(-1)
    p_des = np.asarray(p_des, dtype=float).reshape(3, 1)
    z_des = np.asarray(z_des, dtype=float).reshape(3, 1)
    z_des = z_des / np.linalg.norm(z_des)

    T = fk_poe_world(M, S, q)
    p = T[:3, 3].reshape(3, 1)
    R = T[:3, :3]
    z_ee = R[:, 2].reshape(3, 1)
    z_ee = z_ee / np.linalg.norm(z_ee)

    J = geometric_jacobians_world(S, q)
    Jw = J[:3, :]
    Jp = J[3:, :]

    # z_dot = -skew(z_ee) @ omega: Jacobian of the tool z-axis w.r.t. joint velocities.
    Jz = -vec3_to_skew_symmetric_matrix(z_ee.reshape(3)) @ Jw

    t_pos, position_error = position_task(p, p_des, dt)
    t_z, z_axis_error = z_axis_task(z_ee, z_des, dt)

    n = q.size
    W = np.vstack(
        [
            position_weight * Jp,
            z_axis_weight * Jz,
            damping * np.eye(n),
        ]
    )
    t = np.vstack(
        [
            position_weight * t_pos,
            z_axis_weight * t_z,
            np.zeros((n, 1)),
        ]
    )

    Q = W.T @ W
    Q = 0.5 * (Q + Q.T)
    q_lin = -W.T @ t

    C, d = build_joint_limit_constraints(
        q,
        dt,
        q_min=q_min,
        q_max=q_max,
        qdot_min=qdot_min,
        qdot_max=qdot_max,
    )

    diagnostics = {
        "T": T,
        "p": p.reshape(3),
        "z_ee": z_ee.reshape(3),
        "position_error": position_error,
        "z_axis_error": z_axis_error,
    }
    return Q, q_lin.reshape(-1), C, d.reshape(-1), diagnostics


def solve_ik(
    p_des,
    q0,
    dt=1.0,
    max_iters=100,
    position_tol=1e-3,
    z_axis_tol=1e-3,
    z_des=np.array([0.0, 0.0, -1.0]),
    position_weight=1.0,
    z_axis_weight=1.0,
    damping=1e-3,
    q_min=q_min_rad,
    q_max=q_max_rad,
    qdot_min=qdot_min_rad,
    qdot_max=qdot_max_rad,
    solver="osqp",
):
    q = np.clip(np.asarray(q0, dtype=float).reshape(-1), q_min, q_max)

    last_diag = None
    for it in range(max_iters):
        Q, q_lin, C, d, diag = build_qp(
            q,
            p_des,
            dt=dt,
            z_des=z_des,
            position_weight=position_weight,
            z_axis_weight=z_axis_weight,
            damping=damping,
            q_min=q_min,
            q_max=q_max,
            qdot_min=qdot_min,
            qdot_max=qdot_max,
        )
        last_diag = diag

        if diag["position_error"] <= position_tol and diag["z_axis_error"] <= z_axis_tol:
            return IKResult(
                q=q,
                success=True,
                iterations=it,
                position_error=diag["position_error"],
                z_axis_error=diag["z_axis_error"],
                message="Converged.",
            )

        P = csc_matrix(Q) if solver == "osqp" else Q
        G = csc_matrix(C) if solver == "osqp" else C
        qdot = solve_qp(P=P, q=q_lin, G=G, h=d, solver=solver)
        if qdot is None:
            return IKResult(
                q=q,
                success=False,
                iterations=it,
                position_error=diag["position_error"],
                z_axis_error=diag["z_axis_error"],
                message="QP solver failed.",
            )

        q = q + dt * np.asarray(qdot).reshape(-1)
        q = np.clip(q, q_min, q_max)

    if last_diag is None:
        last_diag = {"position_error": np.inf, "z_axis_error": np.inf}

    return IKResult(
        q=q,
        success=False,
        iterations=max_iters,
        position_error=last_diag["position_error"],
        z_axis_error=last_diag["z_axis_error"],
        message="Maximum iterations reached.",
    )


def solve_ik_multistart(
    p_des,
    sample_count=256,
    try_count=16,
    seed=None,
    z_axis_rank_weight=10.0,
    joint_limit_margin_deg=DEFAULT_JOINT_LIMIT_MARGIN_DEG,
    joint_limit_score_weight=0.5,
    **solve_kwargs,
):
    solve_kwargs = dict(solve_kwargs)
    real_q_min = np.asarray(solve_kwargs.get("q_min", q_min_rad), dtype=float).reshape(-1)
    real_q_max = np.asarray(solve_kwargs.get("q_max", q_max_rad), dtype=float).reshape(-1)
    safe_q_min, safe_q_max = apply_joint_limit_margin(
        real_q_min,
        real_q_max,
        joint_limit_margin_deg,
    )
    solve_kwargs["q_min"] = safe_q_min
    solve_kwargs["q_max"] = safe_q_max

    q0_candidates = rank_initial_guesses(
        p_des,
        sample_count=sample_count,
        try_count=try_count,
        seed=seed,
        z_des=solve_kwargs.get("z_des", np.array([0.0, 0.0, -1.0])),
        z_axis_rank_weight=z_axis_rank_weight,
        q_min=safe_q_min,
        q_max=safe_q_max,
    )

    best_result = None
    best_score = np.inf
    best_success = None
    best_success_score = np.inf
    for q0 in q0_candidates:
        result = solve_ik(p_des=p_des, q0=q0, **solve_kwargs)
        score = solution_score(
            result.position_error,
            result.z_axis_error,
            z_axis_rank_weight,
            q=result.q,
            joint_limit_weight=joint_limit_score_weight,
            q_min=real_q_min,
            q_max=real_q_max,
        )
        if score < best_score:
            best_result = result
            best_score = score
        if result.success:
            if score < best_success_score:
                best_success = result
                best_success_score = score

    selected = best_success if best_success is not None else best_result
    if selected is not None:
        selected.message = (
            f"{selected.message} min_joint_margin="
            f"{min_joint_margin_deg(selected.q, real_q_min, real_q_max):.1f}deg; "
            f"safe_margin={float(joint_limit_margin_deg):.1f}deg."
        )
    return selected
