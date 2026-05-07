"""
Diagnostics for probing the theoretical assumptions.

Each function takes readily available data (agent state, buffer,
true system) and returns a scalar or small array.
"""

import numpy as np
from numpy.linalg import eigvals, eigvalsh, norm


def relative_parameter_error(
    A_est: np.ndarray, B_est: np.ndarray, A_true: np.ndarray, B_true: np.ndarray
) -> dict[str, float]:
    """
    Relative Frobenius error ||Theta_est - Theta_true||_F / ||Theta_true||_F,
    reported jointly and for A, B blocks separately.

    Returns
    -------
    dict with keys 'joint', 'A', 'B'.
    """
    Theta_est = np.hstack([A_est, B_est])
    Theta_true = np.hstack([A_true, B_true])

    joint = norm(Theta_est - Theta_true, "fro") / max(norm(Theta_true, "fro"), 1e-15)
    a_err = norm(A_est - A_true, "fro") / max(norm(A_true, "fro"), 1e-15)
    b_err = norm(B_est - B_true, "fro") / max(norm(B_true, "fro"), 1e-15)

    return {"joint": joint, "A": a_err, "B": b_err}


def support_metrics(
    A_est: np.ndarray,
    B_est: np.ndarray,
    true_supports: list[set],
    threshold: float,  # TODO: tie this to lasso lambda. codex used 5e-2
) -> dict[str, float]:
    """
    Support precision, recall, and F1 averaged over rows, reported
    jointly and for the A and B blocks separately.

    Parameters
    ----------
    A_hat : ndarray (d, d)
    B_hat : ndarray (d, p)
    true_supports : list of sets
        true_supports[i] = set of nonzero column indices in row i
        of [A_star  B_star]. Indices in [0, d) refer to the A block;
        indices in [d, d+p) refer to the B block.
    threshold : float
        Absolute threshold for declaring a coefficient nonzero.

    Returns
    -------
    dict with nested keys for joint, A, and B blocks:
        'joint': {'precision', 'recall', 'f1'},
        'A':     {'precision', 'recall', 'f1'},
        'B':     {'precision', 'recall', 'f1'}.
    """
    Theta_hat = np.hstack([A_est, B_est])
    d = A_est.shape[0]

    def _row_metrics(est_support, true_support):
        tp = len(est_support & true_support)
        precision = tp / max(len(est_support), 1)
        recall = tp / max(len(true_support), 1)
        return precision, recall

    # Joint, A-block, B-block precision/recall lists across rows
    joint_p, joint_r = [], []
    a_p, a_r = [], []
    b_p, b_r = [], []

    for i in range(d):
        est_full = set(np.where(np.abs(Theta_hat[i]) > threshold)[0].tolist())
        true_full = true_supports[i]

        # Block-wise: A indices in [0, d), B indices in [d, d+p)
        est_a = {j for j in est_full if j < d}
        true_a = {j for j in true_full if j < d}
        est_b = {j for j in est_full if j >= d}
        true_b = {j for j in true_full if j >= d}

        p_j, r_j = _row_metrics(est_full, true_full)
        joint_p.append(p_j)
        joint_r.append(r_j)

        # Skip rows where the relevant block has empty true support
        # AND empty estimated support (no signal to evaluate). Without
        # this, rows with no B-support contribute (0, 0) to the average,
        # which is misleading.
        if len(true_a) > 0 or len(est_a) > 0:
            p_a, r_a = _row_metrics(est_a, true_a)
            a_p.append(p_a)
            a_r.append(r_a)
        if len(true_b) > 0 or len(est_b) > 0:
            p_b, r_b = _row_metrics(est_b, true_b)
            b_p.append(p_b)
            b_r.append(r_b)

    def _aggregate(p_list, r_list):
        if len(p_list) == 0:
            return {"precision": np.nan, "recall": np.nan, "f1": np.nan}
        prec = float(np.mean(p_list))
        rec = float(np.mean(r_list))
        f1 = 2 * prec * rec / max(prec + rec, 1e-15)
        return {"precision": prec, "recall": rec, "f1": f1}

    return {
        "joint": _aggregate(joint_p, joint_r),
        "A": _aggregate(a_p, a_r),
        "B": _aggregate(b_p, b_r),
    }


# TODO: consider sampling from cones and reporting min Rayleigh coefficient
def restricted_gram_min_eigenvalue(Z: np.ndarray, true_supports: list[set]):
    """
    Minimum eigenvalue of Z_S^T Z_S / N restricted to each row's
    true support, then the minimum over rows.

    This is an upper bound on the restricted eigenvalue constant kappa.
    Hence, it can be used as a necessary condition for Lasso recovery.

    Parameters
    ----------
    Z : design matrix of shape (N, d+p)
    true_supports : list of sets

    Returns
    -------
    float : minimum restricted eigenvalue across all rows.
    """
    N = Z.shape[0]
    assert N > 0

    min_eig = np.inf
    for support_set in true_supports:
        cols = sorted(support_set)
        if len(cols) == 0:
            continue
        Z_S = Z[:, cols]  # (N, |S_i|)
        gram_S = Z_S.T @ Z_S / N  # (|S_i|, |S_i|)
        gram_S = (gram_S + gram_S.T) / 2.0  # symmetrise
        eigs = eigvalsh(gram_S)
        min_eig = min(min_eig, np.min(eigs))

    return min_eig if np.isfinite(min_eig) else 0.0


def regressor_energy_bound(Z):
    """
    Maximum average column energy: max_j (1/N) sum_i z_{i,j}^2.

    This is B from Proposition 10 (bounded regressor energy).

    Parameters
    ----------
    Z : design matrix of shape (N, d+p)

    Returns
    -------
    float : B
    """
    N = Z.shape[0]
    if N == 0:
        return 0.0
    return np.max(np.linalg.norm(Z, axis=0) / np.sqrt(N))


def closed_loop_spectral_abscissa(A_true, B_true, dre_solver, B_est, t_values=None):
    """
    Maximum real part of eigenvalues of A_true - B_true K_hat(t),
    evaluated at specified time points.

    Parameters
    ----------
    A_true : ndarray (d, d)
    B_true : ndarray (d, p)
    dre_solver : RiccatiODESolver
        With a valid solution loaded.
    B_est : ndarray (d, p)
        The B estimate used to compute K (since K(t) = R^{-1} B_est^T P(t)).
    t_values : list of float or None
        Times at which to evaluate. Default: [0, T - dt].

    Returns
    -------
    dict with keys 't=...' mapping to floats (spectral abscissa).
    """
    if dre_solver.solution is None:
        return {"t=0": np.nan, "t=T": np.nan}

    T = dre_solver.config.T
    dt = dre_solver.config.dt

    if t_values is None:
        t_values = [0.0, T - dt]

    result = {}
    for t in t_values:
        K = dre_solver.get_K(t, B_est)  # K already has sign: u = K @ x = -R^{-1}B^T P x
        # Closed-loop: A_true + B_true K  (since K = -R^{-1}B^TP, u = Kx)
        A_cl = A_true + B_true @ K
        eigs = eigvals(A_cl)
        spec_abs = np.max(np.real(eigs))
        result[f"t={t:.2f}"] = spec_abs

    return result


def basin_entry_episode(error_trajectory, thresholds=(0.05, 0.10, 0.15, 0.20, 0.30)):
    """
    First episode at which the relative parameter error drops below each threshold.

    Parameters
    ----------
    error_trajectory : list of float
        error_trajectory[m] = relative Frobenius error after episode m.
    thresholds : tuple of float

    Returns
    -------
    dict : threshold -> episode index (or None if never reached).
    """
    result = {}
    for eps in thresholds:
        try:
            result[eps] = min(m for m, err in enumerate(error_trajectory) if err <= eps)
        except ValueError:
            result[eps] = None
    return result


def episode_cost(xs, us, Q, R, dt):
    """
    Compute one episode's quadratic cost: sum_k (x_k^T Q x_k + u_k^T R u_k) * dt.

    Parameters
    ----------
    xs : ndarray (H, d) or list of ndarray (d,)
        States at each step.
    us : ndarray (H, p) or list of ndarray (p,)
        Controls at each step.
    Q : ndarray (d, d)
    R : ndarray (p, p)
    dt : float

    Returns
    -------
    float : episode cost.
    """
    X = np.asarray(xs)
    U = np.asarray(us)
    state_cost = np.sum(X * (X @ Q.T), axis=1)  # x^T Q x per step
    control_cost = np.sum(U * (U @ R.T), axis=1)  # u^T R u per step
    return np.sum((state_cost + control_cost) * dt)


def collect_diagnostics(
    agent, buffer, A_true, B_true, true_supports, Q, R, threshold=0.05
):
    """
    Collect all diagnostics for one agent at one episode.

    Returns a dict of scalars.
    """
    if agent.A_est is None:
        return {}

    d = {}

    # Parameter errors
    err = relative_parameter_error(agent.A_est, agent.B_est, A_true, B_true)
    d["error_joint"] = err["joint"]
    d["error_A"] = err["A"]
    d["error_B"] = err["B"]

    # Support recovery
    sup = support_metrics(agent.A_est, agent.B_est, true_supports, threshold)
    d["support_precision_joint"] = sup["joint"]["precision"]
    d["support_recall_joint"] = sup["joint"]["recall"]
    d["support_f1_joint"] = sup["joint"]["f1"]
    d["support_precision_A"] = sup["A"]["precision"]
    d["support_recall_A"] = sup["A"]["recall"]
    d["support_f1_A"] = sup["A"]["f1"]
    d["support_precision_B"] = sup["B"]["precision"]
    d["support_recall_B"] = sup["B"]["recall"]
    d["support_f1_B"] = sup["B"]["f1"]

    # Restricted Gram
    d["gram_min_eig"] = restricted_gram_min_eigenvalue(buffer.Z, true_supports)

    # Regressor energy
    d["regressor_energy"] = regressor_energy_bound(buffer.Z)

    # Spectral abscissa
    B_for_gain = agent.B_est
    spec = closed_loop_spectral_abscissa(A_true, B_true, agent.dre, B_for_gain)
    d["spectral_abscissa_t0"] = spec.get("t=0", np.nan)
    d["spectral_abscissa_tT"] = list(spec.values())[-1]

    return d
