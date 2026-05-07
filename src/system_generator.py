"""
Generates random sparse linear-quadratic systems for benchmarking.

Each row of Theta_star = [A_star  B_star] has at most s nonzero entries.
A_star is shifted to ensure the uncontrolled drift is stable.

Design choices (kept simple deliberately):
- Exact per-row sparsity s (matches the theory).
- Coefficient magnitudes drawn from Uniform(0.3, 1.0) with random sign,
  giving a signal gap that prevents near-zero coefficients
  (which would be hard to distinguish from true zeros).
- Stability shift A <- A - (max Re lambda + margin) * I to ensure
  the uncontrolled drift is stable, if necessary.
- Controllability check via the Kalman rank condition; resample if
  the pair is not controllable.
- No all-zero columns in B (every control direction must be active).
- Modern np.random.Generator API.
"""

import numpy as np
from typing import Set, Tuple, List
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_lyapunov, LinAlgError


def sample_sparse_system(
    d: int,
    p: int,
    s: int,
    seed: int,
    stability_margin: float = 0.5,
    max_attempts: int = 100,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[Set[int]], int]:
    """
    Sample a sparse, stable, and controllable LQ system.
    Uses rejection sampling to ensure exact row-wise sparsity and non-degeneracy.
    """
    rng = np.random.default_rng(seed)

    for attempt in range(1, max_attempts + 1):
        Theta = np.zeros((d, d + p), dtype=np.float64)
        supports = []  # TODO: separate A and B sparsity?

        for i in range(d):
            # 1. Exact Sparsity
            idx = rng.choice(
                d + p, size=s, replace=False
            )  # TODO: later do robustness sweep
            supports.append(set(int(j) for j in idx))

            # 2. Gap Uniform Distribution (protects Lasso signal-to-noise ratio)
            for j in idx:  # TODO replace 0.3 with hyperparam, maybe set upper bound s.t. expected value is 1?
                magnitude = rng.uniform(
                    0.3, 1.0
                )  # info: this is standard in causality literature (TODO find reference)
                sign = rng.choice([-1, 1])
                Theta[i, j] = sign * magnitude

        A = Theta[:, :d]
        B = Theta[:, d:]

        # 3. Rejection Sampling if any control direction is inactive (all-zero column in B)
        if p > 0 and np.any(np.all(B == 0, axis=0)):
            continue

        # 4. Conditional Stability Shift
        max_real_eig = float(
            np.max(np.real(np.linalg.eigvals(A)))
        )  # TODO: do we need this empirically? perhaps hyperparametrise a positive threshold?
        shift = max(0.0, max_real_eig + stability_margin)

        if shift > 0.0:
            A = A - shift * np.eye(d)
            # Update Theta and supports to reflect the diagonal shift
            Theta[:, :d] = A
            for i in range(d):
                supports[i].add(i)

        # 5. Numerically Stable Controllability Check
        if _is_controllable(A, B):
            return A, B, supports, attempt

    raise RuntimeError(
        f"Failed to sample valid system after {max_attempts} "
        f"attempts (d={d}, p={p}, s={s}, seed={seed})"
    )


def _is_controllable(
    A: NDArray[np.float64], B: NDArray[np.float64], tol: float = 1e-10
) -> bool:
    """
    Check controllability via the Infinite-Horizon Controllability Gramian.
    Requires A to be strictly Hurwitz (stable), which is guaranteed by the generator.

    References:
    - Hespanha, "Linear Systems Theory", 2018, Theorem 12.4
    - Chen, "Linear System Theory and Design", 2013, Theorem 6.1
    """
    try:
        W_c = solve_continuous_lyapunov(A, -B @ B.T)
        W_c = (W_c + W_c.T) / 2.0  # symmetrise for numerical stability
        min_eig = float(np.min(np.linalg.eigvalsh(W_c)))
        return min_eig > tol
    except LinAlgError:
        return False


def define_cost_matrices(d, p):
    """
    Returns w.l.o.g. identity cost matrices Q = I_d, R = I_p.

    Since Q, R are symmetric positive definite, there exists a change of basis
    under which the cost is given by identity matrices.
    """
    return np.eye(d), np.eye(p)
