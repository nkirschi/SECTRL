"""
Generates random sparse linear-quadratic systems for benchmarking.

Each row of Theta_star = [A_star  B_star] has at most s nonzero entries.

Design choices (kept simple deliberately):
- Exact per-row sparsity s (matches the theory).
- Coefficient magnitudes drawn from Uniform(0.3, 1.0) with random sign,
  giving a signal gap that prevents near-zero coefficients.
- No artificial stability shift: A is accepted as sampled, allowing
  the algorithm to be tested on the full class of stabilisable systems,
  which is the condition the theory actually requires.
- Rejection sampling on three criteria:
    (i)  No all-zero columns in B (every control direction must be active).
    (ii) Stabilisability: all unstable modes of A are reachable from B,
         verified via the Hautus lemma at each eigenvalue with Re >= 0.
    (iii) Bounded instability: max Re(lambda(A)) <= max_instability,
         so exploration episodes do not cause exponential state blowup.
"""

import numpy as np
from typing import Set, Tuple, List
from numpy.typing import NDArray


def sample_sparse_system(
    d: int,
    p: int,
    s: int,
    seed: int,
    coeff_magnitude: Tuple[float, float] = (0.3, 1.0),
    max_instability: float = 1.0,
    max_attempts: int = 100,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[Set[int]], int]:
    """
    Sample a sparse, stabilisable LQ system.

    Parameters
    ----------
    d : int
        State dimension.
    p : int
        Control dimension.
    s : int
        Exact number of nonzeros per row of [A B].
    seed : int
        Random seed.
    coeff_magnitude : Tuple[float, float]
        Range from which to draw coefficient magnitudes uniformly.
    max_instability : float
        Maximum allowed real part of any eigenvalue of A.
        Limits exponential state growth during the exploration phase
        (state grows at most by e^{max_instability * T} per episode).
        Default 1.0 corresponds to e^1 ≈ 2.7-fold growth over T=1.
    max_attempts : int
        Maximum rejection-sampling attempts.

    Returns
    -------
    A_star, B_star, supports, n_attempts
    """
    rng = np.random.default_rng(seed)

    for attempt in range(1, max_attempts + 1):
        Theta = np.zeros((d, d + p), dtype=np.float64)
        supports = []  # TODO: separate A and B sparsity as option

        for i in range(d):
            idx = rng.choice(d + p, size=s, replace=False)
            supports.append(set(int(j) for j in idx))
            for j in idx:
                magnitude = rng.uniform(*coeff_magnitude)
                sign = rng.choice([-1, 1])
                Theta[i, j] = sign * magnitude

        A = Theta[:, :d]
        B = Theta[:, d:]

        # TODO: if necessary also enforce at least two nonzeros per row of B
        # (i) No inactive control direction
        if p > 0 and np.any(np.all(B == 0, axis=0)):
            continue

        # (ii) Bounded instability
        eigenvalues = np.linalg.eigvals(A)
        max_real = float(np.max(np.real(eigenvalues)))
        if max_real > max_instability:
            continue

        # (iii) Stabilisability
        if not _is_stabilisable(A, B, eigenvalues):
            continue

        return A, B, supports, attempt

    raise RuntimeError(
        f"Failed to sample valid system after {max_attempts} "
        f"attempts (d={d}, p={p}, s={s}, seed={seed}). "
        f"Consider increasing max_instability or max_attempts."
    )


def _is_stabilisable(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    eigenvalues: NDArray[np.complex128] = None,
    tol: float = 1e-8,
) -> bool:
    """
    Check stabilisability via the Hautus lemma.

    (A, B) is stabilisable iff rank([lambda*I - A, B]) = d
    for every eigenvalue lambda of A with Re(lambda) >= 0.

    Only unstable eigenvalues need to be checked: for stable eigenvalues
    (Re(lambda) < 0) the condition is automatically satisfied because any
    stabilising feedback leaves those modes stable.

    Parameters
    ----------
    A : NDArray (d, d)
    B : NDArray (d, p)
    eigenvalues : NDArray or None
        Pre-computed eigenvalues of A. Passed in to avoid recomputing
        when already available from the instability check.
    tol : float
        Rank tolerance.

    References
    ----------
    Hespanha, "Linear Systems Theory" (2018), Theorem 14.3
    """
    d = A.shape[0]
    if eigenvalues is None:
        eigenvalues = np.linalg.eigvals(A)

    for lam in eigenvalues:
        if np.real(lam) >= 0:
            block = np.hstack([lam * np.eye(d) - A, B])
            if np.linalg.matrix_rank(block, tol=tol) < d:
                return False

    return True


def _is_controllable(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    eigenvalues: NDArray[np.complex128] = None,
    tol: float = 1e-8,
) -> bool:
    """
    Check controllability via the Hautus lemma.

    (A, B) is controllable iff rank([lambda*I - A, B]) = d
    for every eigenvalue lambda of A.

    Kept for reference and testing; the sampler uses _is_stabilisable.

    References
    ----------
    Hespanha, "Linear Systems Theory" (2018), Theorem 12.3
    """
    d = A.shape[0]
    if eigenvalues is None:
        eigenvalues = np.linalg.eigvals(A)

    for lam in eigenvalues:
        block = np.hstack([lam * np.eye(d) - A, B])
        if np.linalg.matrix_rank(block, tol=tol) < d:
            return False

    return True


def define_cost_matrices(d: int, p: int):
    """
    Returns identity cost matrices Q = I_d, R = I_p.

    Since Q, R are symmetric positive definite, there exists a change
    of basis under which the cost is given by identity matrices.
    """
    return np.eye(d), np.eye(p)
