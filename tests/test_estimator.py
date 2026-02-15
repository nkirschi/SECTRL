import numpy as np
import pytest

from common import SystemConfig
from estimator import ContinuousLeastSquaresEstimator


def generate_linear_data(A, B, config, excitation_scale=1.0):
    """
    Generates a synthetic trajectory consistent with dX = (Ax + Bu)dt.
    """
    state_dim = A.shape[0]
    action_dim = B.shape[1]

    data = []
    x = np.ones(state_dim)

    for _ in range(int(config.T / config.dt)):
        # 1. Control Policy
        if excitation_scale > 0.0:
            # Random control covers the whole state space
            u = np.random.normal(0, excitation_scale, size=action_dim)
        else:
            # Deterministic feedback (u = -x) causes rank deficiency!
            # The estimator will only see dX = (A - B)X dt
            u = -x[:action_dim]

        # 2. Dynamics
        # dX = (Ax + Bu)dt
        drift = A @ x + B @ u
        dx = drift * config.dt

        # Store tuple
        data.append((x.copy(), u.copy(), dx.copy(), config.dt))

        # Update state (Euler integration)
        x += dx

    return data


def test_perfect_recovery():
    """
    Test 1: Can we recover A and B from clean, excited data?
    """
    # System: Scalar x, Scalar u
    # dx = (2x + 3u)dt
    A_true = np.array([[2.0]])
    B_true = np.array([[3.0]])

    config = SystemConfig(x_dim=1, u_dim=1, T=10.0, dt=0.01)
    estimator = ContinuousLeastSquaresEstimator(config)

    # Generate 1000 steps of clean data with random inputs
    traj = generate_linear_data(A_true, B_true, config)
    estimator.add_trajectory(traj)

    A_hat, B_hat = estimator.estimate()

    # Expect near-perfect recovery (within floating point error)
    np.testing.assert_allclose(A_hat, A_true, atol=1e-5, err_msg="Failed to recover A")
    np.testing.assert_allclose(B_hat, B_true, atol=1e-2, err_msg="Failed to recover B")


def test_estimator_on_stable_system():
    """
    Verify estimator logic using a well-conditioned, stable system.
    If this fails, the estimator code is definitely broken.
    """
    # A = [[-1, 0], [0, -2]]  (Stable, decaying)
    # B = [[1, 0], [0, 0.5]]  (Direct control)
    A_true = np.array([[-1.0, 0.0], [0.0, -2.0]])
    B_true = np.array([[1.0, 0.0], [0.0, 0.5]])

    config = SystemConfig(x_dim=2, u_dim=2, T=10.0, dt=0.01)
    estimator = ContinuousLeastSquaresEstimator(config)
    traj = generate_linear_data(A_true, B_true, config)
    estimator.add_trajectory(traj)
    A_hat, B_hat = estimator.estimate()

    np.testing.assert_allclose(
        A_hat, A_true, atol=1e-5, err_msg="A estimation failed on stable system"
    )
    np.testing.assert_allclose(
        B_hat, B_true, atol=1e-5, err_msg="B estimation failed on stable system"
    )


def test_rank_deficiency_failure():
    """
    Test 2: Verify that 'Closed Loop' data (u = -Kx) confuses the estimator.
    Rationale: If u is perfectly correlated with x, we cannot distinguish A from B.
    The estimator should return a 'safe' regularized solution, but it will be WRONG physically.
    """
    # System: dx = (1x + 1u)dt.
    # If we use u = -x, then dx = 0.
    # The estimator sees x=1, u=-1, dx=0.
    # It could be A=1, B=1. Or A=0, B=0. Or A=2, B=2.
    A_true = np.array([[1.0]])
    B_true = np.array([[1.0]])

    config = SystemConfig(x_dim=1, u_dim=1, T=10.0, dt=0.01)
    estimator = ContinuousLeastSquaresEstimator(config)

    # Generate data with NO random excitation (u = -x exactly)
    traj = generate_linear_data(A_true, B_true, config, excitation_scale=0.0)
    estimator.add_trajectory(traj)

    A_hat, B_hat = estimator.estimate()

    # The estimates should NOT match the true physics because the data is degenerate.
    # If they do match, it's luck (regularization bias). usually they will be smaller.
    error = np.linalg.norm(A_hat - A_true) + np.linalg.norm(B_hat - B_true)

    # We Assert that error is LARGE to prove rank deficiency is real
    assert error > 0.1, (
        "Estimator miraculously guessed the right parameters despite rank-deficient data!"
    )


def test_scale_invariance():
    """
    Test 3: Does it work if signals are huge (1e6) or tiny (1e-6)?
    Rationale: Fixed regularization (lambda=1.0) fails on tiny signals.
    Adaptive regularization (lambda ~ trace) is needed.
    """
    # System: dx = x * dt
    A_true = np.array([[1.0]])
    B_true = np.array([[0.0]])
    config = SystemConfig(x_dim=1, u_dim=1, T=10.0, dt=0.01)

    # Case A: Tiny signals (Micro-scale)
    est_tiny = ContinuousLeastSquaresEstimator(config)
    traj_tiny = generate_linear_data(A_true, B_true, config)
    # Scale data down by 1e-6
    traj_scaled_down = [
        (x * 1e-6, u * 1e-6, dx * 1e-6, dt) for x, u, dx, dt in traj_tiny
    ]
    est_tiny.add_trajectory(traj_scaled_down)
    A_tiny, _ = est_tiny.estimate()

    # Case B: Huge signals (Mega-scale)
    est_huge = ContinuousLeastSquaresEstimator(config)
    # Scale data up by 1e6
    traj_scaled_up = [(x * 1e6, u * 1e6, dx * 1e6, dt) for x, u, dx, dt in traj_tiny]
    est_huge.add_trajectory(traj_scaled_up)
    A_huge, _ = est_huge.estimate()

    # Both should recover A=1.0 roughly
    # (Allow loose tolerance because numerical precision suffers at extremes)
    np.testing.assert_allclose(
        A_tiny, A_true, rtol=0.1, err_msg="Failed on tiny signals"
    )
    np.testing.assert_allclose(
        A_huge, A_true, rtol=0.1, err_msg="Failed on huge signals"
    )


@pytest.mark.slow
def test_large_batch_coupled_2d():
    """
    Stress Test: Large batch accumulation on a Coupled 2D System.
    Verifies that cross-terms (off-diagonals) converge correctly.
    """
    # System: Damped Oscillator (Spiral Sink)
    # dx1 = -x1 + x2
    # dx2 = -x1 - x2
    # Eigenvalues: -1 +/- i
    A_true = np.array([[-1.0, 1.0], [-1.0, -1.0]])
    B_true = np.array([[1.0, 0.0], [0.0, 1.0]])  # Independent control

    config = SystemConfig(x_dim=2, u_dim=2, T=10.0, dt=0.01)
    estimator = ContinuousLeastSquaresEstimator(config)

    # Parameters
    N_episodes = 1000

    # Generate data in a loop to simulate online learning
    for _ in range(N_episodes):
        traj = generate_linear_data(A_true, B_true, config)
        estimator.add_trajectory(traj)

    A_hat, B_hat = estimator.estimate()

    np.testing.assert_allclose(
        np.diag(A_hat),
        np.diag(A_true),
        atol=1e-9,
        err_msg="Failed to identify decay rates (diagonals)",
    )
    np.testing.assert_allclose(
        A_hat[0, 1],
        A_true[0, 1],
        atol=1e-9,
        err_msg="Failed to identify coupling term (x2 -> x1)",
    )
    np.testing.assert_allclose(
        A_hat[1, 0],
        A_true[1, 0],
        atol=1e-9,
        err_msg="Failed to identify coupling term (x1 -> x2)",
    )
    np.testing.assert_allclose(
        B_hat, B_true, atol=1e-9, err_msg="Failed to identify control matrix"
    )
