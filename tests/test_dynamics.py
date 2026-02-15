import numpy as np
import pytest

from common import SystemConfig
from dynamics import ContinuousLQREnv


@pytest.fixture
def config():
    return SystemConfig(x_dim=2, u_dim=1, dt=0.01)


@pytest.mark.quick
def test_deterministic_dynamics():
    """
    Verify that with Σ=0, the system evolves exactly as a deterministic Euler step.
    dx = (Ax + Bu) * dt
    Scenario: Resonance. The amplitude should match 0.5 * t * sin(t).
    Dynamics: x'' = -x + u with u = cos(t)
    """
    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Σ = np.zeros((2, 2))
    x0 = np.array([0.0, 0.0])

    config = SystemConfig(x_dim=2, u_dim=1, dt=0.001, T=6.0)
    env = ContinuousLQREnv(A, B, Σ, x0, config)

    times = np.linspace(0.0, config.T, int(config.T / config.dt))
    expected_positions = 0.5 * times * np.sin(times)
    actual_positions = np.empty_like(expected_positions)

    for i, t in enumerate(times):
        u = np.array([np.cos(t)])
        x, _, _ = env.step(u)
        actual_positions[i] = x[0]

    np.testing.assert_allclose(
        actual_positions,
        expected_positions,
        atol=1e-2,
        err_msg="Simulation trajectory diverged from analytical resonance solution",
    )


@pytest.mark.quick
def test_stochastic_scaling(config):
    """
    Verify that noise scales correctly with sqrt(dt).
    If A=0, B=0, Σ=I, then Var(dx) should be approx dt * I.
    """
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    Σ = np.eye(2)  # Identity diffusion
    x0 = np.zeros(2)

    # Use a large sample size for statistical significance
    N_samples = 5000
    env = ContinuousLQREnv(A, B, Σ, x0, config)

    x, x_prev = x0, x0
    dx_samples = []
    for _ in range(N_samples):
        x, _, _ = env.step(np.array([0.0]))
        dx_samples.append(x - x_prev)
        x_prev = x

    dx_samples = np.array(dx_samples)

    # Theoretical Variance of dx is Σ * Σ^T * dt = I * 0.01
    sample_var = np.var(dx_samples, axis=0)
    expected_var = config.dt_sim
    np.testing.assert_allclose(sample_var, expected_var, rtol=0.05)

    sample_mean = np.mean(dx_samples, axis=0)
    expected_mean = 0.0
    np.testing.assert_allclose(sample_mean, expected_mean, atol=0.005)


@pytest.mark.quick
def test_reset_behavior(config):
    """
    Verify reset restores x0 and t=0.
    """
    A = np.eye(2)
    B = np.zeros((2, 1))
    Σ = np.zeros((2, 2))
    x0 = np.array([5.0, -5.0])

    env = ContinuousLQREnv(A, B, Σ, x0, config)

    # Step forward
    env.step(np.array([0.0]))
    assert env.t > 0
    assert not np.array_equal(env.x, x0)

    # Reset
    resetted_state = env.reset()

    assert env.t == 0.0
    np.testing.assert_array_equal(resetted_state, x0)
    np.testing.assert_array_equal(env.x, x0)


@pytest.mark.quick
def test_immutability(config):
    """
    Verify that modifying input arrays DOES NOT corrupt the environment.
    """
    x0 = np.array([1.0, 1.0])
    A = np.eye(2)
    B = np.zeros((2, 1))
    Σ = np.eye(2)

    env = ContinuousLQREnv(A, B, Σ, x0, config)

    # Modify the original x0
    x0[0] = 9999.0

    # Reset env - it should NOT use 9999.0
    s = env.reset()
    assert s[0] == 1.0
