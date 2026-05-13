import numpy as np
import pytest

from estimator import (
    RegressionBuffer,
    DiscreteRidgeEstimator,
    RowLassoEstimator,
)

# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_dims():
    """Provides standard dimensions for the buffer and estimators."""
    return {
        "x_dim": 4,
        "u_dim": 2,
        "max_episodes": 5,
        "steps_per_episode": 20,
    }

@pytest.fixture
def synthetic_data(default_dims):
    """Generates synthetic Z and Y data from a known, sparse linear system."""
    d, p = default_dims["x_dim"], default_dims["u_dim"]
    H = default_dims["steps_per_episode"]
    
    # Create a known, sparse Theta = [A B]
    Theta_true = np.zeros((d, d + p))
    Theta_true[0, 0] = 0.8        # A element
    Theta_true[1, d] = -0.5       # B element
    Theta_true[3, 2] = 0.9        # A element
    
    # Generate random features Z = [X; U]
    rng = np.random.default_rng(42)
    zs = rng.normal(0, 1, size=(H, d + p))
    
    # Generate noiseless targets Y = Z Theta^T
    ys = zs @ Theta_true.T
    
    return zs, ys, Theta_true


# ─────────────────────────────────────────────────────────────────────
# Tests: DiscreteRidgeEstimator
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.quick
def test_ridge_estimator_empty_buffer(default_dims):
    """Test that estimation fails safely on an empty buffer."""
    d, p = default_dims["x_dim"], default_dims["u_dim"]
    buf = RegressionBuffer(d, p, 1, 10)
    est = DiscreteRidgeEstimator(d, p)
    
    with pytest.raises(RuntimeError, match="Buffer is empty"):
        est.estimate(buf)


@pytest.mark.quick
def test_ridge_estimator_perfect_recovery(default_dims, synthetic_data):
    """Test that Ridge recovers the true parameters with negligible regularisation."""
    d, p = default_dims["x_dim"], default_dims["u_dim"]
    H = default_dims["steps_per_episode"]
    zs, ys, Theta_true = synthetic_data
    
    buf = RegressionBuffer(d, p, max_episodes=2, steps_per_episode=H)
    buf.add_episode(zs, ys)
    
    # Use an extremely small mu to approach OLS limit
    est = DiscreteRidgeEstimator(d, p, mu=1e-10)
    A_hat, B_hat = est.estimate(buf)
    Theta_hat = np.hstack([A_hat, B_hat])
    
    np.testing.assert_allclose(Theta_hat, Theta_true, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────
# Tests: RowLassoEstimator
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.quick
def test_lasso_theoretical_schedule(default_dims):
    """Test that the lambda scaling schedule is calculated correctly."""
    d, p = default_dims["x_dim"], default_dims["u_dim"]
    sigma_bar = 0.5
    M = 100
    delta = 0.05
    c_lambda = 2.0
    
    est = RowLassoEstimator(
        d, p, 
        sigma_bar=sigma_bar, 
        max_episodes=M, 
        c_lambda=c_lambda, 
        delta=delta
    )
    
    N = 50
    # Expected formula: c_lambda * sigma_bar * sqrt(log((d+p)*M*d/delta) / N)
    log_term = np.log((d + p) * M * d / delta)
    expected_lambda = c_lambda * sigma_bar * np.sqrt(log_term / N)
    
    assert np.isclose(est._get_lambda(N), expected_lambda)


@pytest.mark.quick
def test_lasso_estimator_sparse_recovery(default_dims, synthetic_data):
    """Test that Lasso forces unused parameters exactly to zero."""
    d, p = default_dims["x_dim"], default_dims["u_dim"]
    H = default_dims["steps_per_episode"]
    zs, ys, Theta_true = synthetic_data
    
    buf = RegressionBuffer(d, p, max_episodes=2, steps_per_episode=H)
    buf.add_episode(zs, ys)
    
    # Use a fixed lambda large enough to zero out noise but keep strong signals
    est = RowLassoEstimator(d, p, lambda_fixed=0.01)
    A_hat, B_hat = est.estimate(buf)
    Theta_hat = np.hstack([A_hat, B_hat])
    
    # Check that the non-zero elements are approximately recovered
    assert np.abs(Theta_hat[0, 0] - Theta_true[0, 0]) < 0.1
    assert np.abs(Theta_hat[1, d] - Theta_true[1, d]) < 0.1
    
    # Check that a known zero element is forced EXACTLY to 0.0 by the L1 penalty
    # (sklearn's coordinate descent enforces exact zeros)
    assert Theta_hat[0, 1] == 0.0
    assert Theta_hat[2, 0] == 0.0


@pytest.mark.quick
def test_lasso_warm_starting(default_dims, synthetic_data):
    """Test that the warm-start coefficients are updated after an estimation pass."""
    d, p = default_dims["x_dim"], default_dims["u_dim"]
    H = default_dims["steps_per_episode"]
    zs, ys, _ = synthetic_data
    
    buf = RegressionBuffer(d, p, max_episodes=2, steps_per_episode=H)
    buf.add_episode(zs, ys)
    
    est = RowLassoEstimator(d, p, lambda_fixed=0.01)
    
    # Initially all zeros
    assert np.all(est._warm_coefs[0] == 0.0)
    
    A_hat, B_hat = est.estimate(buf)
    
    # After estimation, warm_coefs should match the latest Theta_hat
    Theta_hat = np.hstack([A_hat, B_hat])
    np.testing.assert_allclose(est._warm_coefs[0], Theta_hat[0, :])