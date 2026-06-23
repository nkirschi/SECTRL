import numpy as np
import pytest

from common import SystemConfig, EstimatorConfig
from estimator import (
    RegressionBuffer,
    DiscreteRidgeEstimator,
    RowLassoEstimator,
)

@pytest.fixture
def default_dims():
    return {
        "d": 4,
        "p": 2,
        "max_episodes": 5,
        "steps_per_episode": 20,
    }

@pytest.fixture
def configs(default_dims):
    sys_cfg = SystemConfig(
        d=default_dims["d"], p=default_dims["p"], 
        s_A=2, s_B=1, a_min=0.1, a_max=1.9, b_min=0.1, b_max=1.9, sigma=0.5, dt=0.01, T=1.0
    )
    est_cfg = EstimatorConfig(
        mu_ridge=1e-10, lambda_lasso=None, c_lambda=2.0, delta=0.05, 
        lasso_max_iter=1000, lasso_tol=1e-6
    )
    return sys_cfg, est_cfg

@pytest.fixture
def synthetic_data(default_dims):
    d, p = default_dims["d"], default_dims["p"]
    H = default_dims["steps_per_episode"]
    
    Theta_true = np.zeros((d, d + p))
    Theta_true[0, 0] = 0.8        
    Theta_true[1, d] = -0.5       
    Theta_true[3, 2] = 0.9        
    
    rng = np.random.default_rng(42)
    zs = rng.normal(0, 1, size=(H, d + p))
    ys = zs @ Theta_true.T
    
    return zs, ys, Theta_true

@pytest.mark.quick
def test_ridge_estimator_perfect_recovery(default_dims, configs, synthetic_data):
    _, est_cfg = configs
    d, p = default_dims["d"], default_dims["p"]
    H = default_dims["steps_per_episode"]
    zs, ys, Theta_true = synthetic_data
    
    buf = RegressionBuffer(d, p, max_episodes=2, steps_per_episode=H)
    buf.add_episode(zs, ys)
    
    est = DiscreteRidgeEstimator(est_cfg)
    Theta_hat = est.fit(buf.Z, buf.Y)
    
    np.testing.assert_allclose(Theta_hat, Theta_true, atol=1e-6)


@pytest.mark.quick
def test_lasso_estimator_sparse_recovery(default_dims, configs, synthetic_data):
    sys_cfg, _ = configs
    est_cfg = EstimatorConfig(
        mu_ridge=1e-10, lambda_lasso=0.01, c_lambda=2.0, delta=0.05, 
        lasso_max_iter=1000, lasso_tol=1e-6
    )
    
    d, p = default_dims["d"], default_dims["p"]
    H = default_dims["steps_per_episode"]
    zs, ys, Theta_true = synthetic_data
    
    buf = RegressionBuffer(d, p, max_episodes=2, steps_per_episode=H)
    buf.add_episode(zs, ys)
    
    est = RowLassoEstimator(sys_cfg, est_cfg)
    Theta_hat = est.fit(buf.Z, buf.Y)
    
    assert np.abs(Theta_hat[0, 0] - Theta_true[0, 0]) < 0.1
    assert np.abs(Theta_hat[1, d] - Theta_true[1, d]) < 0.1
    
    assert Theta_hat[0, 1] == 0.0
    assert Theta_hat[2, 0] == 0.0

@pytest.mark.quick
def test_lasso_warm_starting(default_dims, configs, synthetic_data):
    sys_cfg, _ = configs
    est_cfg = EstimatorConfig(
        mu_ridge=1e-10, lambda_lasso=0.01, c_lambda=2.0, delta=0.05,
        lasso_max_iter=1000, lasso_tol=1e-6
    )

    d, p = default_dims["d"], default_dims["p"]
    H = default_dims["steps_per_episode"]
    zs, ys, _ = synthetic_data

    buf = RegressionBuffer(d, p, max_episodes=2, steps_per_episode=H)
    buf.add_episode(zs, ys)

    est = RowLassoEstimator(sys_cfg, est_cfg)

    assert np.all(est._coef == 0.0)  # warm-start state starts empty
    Theta_hat = est.fit(buf.Z, buf.Y)
    # The carried warm-start coefficients are exactly the returned estimate.
    np.testing.assert_allclose(est._coef, Theta_hat)


@pytest.mark.quick
def test_lasso_streaming_matches_full_fit(default_dims, configs):
    """
    The incremental estimator must reproduce, episode by episode, what a
    from-scratch sklearn Lasso fit on the full buffer would give -- this both
    pins correctness and guards the private Cython gram API across upgrades.
    """
    from dataclasses import replace
    from sklearn.linear_model import Lasso

    sys_cfg, est_cfg = configs
    # Match the warmup to the schedule so every fit uses the same lambda: this
    # test pins the streaming math, not the warmup behaviour.
    est_cfg = replace(est_cfg, lambda_warmup=0.02)
    d, p = default_dims["d"], default_dims["p"]
    H = default_dims["steps_per_episode"]
    z_dim = d + p

    rng = np.random.default_rng(7)
    Theta_true = np.zeros((d, z_dim))
    Theta_true[0, 0], Theta_true[1, d], Theta_true[3, 2] = 0.8, -0.5, 0.9

    buf = RegressionBuffer(d, p, max_episodes=5, steps_per_episode=H)
    est = RowLassoEstimator(sys_cfg, est_cfg, lamda_schedule=lambda n: 0.02)

    for _ in range(4):  # several episodes -> exercises incremental accumulation
        zs = rng.normal(0, 1, size=(H, z_dim))
        ys = zs @ Theta_true.T + 0.05 * rng.normal(0, 1, size=(H, d))
        buf.add_episode(zs, ys)

        theta_stream = est.fit(buf.Z, buf.Y)

        # Reference: independent full-data Lasso per row, normalised objective.
        Z, Y = buf.Z, buf.Y
        theta_ref = np.zeros((d, z_dim))
        for i in range(d):
            m = Lasso(alpha=0.02, fit_intercept=False,
                      max_iter=est_cfg.lasso_max_iter, tol=est_cfg.lasso_tol)
            m.fit(Z, Y[:, i])
            theta_ref[i] = m.coef_

        np.testing.assert_allclose(theta_stream, theta_ref, atol=1e-7)


@pytest.mark.quick
def test_lasso_warmup_first_fit_only(default_dims, configs):
    """
    The first scheduled fit uses the small warmup penalty (so a weak initial
    signal is not shrunk to zero -- breaking the trap), and the theoretical
    schedule applies from the second fit on.
    """
    from dataclasses import replace

    sys_cfg, est_cfg = configs
    est_cfg = replace(est_cfg, lambda_warmup=1e-6)
    d, p = default_dims["d"], default_dims["p"]
    H = default_dims["steps_per_episode"]

    rng = np.random.default_rng(1)
    Theta_true = np.zeros((d, d + p))
    Theta_true[0, 0] = 0.5
    zs = rng.normal(0, 1, size=(H, d + p))
    ys = zs @ Theta_true.T + 0.01 * rng.normal(0, 1, size=(H, d))

    # A schedule so large it would zero every coefficient if ever applied.
    est = RowLassoEstimator(sys_cfg, est_cfg, lamda_schedule=lambda n: 100.0)

    buf = RegressionBuffer(d, p, max_episodes=3, steps_per_episode=H)
    buf.add_episode(zs, ys)
    first = est.fit(buf.Z, buf.Y)
    assert np.any(first != 0.0)  # warmup -> nonzero estimate

    buf.add_episode(zs, ys)
    second = est.fit(buf.Z, buf.Y)
    assert np.all(second == 0.0)  # schedule (lambda=100) -> shrunk to zero