import numpy as np
import pytest

from dataclasses import dataclass
from system_generator import (
    sample_sparse_system,
    _is_controllable,
    generate_cost_matrices,
)


@dataclass(frozen=True)
class MockSystemConfig:
    """Mock configuration to decouple tests from the actual common.py."""

    x_dim: int
    u_dim: int


@pytest.fixture
def default_cfg():
    return MockSystemConfig(x_dim=10, u_dim=3)


@pytest.mark.quick
def test_sample_sparse_system_shapes_and_reproducibility(default_cfg):
    """Test that the output matrices have the correct dimensions and seeds are deterministic."""
    s = 3
    A1, B1, supports1 = sample_sparse_system(default_cfg, s=s, seed=42)
    A2, B2, supports2 = sample_sparse_system(default_cfg, s=s, seed=42)

    # Shape checks
    assert A1.shape == (default_cfg.x_dim, default_cfg.x_dim)
    assert B1.shape == (default_cfg.x_dim, default_cfg.u_dim)
    assert len(supports1) == default_cfg.x_dim

    # Reproducibility checks
    np.testing.assert_allclose(A1, A2)
    np.testing.assert_allclose(B1, B2)
    assert supports1 == supports2


@pytest.mark.quick
def test_sample_sparse_system_sparsity_and_gap_values(default_cfg):
    """Test the exact sparsity constraints and the 'gap' uniform distribution."""
    s = 3
    A, B, supports = sample_sparse_system(default_cfg, s=s, seed=100)
    Theta = np.hstack([A, B])

    for i in range(default_cfg.x_dim):
        # The support length should be s, plus potentially the diagonal
        # if it wasn't already in the initial uniform sample.
        assert s <= len(supports[i]) <= s + 1

        # Reconstruct the expected support from the matrix to verify the sets
        actual_non_zeros = set(np.where(np.abs(Theta[i]) > 1e-12)[0])
        assert supports[i] == actual_non_zeros

        # Check the gap uniform distribution for off-diagonal elements
        min_val, max_val = 0.3, 1.0
        for j in supports[i]:
            if j != i:  # Ignore diagonal, which is modified by the stability shift
                val = np.abs(Theta[i, j])
                assert min_val <= val <= max_val, (
                    f"Value {val} outside gap distribution"
                )


@pytest.mark.quick
def test_sample_sparse_system_stability(default_cfg):
    """Test that the generated A matrix is strictly stable according to the margin."""
    margin = 0.5
    A, _, _ = sample_sparse_system(default_cfg, s=3, seed=200, stability_margin=margin)

    eigs = np.linalg.eigvals(A)
    max_real_part = np.max(np.real(eigs))

    # We account for minor floating point inaccuracies
    assert max_real_part <= -margin + 1e-10


@pytest.mark.quick
def test_is_controllable_lyapunov_check():
    """Test the continuous Lyapunov controllability check."""
    # A known stable, controllable system (1D integrator with friction)
    A_cont = np.array([[-1.0]])
    B_cont = np.array([[1.0]])
    assert _is_controllable(A_cont, B_cont) is True

    # A known uncontrollable system (B is zero)
    A_uncont = np.array([[-1.0]])
    B_uncont = np.array([[0.0]])
    assert _is_controllable(A_uncont, B_uncont) is False

    # A completely disconnected 2D system
    A_disc = np.array([[-1.0, 0.0], [0.0, -1.0]])
    B_disc = np.array([[1.0], [0.0]])  # State 2 cannot be affected
    assert _is_controllable(A_disc, B_disc) is False


@pytest.mark.quick
def test_generate_cost_matrices():
    """Test identity cost matrix generation."""
    d, p = 5, 2
    Q, R = generate_cost_matrices(d, p)

    np.testing.assert_allclose(Q, np.eye(d))
    np.testing.assert_allclose(R, np.eye(p))


@pytest.mark.quick
def test_rejection_sampling_degeneracy():
    """Ensure the generator handles edge cases without crashing."""
    # By setting p=1 and s=1, we maximise the chance of a zero column in B
    # The generator should silently reject them and eventually find a valid one.
    cfg = MockSystemConfig(x_dim=5, u_dim=1)
    _, B, _ = sample_sparse_system(cfg, s=1, seed=99, max_attempts=1000)
    assert np.any(B != 0)  # B ist just a single column here
