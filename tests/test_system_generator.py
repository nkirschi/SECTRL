"""
Unit tests for system_generator.py.

Run with:  pytest test_system_generator.py -v
"""

import numpy as np
import pytest

from system_generator import (
    sample_sparse_system,
    _is_stabilisable,
    _is_controllable,
    define_cost_matrices,
)


class TestIsStabilisable:
    def test_stable_system_always_stabilisable(self):
        # Any Hurwitz A is stabilisable regardless of B,
        # since there are no unstable modes to check.
        A = np.diag([-1.0, -2.0, -0.5])
        B = np.zeros((3, 1))  # zero B — still stabilisable
        assert _is_stabilisable(A, B) is True

    def test_unstable_mode_reachable(self):
        # A has one unstable eigenvalue (+0.5).
        # B reaches that mode -> stabilisable.
        A = np.diag([0.5, -1.0])
        B = np.array([[1.0], [0.0]])  # only reaches state 0 (eigenvalue +0.5)
        assert _is_stabilisable(A, B) is True

    def test_unstable_mode_unreachable(self):
        # A has one unstable eigenvalue (+0.5).
        # B only reaches the stable mode -> NOT stabilisable.
        A = np.diag([0.5, -1.0])
        B = np.array([[0.0], [1.0]])  # only reaches state 1 (eigenvalue -1.0)
        assert _is_stabilisable(A, B) is False

    def test_two_unstable_modes_both_reachable(self):
        A = np.diag([0.5, 1.0, -1.0])
        B = np.eye(3)[:, :2]  # reaches first two states
        assert _is_stabilisable(A, B) is True

    def test_two_unstable_modes_one_unreachable(self):
        A = np.diag([0.5, 1.0, -1.0])
        B = np.eye(3)[:, :1]  # only reaches state 0
        assert _is_stabilisable(A, B) is False

    def test_marginally_stable_eigenvalue_is_checked(self):
        # Re(lambda) = 0 (imaginary axis) must be included in the check.
        A = np.diag([0.0, -1.0])  # one eigenvalue exactly on imaginary axis
        B_reach = np.array([[1.0], [0.0]])  # reaches the marginal mode
        B_no_reach = np.array([[0.0], [1.0]])  # does not
        assert _is_stabilisable(A, B_reach) is True
        assert _is_stabilisable(A, B_no_reach) is False

    def test_stabilisable_not_controllable(self):
        # Classic example: A decoupled, B only reaches unstable mode.
        A = np.diag([0.5, -1.0])
        B = np.array([[1.0], [0.0]])
        assert _is_stabilisable(A, B) is True
        assert _is_controllable(A, B) is False

    def test_accepts_precomputed_eigenvalues(self):
        A = np.diag([0.5, -1.0])
        B = np.array([[1.0], [0.0]])
        eigs = np.linalg.eigvals(A)
        result_with = _is_stabilisable(A, B, eigenvalues=eigs)
        result_without = _is_stabilisable(A, B, eigenvalues=None)
        assert result_with == result_without


class TestIsControllable:
    def test_companion_form_is_controllable(self):
        # Single integrator chain: always controllable.
        A = np.array([[-1.0, 1.0], [0.0, -2.0]])
        B = np.array([[0.0], [1.0]])
        assert _is_controllable(A, B) is True

    def test_decoupled_unreachable_mode_not_controllable(self):
        A = np.diag([-1.0, -2.0])
        B = np.array([[1.0], [0.0]])  # mode 2 (eig -2) unreachable
        assert _is_controllable(A, B) is False

    def test_full_rank_B_is_controllable(self):
        # If B has full column rank >= d, always controllable.
        A = np.diag([-1.0, -1.5, -2.0])
        B = np.eye(3)  # identity B
        assert _is_controllable(A, B) is True

    def test_stabilisable_not_controllable(self):
        # A has one unstable eigenvalue (+0.5).
        # B reaches that mode but not the stable one -> stabilisable and NOT controllable.
        A = np.diag([0.5, -1.0])
        B = np.array([[1.0], [0.0]])  # only reaches state 0 (eigenvalue +0.5)
        assert _is_stabilisable(A, B) is True
        assert _is_controllable(A, B) is False


@pytest.fixture
def d():
    return 8


@pytest.fixture
def p():
    return 3


@pytest.fixture
def s():
    return 2


class TestSampleSparseSystem:
    def test_returns_correct_shapes(self, d, p, s):
        A, B, supports, _ = sample_sparse_system(d, p, s, seed=0)
        assert A.shape == (d, d)
        assert B.shape == (d, p)
        assert len(supports) == d

    def test_reproducible_with_same_seed(self, d, p, s):
        A1, B1, _, _ = sample_sparse_system(d, p, s, seed=42)
        A2, B2, _, _ = sample_sparse_system(d, p, s, seed=42)
        assert np.allclose(A1, A2) and np.allclose(B1, B2)

    def test_different_seeds_give_different_systems(self, d, p, s):
        A1, _, _, _ = sample_sparse_system(d, p, s, seed=0)
        A2, _, _, _ = sample_sparse_system(d, p, s, seed=1)
        assert not np.allclose(A1, A2)

    def test_exact_row_sparsity(self, d, p, s):
        # Each row of Theta = [A B] should have at most s+d_diagonal nonzeros.
        # After sampling: support has exactly s entries; diagonal entries may
        # be added if any fall outside the original support (but only when the
        # stability shift was applied, which no longer happens here).
        A, B, supports, _ = sample_sparse_system(d, p, s, seed=7)
        Theta = np.hstack([A, B])
        for i in range(d):
            # Nonzeros at indices in supports[i]
            nonzero_cols = set(np.where(np.abs(Theta[i]) > 1e-12)[0].tolist())
            assert nonzero_cols == supports[i], (
                f"Row {i}: nonzero cols {nonzero_cols} != support {supports[i]}"
            )

    def test_coefficient_magnitudes_in_gap_range(self, d, p, s):
        # All nonzero entries should have magnitude in [a, b].
        for a, b in [(0.3, 1.0), (0.5, 2.0), (1.0, 5.0)]:
            A, B, supports, _ = sample_sparse_system(d, p, s, 0, coeff_magnitude=(a, b))
            Theta = np.hstack([A, B])
            for i, supp in enumerate(supports):
                for j in supp:
                    mag = abs(Theta[i, j])
                    assert a <= mag <= b, (
                        f"Row {i}, col {j}: magnitude {mag:.4f} outside [{a}, {b}]"
                    )

    def test_no_all_zero_b_columns(self, d, p, s):
        A, B, _, _ = sample_sparse_system(d, p, s, seed=3)
        assert not np.any(np.all(B == 0, axis=0)), "B has an all-zero column"

    def test_max_instability_respected(self, d, p, s):
        # The maximum real part of A's eigenvalues must not exceed max_instability.
        max_inst = 0.8
        for seed in range(20):
            A, _, _, _ = sample_sparse_system(
                d, p, s, seed=seed, max_instability=max_inst
            )
            max_re = float(np.max(np.real(np.linalg.eigvals(A))))
            assert max_re <= max_inst + 1e-10, (
                f"seed={seed}: max Re(eig)={max_re:.4f} > {max_inst}"
            )

    def test_sampled_system_is_stabilisable(self, d, p, s):
        for seed in range(20):
            A, B, _, _ = sample_sparse_system(d, p, s, seed=seed)
            assert _is_stabilisable(A, B), (
                f"seed={seed}: sampled system is not stabilisable"
            )

    def test_supports_are_valid_index_sets(self, d, p, s):
        _, _, supports, _ = sample_sparse_system(d, p, s, seed=0)
        dp = d + p
        for i, supp in enumerate(supports):
            assert all(0 <= j < dp for j in supp), (
                f"Row {i} support contains out-of-range index"
            )

    def test_accepts_unstable_systems(self, d, p, s):
        # Without the stability shift, some accepted systems should have
        # at least one eigenvalue with positive real part.
        has_unstable = False
        for seed in range(50):
            A, _, _, _ = sample_sparse_system(d, p, s, seed=seed, max_instability=1.0)
            if np.max(np.real(np.linalg.eigvals(A))) > 0:
                has_unstable = True
                break
        assert has_unstable, (
            "No unstable systems accepted across 50 seeds — "
            "stability shift may still be active"
        )

    def test_high_max_instability_accepts_more_systems(self, d, p, s):
        # Increasing max_instability should require fewer attempts on average.
        attempts_tight = [
            sample_sparse_system(d, p, s, seed=seed, max_instability=0.3)[3]
            for seed in range(20)
        ]
        attempts_loose = [
            sample_sparse_system(d, p, s, seed=seed, max_instability=2.0)[3]
            for seed in range(20)
        ]
        assert np.mean(attempts_loose) <= np.mean(attempts_tight), (
            "Looser instability bound should not require more attempts"
        )

    def test_raises_on_impossible_config(self, d, p, s):
        # s=1 in a large system with p=1 makes many rows have only A-block
        # support, and controllability can be hard to achieve. With only 5
        # attempts this should fail.
        with pytest.raises(RuntimeError, match="Failed to sample"):
            sample_sparse_system(
                d=20,
                p=1,
                s=1,
                seed=0,
                max_attempts=5,
            )
        with pytest.raises(RuntimeError, match="Failed to sample"):
            sample_sparse_system(
                d=d,
                p=p,
                s=s,
                seed=0,
                max_instability=0.01,  # extremely tight: almost forces all eigs < 0.01
                max_attempts=5,
            )


class TestDefineCostMatrices:
    def test_returns_identity_matrices(self):
        Q, R = define_cost_matrices(5, 3)
        assert np.allclose(Q, np.eye(5))
        assert np.allclose(R, np.eye(3))

    def test_correct_shapes(self):
        for d, p in [(2, 1), (10, 5), (20, 3)]:
            Q, R = define_cost_matrices(d, p)
            assert Q.shape == (d, d)
            assert R.shape == (p, p)

    def test_positive_definite(self):
        Q, R = define_cost_matrices(4, 2)
        assert np.all(np.linalg.eigvalsh(Q) > 0)
        assert np.all(np.linalg.eigvalsh(R) > 0)
