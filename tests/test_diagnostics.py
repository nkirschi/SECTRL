import numpy as np
import pytest
from unittest.mock import MagicMock

from diagnostics import (
    relative_parameter_error,
    support_metrics,
    restricted_gram_min_eigenvalue,
    regressor_energy_bound,
    closed_loop_spectral_abscissa,
    basin_entry_episode,
    episode_cost,
    collect_diagnostics,
)
from estimator import DiscreteDataBuffer


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_buffer(Z, Y):
    """Build a DiscreteDataBuffer pre-populated with given Z, Y."""
    N, dp = Z.shape
    d = Y.shape[1]
    p = dp - d
    # Use single 'episode' of length N to populate
    buf = DiscreteDataBuffer(d, p, max_episodes=1, steps_per_episode=N)
    buf.add_episode(Z, Y)
    return buf


def make_mock_dre_solver(T=1.0, dt=0.1, K_func=None):
    """Build a mock RiccatiODESolver with custom K(t) behaviour."""
    solver = MagicMock()
    solver.config = MagicMock()
    solver.config.T = T
    solver.config.dt = dt
    solver.solution = MagicMock()  # non-None to indicate "solved"

    if K_func is None:
        # Default: zero gain
        K_func = lambda t, B: np.zeros((B.shape[1], B.shape[0]))

    solver.get_K = MagicMock(side_effect=K_func)
    return solver


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


class TestRelativeParameterError:
    def test_perfect_estmate_gives_zero(self):
        A_true = np.array([[1.0, 0.5], [0.0, -1.0]])
        B_true = np.array([[0.5], [1.0]])
        A_est = A_true.copy()
        B_est = B_true.copy()
        err = relative_parameter_error(A_est, B_est, A_true, B_true)
        assert err["joint"] == pytest.approx(0.0, abs=1e-15)
        assert err["A"] == pytest.approx(0.0, abs=1e-15)
        assert err["B"] == pytest.approx(0.0, abs=1e-15)

    def test_known_relative_error(self):
        # Theta_true = identity-like, Theta_est = 2 * Theta_true
        # Then error = ||Theta_true||_F, relative error = 1.0
        A_true = np.eye(3)
        B_true = np.array([[1.0], [0.0], [0.0]])
        A_est = 2 * A_true
        B_est = 2 * B_true
        err = relative_parameter_error(A_est, B_est, A_true, B_true)
        assert err["joint"] == pytest.approx(1.0)
        assert err["A"] == pytest.approx(1.0)
        assert err["B"] == pytest.approx(1.0)

    def test_blocks_are_separated(self):
        # Error only in A block
        A_true = np.eye(2)
        B_true = np.eye(2)
        A_est = A_true + np.ones((2, 2))
        B_est = B_true.copy()
        err = relative_parameter_error(A_est, B_est, A_true, B_true)
        assert err["A"] > 0
        assert err["B"] == pytest.approx(0.0, abs=1e-15)

    def test_zero_truth_uses_eps_floor(self):
        # If Theta_true is zero, relative error should not divide by zero
        A_true = np.zeros((2, 2))
        B_true = np.zeros((2, 1))
        A_est = np.ones((2, 2))
        B_est = np.ones((2, 1))
        err = relative_parameter_error(A_est, B_est, A_true, B_true)
        # Result should be finite (very large, but not inf or nan)
        assert np.isfinite(err["joint"])
        assert np.isfinite(err["A"])
        assert np.isfinite(err["B"])

    def test_returns_dict_with_correct_keys(self):
        A = np.eye(2)
        B = np.eye(2)
        err = relative_parameter_error(A, B, A, B)
        assert set(err.keys()) == {"joint", "A", "B"}



class TestSupportMetrics:
    def test_perfect_recovery(self):
        # Two rows, each with 2 nonzeros at known positions
        A_est = np.array([[1.0, 0.0, 0.5], [0.0, 0.7, 0.0]])
        B_est = np.array([[0.0], [0.6]])
        true_supports = [{0, 2}, {1, 3}]  # row 0: cols 0,2; row 1: cols 1,3
        m = support_metrics(A_est, B_est, true_supports, threshold=0.1)
        assert m["joint"]["precision"] == pytest.approx(1.0)
        assert m["joint"]["recall"] == pytest.approx(1.0)
        assert m["joint"]["f1"] == pytest.approx(1.0)

    def test_threshold_excludes_small_values(self):
        # Coefficient 0.04 is below threshold 0.05
        A_est = np.array([[1.0, 0.04]])
        B_est = np.array([[0.0]])
        true_supports = [{0}]  # only column 0 is in true support
        m = support_metrics(A_est, B_est, true_supports, threshold=0.05)
        assert m["joint"]["precision"] == pytest.approx(1.0)
        assert m["joint"]["recall"] == pytest.approx(1.0)

    def test_false_positives_lower_precision(self):
        # Estimate has two nonzeros, only one matches truth
        A_est = np.array([[1.0, 1.0]])
        B_est = np.array([[0.0]])
        true_supports = [{0}]
        m = support_metrics(A_est, B_est, true_supports, threshold=0.5)
        assert m["joint"]["precision"] == pytest.approx(0.5)
        assert m["joint"]["recall"] == pytest.approx(1.0)

    def test_false_negatives_lower_recall(self):
        # Estimate misses one of two true nonzeros
        A_est = np.array([[1.0, 0.0]])
        B_est = np.array([[0.0]])
        true_supports = [{0, 1}]
        m = support_metrics(A_est, B_est, true_supports, threshold=0.5)
        assert m["joint"]["precision"] == pytest.approx(1.0)
        assert m["joint"]["recall"] == pytest.approx(0.5)

    def test_empty_estmate_gives_zero_precision(self):
        # Estimate is all zeros (below threshold)
        A_est = np.zeros((1, 2))
        B_est = np.zeros((1, 1))
        true_supports = [{0}]
        m = support_metrics(A_est, B_est, true_supports, threshold=0.5)
        # No selections: precision uses max(.., 1) so it's 0/1 = 0
        assert m["joint"]["precision"] == pytest.approx(0.0)
        assert m["joint"]["recall"] == pytest.approx(0.0)
        assert m["joint"]["f1"] == pytest.approx(0.0)

    def test_f1_is_harmonic_mean(self):
        # P = 0.5, R = 0.5  =>  F1 = 0.5
        A_est = np.array([[1.0, 1.0, 0.0, 0.0]])  # selects cols 0, 1
        B_est = np.array([[]]).reshape(1, 0)
        true_supports = [{0, 2}]  # truth: cols 0, 2
        m = support_metrics(A_est, B_est, true_supports, threshold=0.5)
        # TP=1, est_size=2 -> P=0.5; truth_size=2 -> R=0.5; F1=0.5
        assert m["joint"]["precision"] == pytest.approx(0.5)
        assert m["joint"]["recall"] == pytest.approx(0.5)
        assert m["joint"]["f1"] == pytest.approx(0.5)

    def test_averaged_over_rows(self):
        # Two rows: row 0 perfect (F1=1), row 1 worst (F1=0)
        A_est = np.array([[1.0, 0.0], [0.0, 0.0]])
        B_est = np.zeros((2, 1))
        true_supports = [{0}, {1}]  # row 1's truth not recovered
        m = support_metrics(A_est, B_est, true_supports, threshold=0.5)
        # Row 0: P=1, R=1; Row 1: P=0, R=0 (no estimated selections)
        assert m["joint"]["precision"] == pytest.approx(0.5)
        assert m["joint"]["recall"] == pytest.approx(0.5)

    def test_returns_joint_a_b_keys(self):
        # Top-level keys should be 'joint', 'A', 'B'
        A_est = np.array([[1.0]])
        B_est = np.array([[1.0]])
        true_supports = [{0, 1}]
        m = support_metrics(A_est, B_est, true_supports, threshold=0.5)
        assert set(m.keys()) == {"joint", "A", "B"}
        for block in ("joint", "A", "B"):
            assert set(m[block].keys()) == {"precision", "recall", "f1"}

    def test_a_block_only_uses_a_columns(self):
        # 1 state, 2 controls; true support {0, 1, 2}
        # estimate: A perfectly recovered, B both columns wrong
        A_est = np.array([[1.0]])  # column 0
        B_est = np.array([[0.0, 0.0]])  # columns 1, 2 -- both missing
        true_supports = [{0, 1, 2}]
        m = support_metrics(A_est, B_est, true_supports, threshold=0.5)
        # A block: estimated {0}, true {0} -> P=1, R=1, F1=1
        assert m["A"]["precision"] == pytest.approx(1.0)
        assert m["A"]["recall"] == pytest.approx(1.0)
        # B block: estimated {}, true {1, 2} -> P=0, R=0, F1=0
        assert m["B"]["precision"] == pytest.approx(0.0)
        assert m["B"]["recall"] == pytest.approx(0.0)

    def test_b_block_index_offset(self):
        # B-block columns are indices [d, d+p)
        d, p = 2, 2
        # Row support: {0} in A, {2} in B (column index 2 = first B column)
        A_est = np.array([[1.0, 0.0], [0.0, 1.0]])
        B_est = np.array([[1.0, 0.0], [0.0, 1.0]])
        true_supports = [
            {0, 2},
            {1, 3},
        ]  # row 0: A col 0, B col 0; row 1: A col 1, B col 1

        m = support_metrics(A_est, B_est, true_supports, threshold=0.5)
        # Both blocks should be perfectly recovered
        assert m["A"]["f1"] == pytest.approx(1.0)
        assert m["B"]["f1"] == pytest.approx(1.0)

    def test_block_with_empty_supports_is_skipped(self):
        # Row 0 has no B support; B metrics should still be computable
        # from row 1, not contaminated by row 0's "empty match empty" pair
        A_est = np.array([[1.0, 0.0], [0.0, 0.0]])
        B_est = np.array([[0.0], [1.0]])
        true_supports = [{0}, {1, 2}]  # row 0: A only; row 1: A and B

        m = support_metrics(A_est, B_est, true_supports, threshold=0.5)
        # B block evaluated only on row 1 (row 0 skipped: both est and true empty)
        # Row 1: B_est = {2} (B_est[1, 0] = 1.0, abs index 2), B_true = {2}
        assert m["B"]["precision"] == pytest.approx(1.0)
        assert m["B"]["recall"] == pytest.approx(1.0)

    def test_block_with_no_data_returns_nan(self):
        # All rows have only A-block support; B-block should be NaN
        A_est = np.array([[1.0, 0.0]])
        B_est = np.zeros((1, 2))
        true_supports = [{0}]

        m = support_metrics(A_est, B_est, true_supports, threshold=0.5)
        assert np.isnan(m["B"]["precision"])
        assert np.isnan(m["B"]["recall"])
        assert np.isnan(m["B"]["f1"])



class TestRestrictedGramMinEigenvalue:
    def test_orthonormal_columns_give_one(self):
        # If Z has columns satisfying Z^T Z / N = I on the support,
        # the min eigenvalue is 1.
        N = 100
        # use random orthonormal-ish columns scaled to give Z^T Z / N = I
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((N, 3))
        # Whiten so that Z^T Z / N = I
        gram = Z.T @ Z / N
        L = np.linalg.cholesky(gram)
        Z = Z @ np.linalg.inv(L.T)
        # Verify
        assert np.allclose(Z.T @ Z / N, np.eye(3), atol=1e-10)

        true_supports = [{0, 1}, {1, 2}]
        min_eig = restricted_gram_min_eigenvalue(Z, true_supports)
        assert min_eig == pytest.approx(1.0, abs=1e-10)

    def test_singular_columns_give_near_zero(self):
        # If two columns are duplicated, the gram is rank-deficient,
        # so the min eigenvalue (over their indices) is ~0.
        N = 100
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((N, 3))
        Z[:, 1] = Z[:, 0]  # column 1 duplicates column 0

        # Support including both duplicate columns
        true_supports = [{0, 1}, {2}]
        min_eig = restricted_gram_min_eigenvalue(Z, true_supports)
        # Should be near zero (rank-deficient gram on {0, 1})
        assert min_eig == pytest.approx(0.0)

    def test_returns_minimum_over_rows(self):
        # Construct a Z where one support has small eigenvalue and another has large.
        N = 100
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((N, 4))
        Z[:, 3] *= 0.01  # Make column 3 very small

        true_supports = [{0, 1}, {3}]  # second support has tiny scale
        result = restricted_gram_min_eigenvalue(Z, true_supports)

        # Manually compute for support {3}: ||Z[:, 3]||^2 / N
        expected_min = np.sum(Z[:, 3] ** 2) / N
        assert result == pytest.approx(expected_min)

    def test_empty_support_is_skipped(self):
        # An empty support set should not crash; min over remaining rows.
        N = 50
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((N, 3))
        true_supports = [set(), {0, 1}]  # first row has empty support
        # Should compute eigenvalue only for second row
        result = restricted_gram_min_eigenvalue(Z, true_supports)
        assert np.isfinite(result)
        assert result >= 0


class TestRegressorEnergyBound:
    def test_matches_max_column_energy(self):
        Z = np.array(
            [
                [3, 5, 8],
                [4, 12, 15],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=float,
        )
        # Column norms: sqrt(25/4), sqrt(169/4), sqrt(289/4) -> max = 17/2
        assert regressor_energy_bound(Z) == pytest.approx(17 / 2)

    def test_zero_rows_returns_zero(self):
        Z = np.empty((0, 3))  # zero rows
        assert regressor_energy_bound(Z) == 0.0



class TestClosedLoopSpectralAbscissa:
    def test_no_solution_returns_nan(self):
        solver = MagicMock()
        solver.solution = None
        result = closed_loop_spectral_abscissa(
            A_true=np.eye(2), B_true=np.eye(2), dre_solver=solver, B_est=np.eye(2)
        )
        assert all(np.isnan(v) for v in result.values())

    def test_zero_gain_recovers_open_loop(self):
        # If K = 0, A_cl = A_true, so spectral abscissa = max Re lambda(A_true).
        A_true = np.diag([-1.0, -2.0])  # eigenvalues -1, -2
        B_true = np.eye(2)
        B_est = np.eye(2)

        # Mock K(t, B) = 0 always
        solver = make_mock_dre_solver(
            T=1.0, dt=0.1, K_func=lambda t, B: np.zeros((B.shape[1], A_true.shape[0]))
        )
        result = closed_loop_spectral_abscissa(A_true, B_true, solver, B_est)
        # Both time points should give max Re eig = -1
        for v in result.values():
            assert v == pytest.approx(-1.0)

    def test_stabilising_gain_makes_abscissa_negative(self):
        # A_true = I (unstable), B_true = I, K = -2I
        # Then A_cl = A_true + B_true K = I + (-2I) = -I, abscissa = -1
        A_true = np.eye(2)
        B_true = np.eye(2)
        B_est = np.eye(2)

        solver = make_mock_dre_solver(
            T=1.0,
            dt=0.1,
            K_func=lambda t, B: -2.0 * np.eye(B.shape[1], A_true.shape[0]),
        )
        result = closed_loop_spectral_abscissa(A_true, B_true, solver, B_est)
        for v in result.values():
            assert v == pytest.approx(-1.0)

    def test_evaluates_at_two_time_points_by_default(self):
        A_true = np.eye(2)
        B_true = np.eye(2)
        solver = make_mock_dre_solver(T=1.0, dt=0.1)
        result = closed_loop_spectral_abscissa(A_true, B_true, solver, np.eye(2))
        assert len(result) == 2

    def test_custom_t_values(self):
        A_true = np.eye(2)
        B_true = np.eye(2)
        solver = make_mock_dre_solver(T=1.0, dt=0.1)
        result = closed_loop_spectral_abscissa(
            A_true, B_true, solver, np.eye(2), t_values=[0.0, 0.5, 0.9]
        )
        assert len(result) == 3


# ---------------------------------------------------------------------
# basin_entry_episode
# ---------------------------------------------------------------------


class TestBasinEntryEpisode:
    def test_monotone_decreasing_trajectory(self):
        traj = [1.0, 0.5, 0.25, 0.10, 0.05, 0.02]
        result = basin_entry_episode(traj, thresholds=(0.30, 0.10, 0.05))
        # threshold 0.30: idx 0=1.0, idx 1=0.5, idx 2=0.25 (first <= 0.30)
        assert result[0.30] == 2
        # threshold 0.10: first idx with value <= 0.10 is idx 3
        assert result[0.10] == 3
        # threshold 0.05: first idx with value <= 0.05 is idx 4
        assert result[0.05] == 4

    def test_threshold_uses_le_not_lt(self):
        # Boundary: error == threshold should count as "entered"
        traj = [0.20, 0.15, 0.15, 0.10]
        result = basin_entry_episode(traj, thresholds=(0.15,))
        assert result[0.15] == 1  # idx 1 has 0.15 <= 0.15

    def test_never_reached_returns_none(self):
        traj = [1.0, 0.9, 0.8]
        result = basin_entry_episode(traj, thresholds=(0.5,))
        assert result[0.5] is None

    def test_already_in_basin(self):
        # Trajectory starts below threshold
        traj = [0.05, 0.04, 0.03]
        result = basin_entry_episode(traj, thresholds=(0.10,))
        assert result[0.10] == 0

    def test_multiple_thresholds(self):
        traj = [1.0, 0.5, 0.2, 0.05]
        result = basin_entry_episode(traj, thresholds=(0.5, 0.2, 0.05))
        assert result[0.5] == 1
        assert result[0.2] == 2
        assert result[0.05] == 3

    def test_empty_trajectory(self):
        result = basin_entry_episode([], thresholds=(0.1,))
        assert result[0.1] is None

    def test_non_monotone_picks_first_crossing(self):
        # Even if the trajectory bounces back up, it should pick the first crossing
        traj = [1.0, 0.05, 0.5, 0.5, 0.04]
        result = basin_entry_episode(traj, thresholds=(0.1,))
        assert result[0.1] == 1


# ---------------------------------------------------------------------
# episode_cost
# ---------------------------------------------------------------------


class TestEpisodeCost:
    def test_zero_state_zero_control_gives_zero(self):
        H, d, p = 5, 2, 1
        xs = np.zeros((H, d))
        us = np.zeros((H, p))
        Q, R = np.eye(d), np.eye(p)
        assert episode_cost(xs, us, Q, R, dt=0.1) == 0.0

    def test_identity_cost_matches_squared_norms(self):
        H, d, p = 3, 2, 2
        xs = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        us = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        Q, R = np.eye(d), np.eye(p)
        dt = 0.5

        # Expected: sum (||x||^2 + ||u||^2) * dt = (1+1) + (1+1) + (2+0) = 6
        # Times 0.5 = 3.0
        result = episode_cost(xs, us, Q, R, dt)
        assert result == pytest.approx(3.0)

    def test_diagonal_Q_weighting(self):
        # Q diagonal with different weights
        H, d, p = 1, 2, 1
        xs = np.array([[1.0, 1.0]])
        us = np.zeros((H, p))
        Q = np.diag([2.0, 3.0])
        R = np.eye(p)

        # x^T Q x = 2*1 + 3*1 = 5
        # cost = 5 * dt
        result = episode_cost(xs, us, Q, R, dt=0.1)
        assert result == pytest.approx(0.5)

    def test_accepts_lists_and_arrays(self):
        H, d, p = 2, 1, 1
        xs_list = [np.array([1.0]), np.array([1.0])]
        us_list = [np.array([0.5]), np.array([0.5])]
        Q, R = np.eye(d), np.eye(p)

        cost_list = episode_cost(xs_list, us_list, Q, R, dt=1.0)
        cost_array = episode_cost(np.array(xs_list), np.array(us_list), Q, R, dt=1.0)
        assert cost_list == pytest.approx(cost_array)

    def test_dt_scaling(self):
        # Cost scales linearly in dt
        H, d, p = 2, 1, 1
        xs = np.ones((H, d))
        us = np.ones((H, p))
        Q, R = np.eye(d), np.eye(p)
        c1 = episode_cost(xs, us, Q, R, dt=1.0)
        c2 = episode_cost(xs, us, Q, R, dt=2.0)
        assert c2 == pytest.approx(2.0 * c1)


# ---------------------------------------------------------------------
# collect_diagnostics
# ---------------------------------------------------------------------


class TestCollectDiagnostics:
    def _make_agent(self, A_est, B_est, dre_solver=None):
        agent = MagicMock()
        agent.A_est = A_est
        agent.B_est = B_est
        agent.dre = dre_solver if dre_solver else make_mock_dre_solver()
        return agent

    def test_returns_empty_when_no_estmate(self):
        agent = MagicMock()
        agent.A_est = None
        result = collect_diagnostics(
            agent,
            buffer=None,
            A_true=np.eye(2),
            B_true=np.eye(2),
            true_supports=[{0}, {1}],
            Q=np.eye(2),
            R=np.eye(2),
        )
        assert result == {}

    def test_returns_all_expected_keys(self):
        d, p = 2, 1
        A_true = np.eye(d)
        B_true = np.ones((d, p))
        Z = np.random.default_rng(0).standard_normal((20, d + p))
        Y = np.random.default_rng(1).standard_normal((20, d))
        buf = make_buffer(Z, Y)

        agent = self._make_agent(A_true, B_true)
        true_supports = [{0, 1}, {1, 2}]

        result = collect_diagnostics(
            agent, buf, A_true, B_true, true_supports, np.eye(d), np.eye(p)
        )

        expected_keys = {
            "error_joint",
            "error_A",
            "error_B",
            "support_precision",
            "support_recall",
            "support_f1",
            "gram_min_eig",
            "regressor_energy",
            "spectral_abscissa_t0",
            "spectral_abscissa_tT",
        }
        assert set(result.keys()) == expected_keys

    def test_perfect_estmate_has_zero_error(self):
        d, p = 2, 1
        A_true = np.eye(d)
        B_true = np.ones((d, p))
        Z = np.random.default_rng(0).standard_normal((20, d + p))
        Y = np.zeros((20, d))
        buf = make_buffer(Z, Y)

        agent = self._make_agent(A_true.copy(), B_true.copy())
        true_supports = [{0}, {1}]

        result = collect_diagnostics(
            agent, buf, A_true, B_true, true_supports, np.eye(d), np.eye(p)
        )
        assert result["error_joint"] == pytest.approx(0.0, abs=1e-15)
        assert result["error_A"] == pytest.approx(0.0, abs=1e-15)
        assert result["error_B"] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------
# Run as module
# ---------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
