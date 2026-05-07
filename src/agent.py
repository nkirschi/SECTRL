"""
Agents for episodic continuous-time LQ control.

All learning agents use DRE-based feedback (not CARE).
"""

import numpy as np
from planner import RiccatiODESolver
from estimator import DiscreteRidgeEstimator, RowLassoEstimator


class Agent:
    """Base class for all agents."""

    def __init__(self, config, Q, R):
        """
        Parameters
        ----------
        config : SystemConfig
        Q : ndarray (d, d), state cost
        R : ndarray (p, p), control cost
        """
        self.config = config
        self.Q = Q
        self.R = R
        self.dre = RiccatiODESolver(config, Q, R)

        # Current estimated parameters
        self.A_est = None
        self.B_est = None

        # Previous gain (for fallback on DRE failure)
        self._prev_dre_solution = None
        self._prev_B_est = None
        self._dre_valid = False

    def get_control(self, t, x):
        """
        Return control action u(t) given state x(t).

        Parameters
        ----------
        t : float
            Current time within the episode.
        x : ndarray, shape (d,)
            Current state.

        Returns
        -------
        u : ndarray, shape (p,)
        """
        raise NotImplementedError

    def update(self, buffer):
        """
        Update parameter estimate and recompute DRE gain.

        Parameters
        ----------
        buffer : RegressionBuffer
        """
        raise NotImplementedError

    def _solve_dre(self, A, B):
        """
        Solve DRE and handle failures gracefully.

        Returns True if solve succeeded, False if fell back to previous.
        """
        try:
            sol = self.dre.solve(A, B)

            # Check for negative eigenvalues in P at any grid point
            for tau_val in sol.t:
                P = sol.sol(tau_val).reshape(self.config.x_dim, self.config.x_dim)
                eigs = np.linalg.eigvalsh(P)
                if np.min(eigs) < -1e-8:
                    raise ValueError(
                        f"Negative P eigenvalue {np.min(eigs):.2e} at tau={tau_val:.4f}"
                    )

            # Success: store as fallback
            self._prev_dre_solution = sol
            self._prev_B_est = B.copy()
            self._dre_valid = True
            return True

        except Exception:
            # Revert to previous gain schedule
            if self._prev_dre_solution is not None:
                self.dre.solution = self._prev_dre_solution
            self._dre_valid = False
            return False

    def _get_feedback(self, t, x):
        """
        Compute feedback u = -K(t) x using the current DRE solution.

        Uses the B estimate that was used to solve the DRE (since
        K(t) = R^{-1} B^T P(t) depends on B).
        """
        B = self._prev_B_est if not self._dre_valid else self.B_est
        if B is None:
            return np.zeros(self.config.u_dim)

        K = self.dre.get_K(t, B)  # K = -R^{-1} B^T P(t), already negated
        u = K @ x  # u = -K(t) x (K already has the sign)
        return u


class OracleAgent(Agent):
    """
    Uses true (A_true, B_true). Solves DRE once at construction.
    """

    def __init__(self, config, Q, R, A_true, B_true):
        super().__init__(config, Q, R)
        self.A_est = A_true.copy()
        self.B_est = B_true.copy()
        self._solve_dre(A_true, B_true)

    def get_control(self, t, x):
        return self._get_feedback(t, x)

    def update(self, buffer):
        pass  # Oracle never updates


class DenseGreedyAgent(Agent):
    """
    Ridge regression estimator + DRE feedback. No excitation.
    """

    def __init__(self, config, Q, R, A_init, B_init, mu=0.01):
        super().__init__(config, Q, R)
        self.estimator = DiscreteRidgeEstimator(config.x_dim, config.u_dim, mu=mu)

        # Initialise with prior
        self.A_est = A_init.copy()
        self.B_est = B_init.copy()
        self._solve_dre(A_init, B_init)

    def get_control(self, t, x):
        return self._get_feedback(t, x)

    def update(self, buffer):
        A_est, B_est = self.estimator.estimate(buffer)
        self.A_est = A_est
        self.B_est = B_est
        self._solve_dre(A_est, B_est)


class SparseGreedyAgent(Agent):
    """
    Row-wise Lasso estimator + DRE feedback. No excitation.
    """

    def __init__(
        self,
        config,
        Q,
        R,
        A_init,
        B_init,
        lambda_fixed=None,
        sigma_bar=None,
        c_lambda=2.0,
        delta=0.05,
        max_episodes=None,
    ):
        super().__init__(config, Q, R)
        self.estimator = RowLassoEstimator(
            x_dim=config.x_dim,
            u_dim=config.u_dim,
            lambda_fixed=lambda_fixed,
            sigma_bar=sigma_bar,
            c_lambda=c_lambda,
            delta=delta,
            max_episodes=max_episodes,
        )

        # Initialise with prior
        self.A_est = A_init.copy()
        self.B_est = B_init.copy()
        self._solve_dre(A_init, B_init)

    def get_control(self, t, x):
        return self._get_feedback(t, x)

    def update(self, buffer):
        A_est, B_est = self.estimator.estimate(buffer)
        self.A_est = A_est
        self.B_est = B_est
        self._solve_dre(A_est, B_est)


class SparseExcitationAgent(Agent):
    """
    Row-wise Lasso estimator + DRE feedback + additive excitation.

    The control is u(t) = -K(t) x(t) + eta(t), where
    eta(t) ~ N(0, sigma_u^2 I_p) is independent excitation noise.
    """

    def __init__(
        self,
        config,
        Q,
        R,
        A_init,
        B_init,
        sigma_u=0.1,
        excitation_rng=None,
        lambda_fixed=None,
        sigma_bar=None,
        c_lambda=2.0,
        delta=0.05,
        max_episodes=None,
    ):
        super().__init__(config, Q, R)
        self.estimator = RowLassoEstimator(
            x_dim=config.x_dim,
            u_dim=config.u_dim,
            lambda_fixed=lambda_fixed,
            sigma_bar=sigma_bar,
            c_lambda=c_lambda,
            delta=delta,
            max_episodes=max_episodes,
        )
        self.sigma_u = sigma_u
        self._exc_rng = excitation_rng or np.random.RandomState(0)

        # Initialise with prior
        self.A_est = A_init.copy()
        self.B_est = B_init.copy()
        self._solve_dre(A_init, B_init)

    def get_control(self, t, x):
        u_fb = self._get_feedback(t, x)
        eta = self._exc_rng.randn(self.config.u_dim) * self.sigma_u
        return u_fb + eta

    def update(self, buffer):
        A_est, B_est = self.estimator.estimate(buffer)
        self.A_est = A_est
        self.B_est = B_est
        self._solve_dre(A_est, B_est)
