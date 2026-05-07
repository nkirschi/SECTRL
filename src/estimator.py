import numpy as np
from scipy.linalg import solve


class RegressionBuffer:
    """
    Pre-allocated buffer for discrete regression samples (z_k, y_k).

    The regression model is  y_k = Theta_star @ z_k + eps_k  where
        y_k = (x_{k+1} - x_k) / dt,
        z_k = [x_k; u_k],
        eps_k ~ N(0, sigma_bar^2 I_d),  sigma_bar = sigma / sqrt(dt).

    Parameters
    ----------
    x_dim : int
        State dimension d.
    u_dim : int
        Control dimension p.
    max_episodes : int
        Maximum number of episodes M (determines pre-allocation size).
    steps_per_episode : int
        Steps per episode H.
    """

    def __init__(self, x_dim, u_dim, max_episodes, steps_per_episode):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = x_dim + u_dim
        self.H = steps_per_episode
        self.capacity = max_episodes * steps_per_episode

        # Pre-allocate arrays
        self._Z = np.zeros((self.capacity, self.z_dim))
        self._Y = np.zeros((self.capacity, x_dim))
        self._N = 0  # current number of samples stored
        self._episodes = 0

    @property
    def N(self):
        """Total number of samples currently stored."""
        return self._N

    @property
    def episodes(self):
        """Number of episodes added."""
        return self._episodes

    @property
    def Z(self):
        """Design matrix (N x (d+p)), only the filled rows."""
        return self._Z[: self._N]

    @property
    def Y(self):
        """Response matrix (N x d), only the filled rows."""
        return self._Y[: self._N]

    def add_episode(self, zs, ys):
        """
        Append one episode of samples to the buffer.

        Parameters
        ----------
        zs : ndarray, shape (H, d+p)
            Stacked regressors z_k = [x_k; u_k] for each step.
        ys : ndarray, shape (H, d)
            Regression targets y_k = (x_{k+1} - x_k) / dt for each step.
        """
        H = zs.shape[0]
        assert zs.shape == (H, self.z_dim), (
            f"Expected zs shape ({H}, {self.z_dim}), got {zs.shape}"
        )
        assert ys.shape == (H, self.x_dim), (
            f"Expected ys shape ({H}, {self.x_dim}), got {ys.shape}"
        )

        start = self._N
        end = start + H
        assert end <= self.capacity, (
            f"Buffer overflow: {end} > {self.capacity}. Increase max_episodes."
        )

        self._Z[start:end] = zs
        self._Y[start:end] = ys
        self._N = end
        self._episodes += 1


class DiscreteRidgeEstimator:
    """
    Discrete-time ridge regression estimator (dense baseline).

    Solves  Theta_hat = argmin (1/2N)||Y - Z Theta^T||_F^2 + (mu/2)||Theta||_F^2

    in closed form:  Theta^T = (Z^T Z + mu N I)^{-1} Z^T Y.

    Parameters
    ----------
    x_dim : int
        State dimension d.
    u_dim : int
        Control dimension p.
    mu : float
        Ridge regularisation parameter (default 0.01).
    """

    def __init__(self, x_dim, u_dim, mu=0.01):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = x_dim + u_dim
        self.mu = mu

    def estimate(self, buffer):  # TODO remove buffer dependence and pass Z, Y directly
        """
        Estimate (A_hat, B_hat) from the data in buffer.

        Parameters
        ----------
        buffer : RegressionBuffer
            Must have at least one episode of data.

        Returns
        -------
        A_hat : ndarray, shape (d, d)
        B_hat : ndarray, shape (d, p)
        """
        if buffer.N == 0:
            raise RuntimeError("Buffer is empty")

        Z = buffer.Z  # (N, d+p)
        Y = buffer.Y  # (N, d)
        N = buffer.N

        ZtZ = Z.T @ Z  # (d+p, d+p)
        ZtZ = (ZtZ + ZtZ.T) / 2.0  # symmetrise for numerics
        ZtY = Z.T @ Y  # (d+p, d)

        LHS = ZtZ + self.mu * N * np.eye(self.z_dim)

        try:
            Theta_T = solve(LHS, ZtY, assume_a="pos")  # (d+p, d)
        except np.linalg.LinAlgError:
            Theta_T = solve(LHS, ZtY)

        Theta = Theta_T.T  # (d, d+p)

        A_hat = Theta[:, : self.x_dim]
        B_hat = Theta[:, self.x_dim :]

        return A_hat, B_hat


class RowLassoEstimator:
    """
    Row-wise Lasso estimator for sparse Theta = [A B].

    Solves d independent Lasso problems, one per row i of Theta:
        theta_hat_i = argmin (1/2N)||Y_i - Z theta||_2^2 + lambda_m ||theta||_1

    Supports two regularisation modes:
        - Fixed: lambda_m = lambda_fixed for all m.
        - Theoretical schedule: lambda_m = c_lambda * sigma_bar
              * sqrt(log((d+p)*M*d/delta) / N_m)

    Warm-starts each Lasso from the previous episode's solution.

    Parameters
    ----------
    x_dim : int
        State dimension d.
    u_dim : int
        Control dimension p.
    lambda_fixed : float or None
        If given, use this fixed regularisation for all episodes.
    sigma_bar : float or None
        Regression noise std (sigma / sqrt(dt)). Required for
        the theoretical schedule.
    c_lambda : float
        Constant in the theoretical schedule (default 2.0).
    delta : float
        Failure probability for the theoretical schedule (default 0.05).
    max_episodes : int
        Total number of episodes M (for the log factor). Required for
        the theoretical schedule.
    max_iter : int
        Maximum iterations for the Lasso solver (default 5000).
    tol : float
        Convergence tolerance for the Lasso solver (default 1e-6).
    """

    def __init__(
        self,
        x_dim,
        u_dim,
        lambda_fixed=None,
        sigma_bar=None,
        c_lambda=2.0,
        delta=0.05,
        max_episodes=None,
        max_iter=5000,
        tol=1e-4,
    ):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = x_dim + u_dim
        self.lambda_fixed = lambda_fixed
        self.sigma_bar = sigma_bar
        self.c_lambda = c_lambda
        self.delta = delta
        self.max_episodes = max_episodes
        self.max_iter = max_iter
        self.tol = tol

        # Validate: must have either fixed lambda or schedule parameters
        if lambda_fixed is None:
            if sigma_bar is None or max_episodes is None:
                raise ValueError(
                    "Either lambda_fixed or (sigma_bar, max_episodes) must be provided."
                )

        # Warm-start coefficients: one vector per row of Theta
        self._warm_coefs = [np.zeros(self.z_dim) for _ in range(x_dim)]

    def _get_lambda(self, N):
        """Compute regularisation parameter for current sample size N."""
        if self.lambda_fixed is not None:
            return self.lambda_fixed

        d = self.x_dim
        p = self.u_dim
        M = self.max_episodes
        log_term = np.log((d + p) * M * d / self.delta)
        return self.c_lambda * self.sigma_bar * np.sqrt(log_term / N)

    def estimate(self, buffer):
        """
        Estimate (A_hat, B_hat) from the data in buffer.

        Solves d independent Lasso problems with warm-starting.

        Parameters
        ----------
        buffer : RegressionBuffer
            Must have at least one episode of data.

        Returns
        -------
        A_hat : ndarray, shape (d, d)
        B_hat : ndarray, shape (d, p)
        """
        from sklearn.linear_model import Lasso

        if buffer.N == 0:
            raise RuntimeError("Buffer is empty")

        Z = buffer.Z  # (N, d+p)
        Y = buffer.Y  # (N, d)
        N = buffer.N

        alpha = self._get_lambda(N)

        Theta = np.zeros((self.x_dim, self.z_dim))

        for i in range(self.x_dim):
            # sklearn Lasso: min (1/2N)||y - X w||^2 + alpha ||w||_1
            lasso = Lasso(
                alpha=alpha,
                fit_intercept=False,
                warm_start=True,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            # Warm-start from previous episode's solution
            lasso.coef_ = self._warm_coefs[i].copy()

            lasso.fit(Z, Y[:, i])

            Theta[i, :] = lasso.coef_
            self._warm_coefs[i] = lasso.coef_.copy()

        A_hat = Theta[:, : self.x_dim]
        B_hat = Theta[:, self.x_dim :]

        return A_hat, B_hat
