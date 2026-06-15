from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Lasso
from common import EstimatorConfig, SystemConfig


class RegressionBuffer:
    """Pre-allocated buffer for discrete regression samples."""

    def __init__(
        self, x_dim: int, u_dim: int, max_episodes: int, steps_per_episode: int
    ):
        self.x_dim = x_dim
        self.z_dim = x_dim + u_dim
        self.capacity = max_episodes * steps_per_episode
        self._Z = np.zeros((self.capacity, self.z_dim), dtype=np.float64)
        self._Y = np.zeros((self.capacity, x_dim), dtype=np.float64)
        self._N = 0

    @property
    def N(self) -> int:
        return self._N

    @property
    def Z(self) -> NDArray[np.float64]:
        return self._Z[: self._N]

    @property
    def Y(self) -> NDArray[np.float64]:
        return self._Y[: self._N]

    def add_episode(self, zs: NDArray[np.float64], ys: NDArray[np.float64]) -> None:
        H = zs.shape[0]
        start = self._N
        end = start + H
        if end > self.capacity:
            raise RuntimeError("Buffer overflow. Increase max_episodes.")
        self._Z[start:end] = zs
        self._Y[start:end] = ys
        self._N = end


class DiscreteRidgeEstimator:
    """
    Discrete-time ridge regression estimator (dense baseline).
    """

    def __init__(self, est_cfg: EstimatorConfig, lamda_schedule: callable = None):
        self.lamda_fixed = est_cfg.mu_ridge
        self.lamda_schedule = lamda_schedule
        # Incremental accumulators (unnormalised sums)
        self._gram: NDArray[np.float64] | None = None  # Z^T Z
        self._rhs: NDArray[np.float64] | None = None  # Z^T Y
        self._n_seen: int = 0

    def fit(
        self, z_data: NDArray[np.float64], y_data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        n_total = z_data.shape[0]
        if n_total > self._n_seen:
            z_new = z_data[self._n_seen :]
            y_new = y_data[self._n_seen :]
            G_new = z_new.T @ z_new
            R_new = z_new.T @ y_new
            if self._gram is None:
                self._gram = G_new
                self._rhs = R_new
            else:
                self._gram += G_new
                self._rhs += R_new
            self._n_seen = n_total

        n_features = self._gram.shape[0]
        lamda = (
            self.lamda_schedule(self._n_seen)
            if self.lamda_fixed is None
            else self.lamda_fixed
        )
        lhs = self._gram / self._n_seen + lamda * np.eye(n_features)
        rhs = self._rhs / self._n_seen

        try:
            theta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        return theta.T.astype(np.float64)


class RowLassoEstimator:
    """
    Row-wise Lasso estimator for row-sparse parameters.
    """

    def __init__(
        self,
        sys_cfg: SystemConfig,
        est_cfg: EstimatorConfig,
        lamda_schedule: callable = None,
    ):
        self.x_dim = sys_cfg.d
        self.z_dim = sys_cfg.d + sys_cfg.p
        self.lamda_fixed = est_cfg.lambda_lasso
        self.lamda_schedule = lamda_schedule
        # Incremental Gram accumulator (unnormalised)
        self._gram: NDArray[np.float64] | None = None
        self._n_seen: int = 0
        # Instantiate sklearn models once for warm-start across episodes.
        self.models = [
            Lasso(
                fit_intercept=False,
                warm_start=True,
                max_iter=est_cfg.lasso_max_iter,
                tol=est_cfg.lasso_tol,
            )
            for _ in range(self.x_dim)
        ]

    def fit(
        self, z_data: NDArray[np.float64], y_data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        n_total = z_data.shape[0]
        if n_total > self._n_seen:
            z_new = z_data[self._n_seen :]
            G_new = z_new.T @ z_new
            if self._gram is None:
                self._gram = G_new
            else:
                self._gram += G_new
            self._n_seen = n_total

        lamda = (
            self.lamda_schedule(self._n_seen)
            if self.lamda_fixed is None
            else self.lamda_fixed
        )
        theta_hat = np.zeros((self.x_dim, self.z_dim), dtype=np.float64)
        for i in range(self.x_dim):
            self.models[i].alpha = lamda
            self.models[i].precompute = self._gram
            self.models[i].fit(z_data, y_data[:, i])
            theta_hat[i, :] = self.models[i].coef_

        return theta_hat
