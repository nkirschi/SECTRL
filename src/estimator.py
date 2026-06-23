from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import warnings
from sklearn.linear_model._cd_fast import enet_coordinate_descent_gram
from sklearn.exceptions import ConvergenceWarning
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

    Each of the d rows is an independent Lasso sharing the same design Z. All
    quantities the coordinate descent needs -- the Gram Z^T Z, the cross terms
    Z^T Y, and the targets' squared norms -- are accumulated incrementally from
    each episode's new rows, so the per-episode cost is independent of the total
    sample count N. The Cython gram solver is driven directly (it is exactly the
    routine sklearn.Lasso(precompute=Gram) calls internally), warm-started from
    the previous episode's coefficients.
    """

    def __init__(
        self,
        sys_cfg: SystemConfig,
        est_cfg: EstimatorConfig,
        lamda_schedule: callable = None,
        use_warmup: bool = True,
    ):
        self.x_dim = sys_cfg.d
        self.z_dim = sys_cfg.d + sys_cfg.p
        self.lamda_fixed = est_cfg.lambda_lasso
        self.lamda_schedule = lamda_schedule
        self.lamda_warmup = est_cfg.lambda_warmup
        self.use_warmup = use_warmup
        self.max_iter = est_cfg.lasso_max_iter
        self.tol = est_cfg.lasso_tol
        self._fitted_once = False
        # Incremental accumulators (unnormalised sums over all rows seen).
        self._gram = np.zeros((self.z_dim, self.z_dim), dtype=np.float64)  # Z^T Z
        self._rhs = np.zeros((self.x_dim, self.z_dim), dtype=np.float64)   # (Z^T Y)^T
        self._yy = np.zeros(self.x_dim, dtype=np.float64)                  # sum y_i^2
        self._n_seen: int = 0
        # Warm-start coefficients carried across episodes (one row each).
        self._coef = np.zeros((self.x_dim, self.z_dim), dtype=np.float64)
        self._rng = np.random.RandomState(0)  # required by the Cython CD (cyclic)

    def fit(
        self, z_data: NDArray[np.float64], y_data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        n_total = z_data.shape[0]
        if n_total > self._n_seen:
            z_new = z_data[self._n_seen :]
            y_new = y_data[self._n_seen :]
            self._gram += z_new.T @ z_new
            self._rhs += (z_new.T @ y_new).T
            self._yy += np.einsum("ij,ij->j", y_new, y_new)
            self._n_seen = n_total

        is_warmup = (
            self.use_warmup and self.lamda_fixed is None and not self._fitted_once
        )
        if self.lamda_fixed is not None:
            lamda = self.lamda_fixed
        elif is_warmup:
            lamda = self.lamda_warmup
        else:
            lamda = self.lamda_schedule(self._n_seen)
        self._fitted_once = True

        # The Cython gram solver minimises (1/2) w^T G w - q^T w + alpha ||w||_1,
        # so the L1 weight is the sample count times the (normalised) lambda.
        alpha = self._n_seen * lamda
        with warnings.catch_warnings():
            if is_warmup:
                # The tiny warmup penalty is near-OLS on a degenerate early
                # design; CD is not expected to fully converge, and a rough
                # nonzero estimate is all we need to break the trap.
                warnings.simplefilter("ignore", ConvergenceWarning)
            for i in range(self.x_dim):
                # y enters the solver only through ||y||^2 (the dual-gap
                # tolerance), so a length-1 vector with the right norm suffices.
                y_norm = np.array([np.sqrt(self._yy[i])], dtype=np.float64)
                w, _, _, _ = enet_coordinate_descent_gram(
                    self._coef[i], alpha, 0.0, self._gram,
                    np.ascontiguousarray(self._rhs[i]), y_norm,
                    self.max_iter, self.tol, self._rng, 0, 0,
                )
                self._coef[i] = w

        return self._coef.copy()
