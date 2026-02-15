import numpy as np
from scipy.linalg import solve


class ContinuousLeastSquaresEstimator:
    """
    Continuous-time L2-regularized least squares estimator.
    See Basei et al. 2022 equation (2.13)
    """

    def __init__(self, config, λ=1e-12):
        self.config = config
        self.λ = λ
        self.dim = config.x_dim + config.u_dim

        # Accumulators for integrals
        self.integral_ZZ = np.zeros((self.dim, self.dim))  # Integral Z Z^T dt
        self.integral_ZdX = np.zeros((self.dim, config.x_dim))  # Integral Z dX^T
        self.m_episodes = 0

    def add_trajectory(self, trajectory):
        """
        Accumulates data from a single episode trajectory, which is a list of (x, u, dx, dt)
        """
        self.m_episodes += 1

        # vectorised update
        xs, us, dxs, dts = zip(*trajectory)
        X = np.stack(xs)
        U = np.stack(us)
        dX = np.stack(dxs)
        dt = np.array(dts)[:, None]
        Z = np.hstack([X, U])
        self.integral_ZZ += Z.T @ (Z * dt)
        self.integral_ZdX += Z.T @ dX

    def estimate(self):
        if self.m_episodes == 0:
            raise RuntimeError("No trajectories have been added yet")

        int_ZZ = (self.integral_ZZ + self.integral_ZZ.T) / 2.0
        int_ZdX = self.integral_ZdX
        reg = self.λ / self.m_episodes * np.eye(self.dim)

        LHS = int_ZZ + reg  # guaranteed positive definite
        RHS = int_ZdX

        try:
            Theta_T = solve(LHS, RHS, assume_a="pos")
        except np.linalg.LinAlgError:
            # Fallback if numerical issues destroy positive definiteness
            print("Warning: falling back on regular solve as LHS was not pos definite")
            Theta_T = solve(LHS, RHS)

        Theta = Theta_T.T

        A_hat = Theta[:, : self.config.x_dim]
        B_hat = Theta[:, self.config.x_dim :]

        return A_hat, B_hat
