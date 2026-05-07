from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class SystemConfig:
    x_dim: int
    u_dim: int
    T: float
    dt: float

    @property
    def H(self):
        """Steps per episode."""
        return int(round(self.T / self.dt))


@dataclass(frozen=True)
class ExperimentConfig:
    """Full configuration for one experimental benchmark."""

    # Dimensions
    x_dim: int
    u_dim: int
    sparsity: int  # s: max nonzeros per row of [A B]

    # Time
    T: float = 1.0  # episode horizon tau
    dt: float = 0.025  # Euler-Maruyama step size

    # Episodes and seeds
    max_episodes: int = 100
    n_seeds: int = 50

    # Noise
    sigma: float = 0.5  # diffusion coefficient

    # Estimation
    lambda_lasso: float = None  # fixed lambda (if None, use schedule)
    c_lambda: float = 2.0  # constant in theoretical schedule
    delta: float = 0.05  # failure probability
    mu_ridge: float = 0.01  # ridge regularisation

    # Excitation
    sigma_u: float = 0.1  # excitation noise std

    # Diagnostics
    basin_thresholds: tuple = (0.05, 0.10, 0.15, 0.20, 0.30)
    support_threshold: float = 0.05  # threshold for support recovery

    @property
    def H(self):
        """Steps per episode."""
        return int(round(self.T / self.dt))

    @property
    def sigma_bar(self):
        """Regression noise std: sigma / sqrt(dt)."""
        return self.sigma / np.sqrt(self.dt)

    @property
    def system_config(self):
        return SystemConfig(
            x_dim=self.x_dim,
            u_dim=self.u_dim,
            T=self.T,
            dt=self.dt,
        )

    @property
    def theoretical_speedup(self):
        """Predicted basin-entry speedup: (d+p) / (s * log(d+p))."""
        dp = self.x_dim + self.u_dim
        return dp / (self.sparsity * np.log(dp))

    @property
    def theoretical_lambda_schedule(self):
        def get_lambda(N):
            d = self.x_dim
            p = self.u_dim
            M = self.max_episodes
            log_term = np.log((d + p) * M * d / self.delta)
            return self.c_lambda * self.sigma_bar * np.sqrt(log_term / N)

        return [get_lambda(m).item() for m in range(1, self.max_episodes)]
