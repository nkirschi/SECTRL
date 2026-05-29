"""
Agents for episodic continuous-time LQ control.
All learning agents use the Riccati ODE for planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray

from planner import RiccatiODESolver


@dataclass
class LinearControlAgent:
    name: str
    theta_hat: NDArray[np.float64]
    state_dim: int
    control_dim: int
    Q: NDArray[np.float64]
    R: NDArray[np.float64]
    estimator: Any  # None for oracle
    planner: RiccatiODESolver
    sigma_u: float = 0.0

    @property
    def A_hat(self) -> NDArray[np.float64]:
        return self.theta_hat[:, : self.state_dim]

    @property
    def B_hat(self) -> NDArray[np.float64]:
        return self.theta_hat[:, self.state_dim :]

    def get_control(
        self, t: float, x: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        u = self.planner.get_K(t, self.B_hat) @ x
        if self.sigma_u > 0.0:
            u += rng.standard_normal(self.control_dim) * self.sigma_u
        return u.astype(np.float64)

    def update_model(
        self, z_data: NDArray[np.float64], y_data: NDArray[np.float64]
    ) -> None:
        if self.estimator is not None:
            old_theta_hat = self.theta_hat.copy()
            self.theta_hat = self.estimator.fit(z_data, y_data)
            if not self.planner.solve(self.A_hat, self.B_hat):
                print(f"Warning: Riccati ODE solver failed for agent {self.name}.")
                self.theta_hat = old_theta_hat
