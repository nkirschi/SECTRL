from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor, cho_solve
from common import SystemConfig

class RiccatiODESolver:
    """
    Solves the Matrix Riccati DIFFERENTIAL equation backwards.
    Not to be confused with the Riccati algebraic equation.
    See Basei et al. 2022 equation (2.10)
    """
    def __init__(self, sys_cfg: SystemConfig, Q: NDArray[np.float64], R: NDArray[np.float64]):
        self.sys_cfg = sys_cfg
        self.Q = Q
        self.R = R
        self.R_cho = cho_factor(R)
        self.solution = None

    def solve(self, A: NDArray[np.float64], B: NDArray[np.float64], terminal_cost: NDArray[np.float64] | None = None) -> bool:
        R_inv_B_T = cho_solve(self.R_cho, B.T)
        S = B @ R_inv_B_T
        d = self.sys_cfg.d
        I_d = np.eye(d)

        def riccati_ode(_, p_flat):
            P = p_flat.reshape(d, d)
            dP_dtau = A.T @ P + P @ A - P @ S @ P + self.Q
            return dP_dtau.flatten()

        # Analytical Jacobian of the Riccati RHS w.r.t. vec(P) (row-major)
        J_linear = np.kron(A.T, I_d) + np.kron(I_d, A.T)

        def riccati_jac(_, p_flat):
            P = p_flat.reshape(d, d)
            PS = P @ S
            return J_linear - np.kron(I_d, PS) - np.kron(PS, I_d)

        if terminal_cost is not None:
            p_final = terminal_cost.flatten()
        else:
            p_final = np.zeros(d ** 2)

        sol = solve_ivp(
            riccati_ode,
            t_span=[0, self.sys_cfg.T],
            y0=p_final,
            dense_output=True,
            method="Radau",
            rtol=1e-6,
            atol=1e-9,
            jac=riccati_jac,
        )

        if sol.success:
            self.solution = sol
            return True
        return False

    def get_K(self, t: float, B: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns feedback matrix K(t) = -R^-1 B^T P(t).
        P(t) is retrieved via the dense interpolant.
        """
        if self.solution is None:
            return np.zeros((self.sys_cfg.p, self.sys_cfg.d), dtype=np.float64)

        tau = self.sys_cfg.T - t  # map physics time t -> solver time tau
        P_flat = self.solution.sol(tau)
        P = P_flat.reshape(self.sys_cfg.d, self.sys_cfg.d)
        K = cho_solve(self.R_cho, -B.T @ P)
        return K