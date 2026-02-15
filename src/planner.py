import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor, cho_solve


class RiccatiODESolver:
    """
    Solves the Matrix Riccati DIFFERENTIAL equation backwards.
    Not to be confused with the Riccati algebraic equation.
    See Basei et al. 2022 equation (2.10)
    """

    def __init__(self, config, Q, R):
        self.config = config
        self.Q = Q
        self.R = R
        self.R_cho = cho_factor(R)  # R is assumed to be positive definite
        self.solution = None

    def solve(self, A, B, terminal_cost=None):
        """
        Solves for P(t) over [0, T] given estimates A_hat, B_hat.
        """
        # Since R is assumed psd, we use Cholesky here
        # R_inv_B_T = solve(self.R, B.T, assume_a="pos")
        R_inv_B_T = cho_solve(self.R_cho, B.T)
        S = B @ R_inv_B_T

        # We solve BACKWARDS from T to 0.
        # Let tau = T - t. dP/dtau = -dP/dt.
        # dP/dtau = A'P + PA - P B R^-1 B' P + Q
        def riccati_ode(_, p_flat):
            P = p_flat.reshape(self.config.x_dim, self.config.x_dim)
            dP_dtau = A.T @ P + P @ A - P @ S @ P + self.Q
            return dP_dtau.flatten()

        p_final = (
            terminal_cost.flatten()  # optional quadratic terminal cost (generalisation)
            if terminal_cost is not None
            else np.zeros(self.config.x_dim**2)
        )

        sol = solve_ivp(
            riccati_ode,
            t_span=[0, self.config.T],  # Integrate from tau=0 (t=T) to tau=T (t=0)
            y0=p_final,
            dense_output=True,
            method="Radau",
            rtol=1e-9,  # High precision relative tolerance
            atol=1e-12,  # High precision absolute tolerance
        )

        self.solution = sol
        return sol

    def get_K(self, t, B):
        """
        Returns feedback matrix K(t) = -R^-1 B^T P(t).
        P(t) is retrieved via the dense interpolant.
        """
        assert 0 <= t <= self.config.T

        if self.solution is None:
            return np.zeros((self.config.u_dim, self.config.x_dim))

        tau = self.config.T - t  # map physics time t -> solver time tau
        P_flat = self.solution.sol(tau)
        P = P_flat.reshape(self.config.x_dim, self.config.x_dim)

        # K = -R^-1 B^T P
        # K = solve(self.R, -B.T @ P, assume_a="pos")
        K = cho_solve(self.R_cho, -B.T @ P)

        return K
