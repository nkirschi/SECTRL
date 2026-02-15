import numpy as np
from common import SystemConfig


class ContinuousLQREnv:
    """
    Environment simulating continuous-time stochastic linear dynamics:
    x' = (Ax + Bu) dt + Σ dw

    The simulation method is Euler–Maruyama like in Basei et al.
    """

    def __init__(
        self,
        A: np.array,
        B: np.array,
        Σ: np.array,
        x0: np.array,
        config: SystemConfig,
    ):
        assert A.ndim == 2 and B.ndim == 2 and Σ.ndim == 2 and x0.ndim == 1
        assert A.shape[0] == config.x_dim
        assert A.shape[1] == config.x_dim
        assert B.shape[0] == config.x_dim
        assert B.shape[1] == config.u_dim
        assert Σ.shape[0] == config.x_dim
        assert Σ.shape[1] == config.x_dim
        assert x0.shape[0] == config.x_dim

        self.A = A
        self.B = B
        self.Σ = Σ
        self.x0 = x0.copy()
        self.config = config
        self.reset()

    def reset(self):
        self.t = 0.0
        self.x = self.x0.copy()
        return self.x.copy()

    def step(self, u, noise_override=None):
        assert u.ndim == 1
        assert u.shape[0] == self.config.u_dim

        if noise_override is None:
            noise = np.random.normal(size=self.config.x_dim)
        else:
            assert noise_override.shape == self.x.shape
            noise = noise_override

        # Continuous evolution using Euler-Maruyama method
        drift = self.A @ self.x + self.B @ u
        dt = self.config.dt
        dW = noise * np.sqrt(dt)
        dx = drift * dt + self.Σ @ dW
        self.x += dx
        self.t += dt
        return self.x.copy(), dx, dt
