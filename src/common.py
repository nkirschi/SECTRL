from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    x_dim: int
    u_dim: int
    T: float
    dt: float
