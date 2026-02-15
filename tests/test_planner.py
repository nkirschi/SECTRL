import numpy as np
import pytest

from common import SystemConfig
from planner import RiccatiODESolver


@pytest.fixture
def config():
    return SystemConfig(x_dim=1, u_dim=1, T=5.0, dt=0.01)


@pytest.mark.quick
def test_case_1_tanh(config):
    """
    Case 1: Standard LQR (Zero Terminal Cost)
    Exact: P(t) = tanh(T - t)
    """
    solver = RiccatiODESolver(config, Q=np.array([[1.0]]), R=np.array([[1.0]]))

    # Solve (Default terminal cost is 0)
    solver.solve(A=np.array([[0.0]]), B=np.array([[1.0]]))

    times = np.linspace(0, config.T, 5)
    for t in times:
        K = solver.get_K(t, B=np.array([[1.0]]))
        P_numeric = -K[0, 0]  # Since K = -P
        P_exact = np.tanh(config.T - t)

        assert P_numeric == pytest.approx(P_exact, abs=1e-9)


@pytest.mark.quick
def test_case_2_rational(config):
    """
    Case 2: Energy Saver (High Terminal Cost)
    Exact: P(t) = G / (1 + G(T-t))
    """
    G = 10.0
    solver = RiccatiODESolver(config, Q=np.array([[0.0]]), R=np.array([[1.0]]))

    # Solve with explicit terminal cost
    solver.solve(
        A=np.array([[0.0]]), B=np.array([[1.0]]), terminal_cost=np.array([[G]])
    )

    times = np.linspace(0, config.T, 5)
    for t in times:
        K = solver.get_K(t, B=np.array([[1.0]]))
        P_numeric = -K[0, 0]
        P_exact = G / (1 + G * (config.T - t))

        assert P_numeric == pytest.approx(P_exact, abs=1e-9)


@pytest.mark.quick
def test_case_3_damped(config):
    """
    Case 3: Damped Regulator
    Exact: P(t) = 2 / (3 * exp(2(T-t)) - 1)
    """
    solver = RiccatiODESolver(config, Q=np.array([[0.0]]), R=np.array([[1.0]]))

    # Solve with A=-1, P(T)=1
    solver.solve(
        A=np.array([[-1.0]]),
        B=np.array([[1.0]]),
        terminal_cost=np.array([[1.0]]),
    )

    times = np.linspace(0, config.T, 5)
    for t in times:
        K = solver.get_K(t, B=np.array([[1.0]]))
        P_numeric = -K[0, 0]

        tau = config.T - t
        P_exact = 2.0 / (3.0 * np.exp(2 * tau) - 1.0)

        assert P_numeric == pytest.approx(P_exact, abs=1e-9)
