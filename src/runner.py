"""
Core experimental loop with matched-seed design.
Each seed defines one system and one noise stream shared across all agents.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

from dynamics import ContinuousLQREnv
from estimator import RegressionBuffer, DiscreteRidgeEstimator, RowLassoEstimator
from system_generator import sample_sparse_system
from planner import RiccatiODESolver
from agent import LinearControlAgent
from diagnostics import (
    relative_parameter_error,
    support_metrics,
    restricted_gram_min_eigenvalue,
    closed_loop_spectral_abscissa,
    episode_cost,
    self_exploration_metrics,
)
from common import ExperimentConfig


@dataclass
class EpisodeRecord:
    cost: float = 0.0
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SeedResult:
    """All results from one seed, for all agents."""

    seed: int = 0
    A_star: np.ndarray = None
    B_star: np.ndarray = None
    supports: list = None
    btqb_min_eig: float = 0.0
    btqb_max_eig: float = 0.0
    episodes: Dict[str, List[EpisodeRecord]] = field(default_factory=dict)

    @property
    def agent_names(self):
        return list(self.episodes.keys())

    def cumulative_regret(self, agent_name, oracle_name="oracle"):
        oracle_costs = np.array([ep.cost for ep in self.episodes[oracle_name]])
        agent_costs = np.array([ep.cost for ep in self.episodes[agent_name]])
        return np.cumsum(agent_costs - oracle_costs)

    def diagnostic_trajectory(self, agent_name, key):
        return [ep.diagnostics.get(key, np.nan) for ep in self.episodes[agent_name]]


def _build_agent(
    name: str,
    exp_config: ExperimentConfig,
    A_true: np.ndarray,
    B_true: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> LinearControlAgent:
    """Instantiate agents by name."""
    d, p = exp_config.system.d, exp_config.system.p
    planner = RiccatiODESolver(exp_config.system, Q, R)
    prior_theta = np.concatenate([-np.eye(d), np.zeros((d, p))], axis=1)

    if name == "oracle":
        planner.solve(A_true, B_true)
        return LinearControlAgent(
            name,
            np.concatenate([A_true, B_true], axis=1),
            d,
            p,
            Q,
            R,
            None,
            planner,
        )
    if name == "dense_greedy":
        est = DiscreteRidgeEstimator(exp_config.estimators)
        return LinearControlAgent(name, prior_theta.copy(), d, p, Q, R, est, planner)
    if name == "dense_excited":
        est = DiscreteRidgeEstimator(exp_config.estimators)
        return LinearControlAgent(
            name,
            prior_theta.copy(),
            d,
            p,
            Q,
            R,
            est,
            planner,
            sigma_u=exp_config.excitation.sigma_u,
        )
    if name == "sparse_greedy":
        est = RowLassoEstimator(
            exp_config.system, exp_config.estimators, exp_config.theoretical_lambda
        )
        return LinearControlAgent(name, prior_theta.copy(), d, p, Q, R, est, planner)
    if name == "sparse_excited":
        est = RowLassoEstimator(
            exp_config.system, exp_config.estimators, exp_config.theoretical_lambda
        )
        return LinearControlAgent(
            name,
            prior_theta.copy(),
            d,
            p,
            Q,
            R,
            est,
            planner,
            sigma_u=exp_config.excitation.sigma_u,
        )
    raise ValueError(f"Unknown agent: {name}")


def run_paired_experiment(
    exp_config: ExperimentConfig, seed: int, verbose: bool = False
) -> SeedResult:
    """Run one paired-seed experiment: all configured agents on the same system and noise."""
    d, p = exp_config.system.d, exp_config.system.p
    M, H = exp_config.max_episodes, exp_config.system.H

    A_star, B_star, supports, n_attempts = sample_sparse_system(
        d=d,
        p=p,
        s_A=exp_config.system.s_A,
        s_B=exp_config.system.s_B,
        seed=seed,
        a_min=exp_config.system.a_min,
        a_max=exp_config.system.a_max,
        b_min=exp_config.system.b_min,
        b_max=exp_config.system.b_max,
    )

    Q = np.eye(d) * exp_config.cost.q_scale
    R = np.eye(p) * exp_config.cost.r_scale
    sys = ContinuousLQREnv(
        A_star, B_star, exp_config.system.sigma, exp_config.system.dt
    )

    btqb_metrics = self_exploration_metrics(B_star, Q)

    if verbose:
        print(f"Sampled system after {n_attempts} attempt(s)")
    noise_rng = np.random.default_rng(seed + 2_000_000)
    shared_noise = noise_rng.standard_normal((M, H, d))

    explore_rng = np.random.default_rng(seed + 3_000_000)
    shared_exploration = (
        explore_rng.standard_normal((exp_config.m_explore, H, p))
        * exp_config.system.sigma
    )

    x0_rng = np.random.default_rng(seed + 4_000_000)
    x0s = x0_rng.standard_normal((M, d)) * exp_config.x0_std

    agents = {
        name: _build_agent(name, exp_config, A_star, B_star, Q, R)
        for name in exp_config.agents
    }
    buffers = {
        name: RegressionBuffer(d, p, M, H) for name in agents if name != "oracle"
    }
    rngs = {
        name: np.random.default_rng(seed + i * 1000) for i, name in enumerate(agents)
    }

    result = SeedResult(
        seed=seed,
        A_star=A_star,
        B_star=B_star,
        supports=supports,
        btqb_min_eig=btqb_metrics["min_eig"],
        btqb_max_eig=btqb_metrics["max_eig"],
        episodes={n: [] for n in agents},
    )
    theta_true = np.concatenate([A_star, B_star], axis=1)

    n_checkpoints = min(8, M)
    checkpoint_episodes = sorted(
        set(np.round(np.linspace(0, M - 1, n_checkpoints)).astype(int).tolist())
    )

    for m in range(M):
        for name, agent in agents.items():
            x = x0s[m].copy()
            states, controls = np.zeros((H, d)), np.zeros((H, p))
            zs, ys = np.zeros((H, d + p)), np.zeros((H, d))

            for k in range(H):
                t = k * sys.dt

                # Strict Phase Separation: Pure open-loop noise vs Closed-loop exploitation
                if name != "oracle" and m < exp_config.m_explore:
                    u = shared_exploration[m, k]
                else:
                    u = agent.get_control(t, x, rngs[name])
                u = np.clip(u, -exp_config.action_clip, exp_config.action_clip)

                x_next = sys.step(x, u, noise=shared_noise[m, k])
                x_next = np.clip(x_next, -exp_config.state_clip, exp_config.state_clip)

                states[k], controls[k] = x, u
                zs[k], ys[k] = np.concatenate([x, u]), (x_next - x) / sys.dt
                x = x_next

            cost = episode_cost(states, controls, Q, R, sys.dt)

            if name != "oracle":
                buffers[name].add_episode(zs, ys)
                # Only update model after the pure exploration phase
                if m >= exp_config.m_explore - 1:
                    agent.update_model(buffers[name].Z, buffers[name].Y)

            diag = {}
            if name != "oracle" and m >= exp_config.m_explore - 1:
                err = relative_parameter_error(agent.theta_hat, theta_true)
                sup = support_metrics(
                    agent.theta_hat, supports, exp_config.support_threshold
                )
                diag.update(
                    {
                        "error_joint": err["joint"],
                        "error_A": err["A"],
                        "error_B": err["B"],
                        "support_f1_joint": sup["joint"]["f1"],
                        "support_f1_A": sup["A"]["f1"],
                        "support_f1_B": sup["B"]["f1"],
                        "episode_cost": cost,
                        "gram_min_eig": restricted_gram_min_eigenvalue(
                            buffers[name].Z, supports
                        ),
                        "spectral_abscissa_t0": closed_loop_spectral_abscissa(
                            A_star, B_star, agent.planner.get_K(0.0, agent.B_hat)
                        ),
                    }
                )
                if m in checkpoint_episodes:
                    diag["A_est"] = agent.A_hat.copy()
                    diag["B_est"] = agent.B_hat.copy()

            result.episodes[name].append(
                EpisodeRecord(cost=cost, diagnostics=diag)
            )

    return result