"""
Core experimental loop with matched-seed design.
Each seed defines one system and one noise stream shared across all agents.
"""

from __future__ import annotations

import os
import pickle
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
    A_0: np.ndarray,
    B_0: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> LinearControlAgent:
    """Instantiate agents by name."""
    d, p = exp_config.system.d, exp_config.system.p
    planner = RiccatiODESolver(exp_config.system, Q, R)
    prior_theta = np.concatenate([A_0, B_0], axis=1)

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

    planner.solve(A_0, B_0)

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
    # The Lasso warmup is only needed to break the trap when there is no
    # pure-exploration phase.
    use_warmup = exp_config.m_explore == 0
    if name == "sparse_greedy":
        est = RowLassoEstimator(
            exp_config.system, exp_config.estimators,
            exp_config.theoretical_lambda, use_warmup=use_warmup,
        )
        return LinearControlAgent(name, prior_theta.copy(), d, p, Q, R, est, planner)
    if name == "sparse_excited":
        est = RowLassoEstimator(
            exp_config.system, exp_config.estimators,
            exp_config.theoretical_lambda, use_warmup=use_warmup,
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
    if verbose:
        print(f"Sampled system after {n_attempts} attempt(s)")

    Q = np.eye(d) * exp_config.cost.q_scale
    R = np.eye(p) * exp_config.cost.r_scale
    sys = ContinuousLQREnv(
        A_star, B_star, exp_config.system.sigma, exp_config.system.dt
    )

    btqb_metrics = self_exploration_metrics(B_star, Q)

    noise_rng = np.random.default_rng(seed + 2_000_000)
    shared_noise = noise_rng.standard_normal((M, H, d))

    explore_rng = np.random.default_rng(seed + 3_000_000)
    shared_exploration = (
        explore_rng.standard_normal((int(np.ceil(exp_config.m_explore)), H, p))
        * exp_config.system.sigma
    )
    exploration_steps = H * exp_config.m_explore
    if verbose:
        print(f"Pure exploration for {exploration_steps} initial steps")

    x0_rng = np.random.default_rng(seed + 4_000_000)
    x0s = x0_rng.standard_normal((M, d)) * exp_config.x0_std

    def rms_scale(c_min: float, c_max: float, sparsity: int) -> float:
        return np.sqrt(sparsity * (c_min**2 + c_min * c_max + c_max**2) / 3)

    a_scale = rms_scale(
        exp_config.system.a_min, exp_config.system.a_max, exp_config.system.s_A
    )
    b_scale = rms_scale(
        exp_config.system.b_min, exp_config.system.b_max, exp_config.system.s_B
    )
    A_0 = -a_scale * np.eye(d)
    B_0 = b_scale * np.eye(d, p)

    agents = {
        name: _build_agent(name, exp_config, A_star, B_star, A_0, B_0, Q, R)
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
                current_step = m * H + k

                # initial pure exploration for greedy agents
                if (
                    name in ["dense_greedy", "sparse_greedy"]
                    and current_step < exploration_steps
                ):
                    u = shared_exploration[m, k]
                else:
                    u = agent.get_control(t, x, rngs[name])
                u = np.clip(u, -exp_config.action_clip, exp_config.action_clip)

                x_next = sys.step(x, u, noise=shared_noise[m, k])

                states[k], controls[k] = x, u
                zs[k], ys[k] = np.concatenate([x, u]), (x_next - x) / sys.dt
                x_next = np.clip(x_next, -exp_config.state_clip, exp_config.state_clip)
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

            result.episodes[name].append(EpisodeRecord(cost=cost, diagnostics=diag))

    return result


# ---------------------------------------------------------------------------
# Persistence of raw results
# ---------------------------------------------------------------------------

SEED_RESULTS_VERSION = 1
"""
Bump when the SeedResult / EpisodeRecord / ExperimentConfig schema changes
in a backward-incompatible way. The version field is checked on load.
"""

_SEED_RESULTS_FILENAME = "seed_results.pkl"


def persist_raw_results(
    results: List[SeedResult],
    exp_config: ExperimentConfig,
    output_dir: str,
) -> None:
    """
    Pickle the per-seed results and source config so that plots and summary
    files can be regenerated later without re-running the experiment.

    Layout written:
        <output_dir>/seed_results.pkl  -- {version, results, config}
    """
    payload = {
        "version": SEED_RESULTS_VERSION,
        "results": results,
        "config": exp_config,
    }
    path = os.path.join(output_dir, _SEED_RESULTS_FILENAME)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_raw_results(
    output_dir: str,
) -> tuple[List[SeedResult], ExperimentConfig]:
    """
    Inverse of `persist_raw_results`. Raises FileNotFoundError if the pickle
    is missing and ValueError on a version mismatch.
    """
    path = os.path.join(output_dir, _SEED_RESULTS_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No seed_results.pkl in {output_dir}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    version = payload.get("version", 0)
    if version != SEED_RESULTS_VERSION:
        raise ValueError(
            f"seed_results.pkl in {output_dir} has version {version}; "
            f"current loader version is {SEED_RESULTS_VERSION}. "
            f"The schema has changed since this file was written."
        )
    return payload["results"], payload["config"]


def find_result_dirs(path: str) -> list[str]:
    """
    Return all directories at or under `path` that contain a seed_results.pkl.

    If `path` is itself a result directory, returns [path]. Otherwise walks
    recursively. Used by --replot to handle both single result directories
    and sweep parent directories transparently.
    """
    if os.path.isfile(os.path.join(path, _SEED_RESULTS_FILENAME)):
        return [path]
    found: list[str] = []
    for root, _dirs, files in os.walk(path):
        if _SEED_RESULTS_FILENAME in files:
            found.append(root)
    return sorted(found)
