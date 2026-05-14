"""
Core experimental loop with matched-seed design.

Each seed defines one system and one noise stream shared across all agents.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

from dynamics import ContinuousLQREnv
from estimator import RegressionBuffer, RowLassoEstimator
from system_generator import sample_sparse_system, define_cost_matrices
from agent import (
    OracleAgent,
    DenseGreedyAgent,
    SparseGreedyAgent,
    SparseExcitationAgent,
)
from diagnostics import collect_diagnostics, episode_cost
from tqdm import tqdm


AGENT_NAMES = ["oracle", "dense_greedy", "sparse_greedy", "sparse_excitation"]


@dataclass
class EpisodeRecord:
    """Per-episode results for one agent."""

    cost: float = 0.0
    excitation_tax: float = 0.0  # sigma_u^2 * tr(R) * T, used for adjusted regret
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SeedResult:
    """All results from one seed, for all agents."""

    seed: int = 0
    A_star: np.ndarray = None
    B_star: np.ndarray = None
    supports: list = None

    # Per-agent: dict[agent_name] -> list[EpisodeRecord]
    episodes: Dict[str, List[EpisodeRecord]] = field(default_factory=dict)

    @property
    def agent_names(self):
        return list(self.episodes.keys())

    def cumulative_regret(self, agent_name, oracle_name="oracle"):
        """Cumulative regret relative to oracle."""
        oracle_costs = np.array([ep.cost for ep in self.episodes[oracle_name]])
        agent_costs = np.array([ep.cost for ep in self.episodes[agent_name]])
        return np.cumsum(agent_costs - oracle_costs)

    def adjusted_cumulative_regret(self, agent_name, oracle_name="oracle"):
        """
        Cumulative regret with the deterministic excitation tax removed.

        For excitation agents, the observed cost includes sigma_u^2 * tr(R) * T
        per episode regardless of learning quality.  Subtracting this
        isolates the feedback suboptimality from the exploration overhead,
        placing all agents on a comparable scale.

        For non-excitation agents, excitation_tax == 0 so this equals
        cumulative_regret.
        """
        oracle_costs = np.array([ep.cost for ep in self.episodes[oracle_name]])
        agent_costs = np.array([ep.cost for ep in self.episodes[agent_name]])
        excit_taxes = np.array([ep.excitation_tax for ep in self.episodes[agent_name]])
        return np.cumsum(agent_costs - excit_taxes - oracle_costs)

    def final_cumulative_regret(self, agent_name, oracle_name="oracle"):
        return self.cumulative_regret(agent_name, oracle_name)[-1]

    def diagnostic_trajectory(self, agent_name, key):
        """Extract a single diagnostic across episodes."""
        return [ep.diagnostics.get(key, np.nan) for ep in self.episodes[agent_name]]


def _make_stabilising_init(d, p, margin=1.0):
    """
    Return a simple stabilising initial estimate.

    Uses A_init = -margin * I, B_init = 0.
    The closed-loop under DRE with these parameters is stable
    since the drift is strongly negative.
    """
    A_init = -margin * np.eye(d)
    B_init = np.zeros((d, p))  # TODO: eliminate the need for placeholder estimates
    return A_init, B_init


def _create_agents(exp_config, sys_config, Q, R, A_star, B_star, seed):
    """
    Instantiate all four agents.

    Returns dict[agent_name] -> Agent.
    """
    d = exp_config.x_dim
    p = exp_config.u_dim
    M = exp_config.max_episodes

    A_init, B_init = _make_stabilising_init(d, p)

    agents = {}

    agents["oracle"] = OracleAgent(sys_config, Q, R, A_star, B_star)

    agents["dense_greedy"] = DenseGreedyAgent(
        sys_config, Q, R, A_init, B_init, mu=exp_config.mu_ridge
    )

    # Lasso kwargs (shared between sparse agents)
    lasso_kwargs = dict(
        sigma_bar=exp_config.sigma_bar,
        c_lambda=exp_config.c_lambda,
        delta=exp_config.delta,
        max_episodes=M,
    )
    if exp_config.lambda_lasso is not None:
        lasso_kwargs = dict(lambda_fixed=exp_config.lambda_lasso)

    agents["sparse_greedy"] = SparseGreedyAgent(
        sys_config, Q, R, A_init, B_init, **lasso_kwargs
    )

    # Independent excitation RNG seeded deterministically from main seed
    exc_rng = np.random.RandomState(seed + 1_000_000)

    agents["sparse_excitation"] = SparseExcitationAgent(
        sys_config,
        Q,
        R,
        A_init,
        B_init,
        sigma_u=exp_config.sigma_u,
        excitation_rng=exc_rng,
        **lasso_kwargs,
    )

    return agents


def run_paired_experiment(exp_config, seed, verbose=False):
    """
    Run one paired-seed experiment: all agents on the same system and noise.

    Parameters
    ----------
    exp_config : ExperimentConfig
    seed : int
    verbose : bool
        Print progress every 10 episodes.

    Returns
    -------
    SeedResult
    """
    d = exp_config.x_dim
    p = exp_config.u_dim
    M = exp_config.max_episodes
    H = exp_config.H
    dt = exp_config.dt
    sigma = exp_config.sigma

    # Sample system
    A_star, B_star, supports, n_attempts = sample_sparse_system(
        d, p, exp_config.sparsity, seed
    )
    Q, R = define_cost_matrices(d, p)

    if verbose:
        print(f"Sampled system after {n_attempts} attempt(s)")

    sys_config = exp_config.system_config

    # Pre-generate shared noise: (M, H, d) standard normals
    noise_rng = np.random.RandomState(seed + 2_000_000)
    shared_noise = noise_rng.randn(M, H, d)

    # deterministic RNG for the pure exploration phase
    m_explore = int(np.ceil(2 * (d + p) / H))
    explore_rng = np.random.RandomState(seed + 3_000_000)
    shared_exploration = explore_rng.randn(m_explore, H, p) * sigma

    n_checkpoints = min(8, M)
    checkpoint_set = set(
        np.round(np.linspace(0, M - 1, n_checkpoints)).astype(int).tolist()
    )

    x0 = np.zeros(d)

    # Create agents
    agents = _create_agents(exp_config, sys_config, Q, R, A_star, B_star, seed)

    # Create per-agent data buffers
    buffers = {
        name: RegressionBuffer(d, p, M, H) for name in AGENT_NAMES if name != "oracle"
    }

    # Noise covariance (assumed diagonal sigma * I)
    Sigma = sigma * np.eye(d)

    # Result container
    result = SeedResult(
        seed=seed,
        A_star=A_star,
        B_star=B_star,
        supports=supports,
        episodes={name: [] for name in AGENT_NAMES},
    )
    for m in tqdm(range(M), disable=not verbose):
        for name in AGENT_NAMES:
            agent = agents[name]
            env = ContinuousLQREnv(A_star, B_star, Sigma, x0, sys_config)

            # Collect trajectory
            states = []
            controls = []
            zs_list = []
            ys_list = []

            x = env.reset()

            for k in range(H):
                t = k * dt

                if name != "oracle" and m < m_explore:
                    # Inject random Gaussian noise matching the state variance
                    u = shared_exploration[m, k]
                else:
                    u = agent.get_control(t, x)

                # Step with shared noise
                x_new, dx, step_dt = env.step(u, noise_override=shared_noise[m, k])

                # Regression data: z_k = [x_k; u_k], y_k = dx / dt
                z_k = np.concatenate([x, u])
                y_k = dx / dt

                states.append(x.copy())
                controls.append(u.copy())
                zs_list.append(z_k)
                ys_list.append(y_k)

                x = x_new

            # Compute episode cost
            cost = episode_cost(states, controls, Q, R, dt)
            excitation_tax = (
                agent.sigma_u**2 * float(np.trace(R)) * exp_config.T
                if isinstance(agent, SparseExcitationAgent)
                else 0.0
            )

            # Update agent estimate (not for oracle)
            if name != "oracle":
                zs = np.stack(zs_list)  # (H, d+p)
                ys = np.stack(ys_list)  # (H, d)
                buffers[name].add_episode(zs, ys)
                agent.update(buffers[name])

            # Collect diagnostics
            buf = buffers.get(name, None)
            diag = (
                collect_diagnostics(
                    agent,
                    buf,
                    A_star,
                    B_star,
                    supports,
                    Q,
                    R,
                    threshold=exp_config.support_threshold,
                )
                if buf is not None
                else {}
            )

            if name != "oracle" and m in checkpoint_set and agent.A_est is not None:
                diag["A_est"] = agent.A_est.copy()
                diag["B_est"] = agent.B_est.copy()

            result.episodes[name].append(
                EpisodeRecord(
                    cost=cost, excitation_tax=excitation_tax, diagnostics=diag
                )
            )

    return result


def run_benchmark(exp_config, seeds=None, verbose=False):
    """
    Run a full benchmark across multiple seeds.

    Parameters
    ----------
    exp_config : ExperimentConfig
    seeds : list of int, or None (defaults to range(n_seeds))
    verbose : bool

    Returns
    -------
    list of SeedResult
    """
    if seeds is None:
        seeds = list(range(exp_config.n_seeds))

    results = []
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"Running seed {i + 1}/{len(seeds)} (seed={seed})")
        res = run_paired_experiment(exp_config, seed, verbose=verbose)
        results.append(res)

    return results
