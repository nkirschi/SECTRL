"""
Post-processing: aggregate across seeds, statistical tests, and plotting.
"""

import numpy as np
from scipy import stats
from typing import List
import warnings


def aggregate_metric(results, agent_name, metric_fn):
    """
    Extract a per-seed scalar metric and return array over seeds.

    Parameters
    ----------
    results : list of SeedResult
    agent_name : str
    metric_fn : callable(SeedResult, agent_name) -> float

    Returns
    -------
    ndarray, shape (n_seeds,)
    """
    return np.array([metric_fn(r, agent_name) for r in results])


def aggregate_trajectory(results, agent_name, key):
    """
    Extract a per-episode diagnostic trajectory for each seed.

    Parameters
    ----------
    results : list of SeedResult
    agent_name : str
    key : str
        Diagnostic key (e.g. 'error_joint').

    Returns
    -------
    ndarray, shape (n_seeds, M)
    """
    trajs = [r.diagnostic_trajectory(agent_name, key) for r in results]
    return np.array(trajs)


def cumulative_regret_trajectories(results, agent_name, oracle_name="oracle"):
    """
    Cumulative regret trajectories across seeds.

    Returns
    -------
    ndarray, shape (n_seeds, M)
    """
    return np.array([r.cumulative_regret(agent_name, oracle_name) for r in results])


def cost_trajectories(results, agent_name):
    """
    Per-episode cost trajectories across seeds.

    Returns
    -------
    ndarray, shape (n_seeds, M)
    """
    return np.array([[ep.cost for ep in r.episodes[agent_name]] for r in results])


def mean_and_ci(arr, axis=0, confidence=0.95):
    """
    Compute mean and confidence interval across axis.

    Parameters
    ----------
    arr : ndarray
    axis : int
    confidence : float

    Returns
    -------
    mean : ndarray
    ci_low : ndarray
    ci_high : ndarray
    """
    n = arr.shape[axis]
    mean = np.mean(arr, axis=axis)
    se = np.std(arr, axis=axis, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    return mean, ci_low, ci_high


# ─────────────────────────────────────────────────────────────────────
# Statistical tests (all paired by seed)
# ─────────────────────────────────────────────────────────────────────


def paired_t_test(results, agent_a, agent_b):
    """
    Two-sided paired t-test on final cumulative regret.

    Returns
    -------
    dict with 't_stat', 'p_value', 'mean_diff'.
    """
    vals_a = aggregate_metric(
        results, agent_a, lambda r, a: r.final_cumulative_regret(a)
    )
    vals_b = aggregate_metric(
        results, agent_b, lambda r, a: r.final_cumulative_regret(a)
    )
    diffs = vals_a - vals_b
    t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
    return {"t_stat": t_stat, "p_value": p_val, "mean_diff": np.mean(diffs)}


def wilcoxon_test(results, agent_a, agent_b, alternative="greater"):
    """
    One-sided Wilcoxon signed-rank test.

    Default alternative='greater' tests H_a: agent_a > agent_b.

    Returns
    -------
    dict with 'statistic', 'p_value'.
    """
    vals_a = aggregate_metric(
        results, agent_a, lambda r, a: r.final_cumulative_regret(a)
    )
    vals_b = aggregate_metric(
        results, agent_b, lambda r, a: r.final_cumulative_regret(a)
    )
    diffs = vals_a - vals_b

    # Wilcoxon requires nonzero differences
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 2:
        return {"statistic": np.nan, "p_value": np.nan}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_val = stats.wilcoxon(nonzero, alternative=alternative)

    return {"statistic": stat, "p_value": p_val}


def sign_test(results, agent_a, agent_b):
    """
    Exact two-sided sign test.

    Returns
    -------
    dict with 'wins_a', 'wins_b', 'ties', 'n', 'p_value'.
    """
    vals_a = aggregate_metric(
        results, agent_a, lambda r, a: r.final_cumulative_regret(a)
    )
    vals_b = aggregate_metric(
        results, agent_b, lambda r, a: r.final_cumulative_regret(a)
    )
    diffs = vals_a - vals_b

    wins_a = np.sum(diffs > 0)  # agent_a has higher regret
    wins_b = np.sum(diffs < 0)  # agent_b has higher regret
    ties = np.sum(diffs == 0)
    n = wins_a + wins_b  # exclude ties

    if n == 0:
        return {"wins_a": 0, "wins_b": 0, "ties": len(diffs), "n": 0, "p_value": 1.0}

    # Two-sided: P(X >= max(wins_a, wins_b)) under Binomial(n, 0.5)
    k = max(wins_a, wins_b)
    p_val = 2 * stats.binom.sf(k - 1, n, 0.5)
    p_val = min(p_val, 1.0)

    return {
        "wins_a": int(wins_a),
        "wins_b": int(wins_b),
        "ties": int(ties),
        "n": int(n),
        "p_value": p_val,
    }


def all_pairwise_tests(results, dense_name="dense_greedy", sparse_names=None):
    """
    Run all three tests for each (dense, sparse) pair.

    Returns
    -------
    dict[comparison_label] -> dict with t_test, wilcoxon, sign_test results.
    """
    if sparse_names is None:
        sparse_names = ["sparse_greedy", "sparse_excitation"]

    output = {}
    for sp in sparse_names:
        label = f"{dense_name}_vs_{sp}"
        output[label] = {
            "t_test": paired_t_test(results, dense_name, sp),
            "wilcoxon": wilcoxon_test(results, dense_name, sp),
            "sign_test": sign_test(results, dense_name, sp),
        }
    return output


# ─────────────────────────────────────────────────────────────────────
# Non-endpoint robustness
# ─────────────────────────────────────────────────────────────────────


def quarter_horizon_regret(results, agent_name, oracle_name="oracle"):
    """
    Cumulative regret at the quarter-way point (episode M/4).

    Returns ndarray of shape (n_seeds,).
    """
    trajs = cumulative_regret_trajectories(results, agent_name, oracle_name)
    M = trajs.shape[1]
    q = max(M // 4 - 1, 0)
    return trajs[:, q]


def episode_average_regret(results, agent_name, oracle_name="oracle"):
    """
    Episode-averaged cumulative regret: (1/M) sum_{m=1}^{M} R(m).

    Returns ndarray of shape (n_seeds,).
    """
    trajs = cumulative_regret_trajectories(results, agent_name, oracle_name)
    return np.mean(trajs, axis=1)


def seed_wins(results, agent_a, agent_b, oracle_name="oracle"):
    """
    Number of seeds where agent_b has lower final regret than agent_a.

    Returns int.
    """
    vals_a = aggregate_metric(
        results, agent_a, lambda r, a: r.final_cumulative_regret(a, oracle_name)
    )
    vals_b = aggregate_metric(
        results, agent_b, lambda r, a: r.final_cumulative_regret(a, oracle_name)
    )
    return int(np.sum(vals_b < vals_a))


# ─────────────────────────────────────────────────────────────────────
# Basin entry analysis
# ─────────────────────────────────────────────────────────────────────


def basin_entry_analysis(
    results, agent_name, thresholds=(0.05, 0.10, 0.15, 0.20, 0.30)
):
    """
    Basin-entry episode for each seed and threshold.

    Returns
    -------
    dict[threshold] -> ndarray of shape (n_seeds,) with entry episodes
        (np.nan if never reached).
    """
    from diagnostics import basin_entry_episode

    output = {eps: [] for eps in thresholds}

    for r in results:
        error_traj = r.diagnostic_trajectory(agent_name, "error_joint")
        entries = basin_entry_episode(error_traj, thresholds)
        for eps in thresholds:
            val = entries[eps]
            output[eps].append(val if val is not None else np.nan)

    return {eps: np.array(vals) for eps, vals in output.items()}


def basin_entry_ratio(
    results, dense_name="dense_greedy", sparse_name="sparse_greedy", threshold=0.15
):
    """
    Empirical ratio m0_dense / m0_sparse for each seed.

    Returns
    -------
    ratios : ndarray (n_seeds,) — nan where either agent never entered.
    median_ratio : float
    """
    dense_entries = basin_entry_analysis(results, dense_name, (threshold,))[threshold]
    sparse_entries = basin_entry_analysis(results, sparse_name, (threshold,))[threshold]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratios = dense_entries / sparse_entries

    valid = np.isfinite(ratios)
    median = np.nanmedian(ratios) if np.any(valid) else np.nan

    return ratios, median


# ─────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────


def final_summary_table(results, agent_names=None, oracle_name="oracle"):
    """
    Print a summary table of final-episode statistics.

    Returns
    -------
    dict[agent_name] -> dict of metric -> (mean, ci_half_width)
    """
    if agent_names is None:
        agent_names = ["dense_greedy", "sparse_greedy", "sparse_excitation"]

    table = {}
    for name in agent_names:
        final_reg = aggregate_metric(
            results, name, lambda r, a: r.final_cumulative_regret(a, oracle_name)
        )

        # Final-episode diagnostics
        diag_keys = [
            "error_joint",
            "error_A",
            "error_B",
            "support_f1_joint",
            "support_f1_A",
            "support_f1_B",
        ]
        diag_vals = {}
        for key in diag_keys:
            traj = aggregate_trajectory(results, name, key)
            final_vals = traj[:, -1]  # last episode, shape (n_seeds,)
            m = np.mean(final_vals)
            n = len(final_vals)
            se = np.std(final_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
            diag_vals[key] = (float(m), float(t_crit * se))

        reg_m = np.mean(final_reg)
        n = len(final_reg)
        reg_se = np.std(final_reg, ddof=1) / np.sqrt(n) if n > 1 else 0.0
        t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))

        row = {
            "final_regret": (float(reg_m), float(t_crit * reg_se)),
            **diag_vals,
        }
        table[name] = row

    return table


def print_summary(table):
    """Pretty-print the summary table."""
    header = (
        f"{'Agent':<22} {'Regret':>16} {'Err(joint)':>14} "
        f"{'Err(A)':>14} {'Err(B)':>14} {'F1(joint)':>14} {'F1(A)':>14} {'F1(B)':>14}"
    )
    print(header)
    print("-" * len(header))
    for name, row in table.items():

        def fmt(key):
            m, ci = row[key]
            return f"{m:.2f}±{ci:.2f}"

        print(
            f"{name:<22} {fmt('final_regret'):>16} "
            f"{fmt('error_joint'):>14} {fmt('error_A'):>14} "
            f"{fmt('error_B'):>14} {fmt('support_f1_joint'):>14}"
            f"{fmt('support_f1_A'):>14} {fmt('support_f1_B'):>14}"
        )


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────


def plot_trajectories(results, exp_config, save_path=None):
    """
    Plot 6-panel diagnostic figure (matching the paper's Figure 1).

    Panels: cumulative regret, parameter error (log), restricted Gram
    min eigenvalue, support F1, spectral abscissa, episode cost.
    """
    import matplotlib.pyplot as plt

    agent_names = ["oracle", "dense_greedy", "sparse_greedy", "sparse_excitation"]
    colors = {
        "oracle": "green",
        "dense_greedy": "orange",
        "sparse_greedy": "blue",
        "sparse_excitation": "red",
    }
    labels = {
        "oracle": "Oracle",
        "dense_greedy": "Dense-Greedy",
        "sparse_greedy": "Sparse-Greedy",
        "sparse_excitation": "Sparse-Excitation",
    }

    M = len(results[0].episodes["oracle"])
    episodes = np.arange(1, M + 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"$d = {exp_config.x_dim}$, "
        f"$p = {exp_config.u_dim}$, "
        f"$s = {exp_config.sparsity}$, "
        f"$N = {exp_config.max_episodes}$, "
        f"$T = {exp_config.T}$, "
        f"$dt = {exp_config.dt}$, "
        f"$\sigma = {exp_config.sigma}$, "
        f"$\delta = {exp_config.delta}$, "
        f"$c_\lambda = {exp_config.c_lambda}$",
        fontsize=16,
    )

    panel_configs = [
        ("Cumulative Regret", "cumulative_regret", False, False),
        ("Relative Parameter Error", "error_joint", True, True),
        ("Restricted Gram Min Eigenvalue", "gram_min_eig", False, True),
        ("Support F1", "support_f1_joint", False, True),
        ("Closed-Loop Spectral Abscissa", "spectral_abscissa_t0", False, True),
        ("Episode Cost", "episode_cost", False, False),
    ]

    for ax, (title, key, use_log, is_diagnostic) in zip(axes.flat, panel_configs):
        for name in agent_names:
            if is_diagnostic and name == "oracle":
                continue

            if key == "cumulative_regret":
                data = (
                    cumulative_regret_trajectories(results, name, "oracle")
                    if name != "oracle"
                    else np.zeros((len(results), M))
                )
            elif key == "episode_cost":
                data = cost_trajectories(results, name)
            else:
                data = aggregate_trajectory(results, name, key)

            if data.size == 0:
                continue

            mean, ci_lo, ci_hi = mean_and_ci(data, axis=0)
            ax.plot(episodes, mean, color=colors[name], label=labels[name])
            ax.fill_between(episodes, ci_lo, ci_hi, color=colors[name], alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel("Episode")
        if use_log:
            ax.set_yscale("log")

    # Add legend to first panel
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_basin_entry_comparison(results, exp_config, save_path=None):
    """
    Plot empirical vs theoretical basin-entry speedup ratio.
    """
    import matplotlib.pyplot as plt

    ratios, median = basin_entry_ratio(
        results, "dense_greedy", "sparse_greedy", threshold=0.15
    )
    theoretical = exp_config.theoretical_speedup

    fig, ax = plt.subplots(figsize=(6, 4))
    valid = ratios[np.isfinite(ratios)]
    if len(valid) > 0:
        ax.hist(valid, bins=20, alpha=0.7, label="Empirical ratios")
        ax.axvline(median, color="blue", linestyle="--", label=f"Median: {median:.2f}")
    ax.axvline(
        theoretical, color="red", linestyle="-", label=f"Theory: {theoretical:.2f}"
    )
    ax.set_xlabel(r"$m_0^{\mathrm{dense}} / m_0^{\mathrm{sparse}}$")
    ax.set_ylabel("Count")
    ax.set_title("Basin Entry Speedup Ratio")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
