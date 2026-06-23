"""
Post-processing: aggregate across seeds, statistical tests, and plotting.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
import warnings
import matplotlib

from common import ExperimentConfig

matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{bm}",
    }
)


def aggregate_metric(results, agent_name, metric_fn):
    """Extract a per-seed scalar metric and return array over seeds."""
    return np.array([metric_fn(r, agent_name) for r in results])


def aggregate_trajectory(results, agent_name, key):
    """Extract a per-episode diagnostic trajectory for each seed."""
    trajs = [r.diagnostic_trajectory(agent_name, key) for r in results]
    return np.array(trajs)


def cumulative_regret_trajectories(results, agent_name, oracle_name="oracle"):
    """Cumulative regret trajectories, shape (n_seeds, M)."""
    return np.array([r.cumulative_regret(agent_name, oracle_name) for r in results])


def cost_trajectories(results, agent_name):
    """Per-episode cost trajectories across seeds."""
    return np.array([[ep.cost for ep in r.episodes[agent_name]] for r in results])


def per_episode_regret_trajectories(results, agent_name, oracle_name="oracle"):
    """Per-episode cost excess r_m = J_m^agent - J_m^oracle, shape (n_seeds, M)."""
    return cost_trajectories(results, agent_name) - cost_trajectories(
        results, oracle_name
    )


def mean_and_ci(arr, axis=0, confidence=0.95):
    """Compute mean and confidence interval across axis."""
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


def _final_regret(results, agent_name, oracle_name="oracle"):
    """Per-seed final cumulative regret."""
    return cumulative_regret_trajectories(results, agent_name, oracle_name)[:, -1]


def paired_t_test(results, agent_a, agent_b):
    """Two-sided paired t-test on final cumulative regret."""
    vals_a = _final_regret(results, agent_a)
    vals_b = _final_regret(results, agent_b)
    diffs = vals_a - vals_b
    t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "mean_diff": float(np.mean(diffs)),
    }


def wilcoxon_test(results, agent_a, agent_b, alternative="greater"):
    """One-sided Wilcoxon signed-rank test on final cumulative regret."""
    vals_a = _final_regret(results, agent_a)
    vals_b = _final_regret(results, agent_b)
    diffs = vals_a - vals_b
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 2:
        return {"statistic": np.nan, "p_value": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_val = stats.wilcoxon(nonzero, alternative=alternative)
    return {"statistic": float(stat), "p_value": float(p_val)}


def sign_test(results, agent_a, agent_b):
    """Exact two-sided sign test on final cumulative regret."""
    vals_a = _final_regret(results, agent_a)
    vals_b = _final_regret(results, agent_b)
    diffs = vals_a - vals_b
    wins_a = int(np.sum(diffs > 0))
    wins_b = int(np.sum(diffs < 0))
    ties = int(np.sum(diffs == 0))
    n = wins_a + wins_b
    if n == 0:
        return {"wins_a": 0, "wins_b": 0, "ties": ties, "n": 0, "p_value": 1.0}
    k = max(wins_a, wins_b)
    p_val = min(2 * stats.binom.sf(k - 1, n, 0.5), 1.0)
    return {"wins_a": wins_a, "wins_b": wins_b, "ties": ties, "n": n, "p_value": p_val}


def all_pairwise_tests(results, dense_name="dense_greedy", sparse_names=None):
    if sparse_names is None:
        sparse_names = ["sparse_greedy", "sparse_excited"]
    available = set(results[0].agent_names) if results else set()
    output = {}
    for sp in sparse_names:
        if dense_name not in available or sp not in available:
            continue
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


def seed_wins(results, agent_a, agent_b, oracle_name="oracle"):
    """Seeds where agent_b has lower final cumulative regret than agent_a."""
    return int(
        np.sum(
            _final_regret(results, agent_b, oracle_name)
            < _final_regret(results, agent_a, oracle_name)
        )
    )


# ─────────────────────────────────────────────────────────────────────
# Basin entry analysis
# ─────────────────────────────────────────────────────────────────────


def basin_entry_analysis(
    results, agent_name, thresholds=(0.05, 0.10, 0.15, 0.20, 0.30)
):
    def basin_entry_episode(error_trajectory, thresholds):
        result = {}
        for eps in thresholds:
            try:
                result[eps] = min(
                    m for m, err in enumerate(error_trajectory) if err <= eps
                )
            except ValueError:
                result[eps] = None
        return result

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
    dense_m0 = basin_entry_analysis(results, dense_name, (threshold,))[threshold]
    sparse_m0 = basin_entry_analysis(results, sparse_name, (threshold,))[threshold]

    M = len(results[0].diagnostic_trajectory(dense_name, "error_joint")) if results else 0
    dense_censored = np.isnan(dense_m0)
    sparse_censored = np.isnan(sparse_m0)

    # Episodes-to-enter, 1-indexed; a non-entry is censored at M+1 (a lower bound:
    # M episodes were not enough, so the true entry time is at least M+1).
    dense_time = np.where(dense_censored, M + 1, dense_m0 + 1)
    sparse_time = np.where(sparse_censored, M + 1, sparse_m0 + 1)

    ratios = dense_time / sparse_time
    both_censored = dense_censored & sparse_censored
    ratios[both_censored] = np.nan  # neither entered -> uninformative

    median = float(np.nanmedian(ratios)) if np.any(~both_censored) else np.nan
    n = len(ratios)
    stats = {
        "n_seeds": int(n),
        "dense_never_entered": float(np.mean(dense_censored)) if n else float("nan"),
        "sparse_never_entered": float(np.mean(sparse_censored)) if n else float("nan"),
        "both_never_entered": int(np.sum(both_censored)),
    }
    return ratios, median, stats


# ─────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────


def final_summary_table(results, agent_names, oracle_name="oracle"):
    agent_names = [agent for agent in agent_names if agent != oracle_name]

    def _stats(vals):
        n = len(vals)
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        t_crit = float(stats.t.ppf(0.975, df=max(n - 1, 1)))
        return (m, t_crit * se)

    table = {}
    for name in agent_names:
        row = {
            "final_regret": _stats(_final_regret(results, name, oracle_name)),
        }
        for key in [
            "error_joint",
            "error_A",
            "error_B",
            "support_f1_joint",
            "support_f1_A",
            "support_f1_B",
        ]:
            traj = aggregate_trajectory(results, name, key)
            # Remove NaNs representing missing data before the checkpoint
            valid_final_vals = [val for val in traj[:, -1] if np.isfinite(val)]
            row[key] = (
                _stats(valid_final_vals) if valid_final_vals else (np.nan, np.nan)
            )
        table[name] = row
    return table


def print_summary(table):
    header = (
        f"{'Agent':<22} {'Regret':>14} {'Err(joint)':>12}"
        f"{'Err(A)':>12} {'Err(B)':>12} {'F1(joint)':>12} {'F1(A)':>12} {'F1(B)':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, row in table.items():

        def fmt(key):
            m, ci = row[key]
            return f"{float(m):.2f}±{float(ci):.2f}"

        print(
            f"{name:<22} {fmt('final_regret'):>14} "
            f"{fmt('error_joint'):>12} {fmt('error_A'):>12} {fmt('error_B'):>12} "
            f"{fmt('support_f1_joint'):>12} {fmt('support_f1_A'):>12} {fmt('support_f1_B'):>12}"
        )


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────


def plot_trajectories(results, exp_config: ExperimentConfig, save_path=None):
    import matplotlib.pyplot as plt

    ALL_AGENTS = list(exp_config.agents)
    LEARNING_AGENTS = [a for a in ALL_AGENTS if a != "oracle"]
    COLORS = {
        "oracle": "green",
        "dense_greedy": "blue",
        "dense_excited": "purple",
        "sparse_greedy": "red",
        "sparse_excited": "orange",
    }
    LABELS = {
        "oracle": "Oracle",
        "dense_greedy": "Dense-Greedy",
        "dense_excited": "Dense-Excited",
        "sparse_greedy": "Sparse-Greedy",
        "sparse_excited": "Sparse-Excited",
    }

    M = exp_config.max_episodes
    episodes = np.arange(1, M + 1)

    # (title, key, y_scale, agent_list)
    PANELS = [
        # Row 0
        (r"Cumulative Regret $R_m$", "cumul_regret", "lin", ALL_AGENTS),
        (r"Cumulative Regret $R_m$ (exp)", "cumul_regret", "exp", ALL_AGENTS),
        (r"Per-episode Regret $r_m$", "per_ep_regret", "lin", ALL_AGENTS),
        (r"Episode Cost $J(\bm{\pi}_m)$", "episode_cost", "lin", ALL_AGENTS),
        # Row 1
        (
            r"Parameter Error in $\mathbf{\Theta}$ (log)",
            "error_joint",
            "log",
            LEARNING_AGENTS,
        ),
        (r"Parameter Error in $\mathbf{A}$ (log)", "error_A", "log", LEARNING_AGENTS),
        (r"Parameter Error in $\mathbf{B}$ (log)", "error_B", "log", LEARNING_AGENTS),
        (
            r"Spectral Abscissa $\max \mathrm{Re}(\lambda(\mathbf{A}_\star + \mathbf{B}_\star \mathbf{K}_m(0)))$",
            "spectral_abscissa_t0",
            "lin",
            LEARNING_AGENTS,
        ),
        # Row 2
        (
            r"Support F1 in $\mathbf{\Theta}$",
            "support_f1_joint",
            "lin",
            LEARNING_AGENTS,
        ),
        (r"Support F1 in $\mathbf{A}$", "support_f1_A", "lin", LEARNING_AGENTS),
        (r"Support F1 in $\mathbf{B}$", "support_f1_B", "lin", LEARNING_AGENTS),
        (
            r"Gram Min Eigenvalue $\min_i \lambda_{\min}(\mathbf{Z}_{S_i}^\top \mathbf{Z}_{S_i}/N_m)$",
            "gram_min_eig",
            "lin",
            LEARNING_AGENTS,
        ),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)

    fig.suptitle(
        f"$d={exp_config.system.d}$, $p={exp_config.system.p}$, "
        f"$s={exp_config.system.sparsity}$, $M={exp_config.max_episodes}$, "
        f"$T={exp_config.system.T}$, "
        + r"$\mathrm{d}t$="
        + f"{exp_config.system.dt}, "
        + r"$\sigma_x=$"
        + f"{exp_config.system.sigma}, "
        + r"$\sigma_u=$"
        + f"{exp_config.excitation.sigma_u}, "
        + (
            (r"$c_\lambda=$" + f"{exp_config.estimators.c_lambda}, ")
            if exp_config.estimators.lambda_lasso is None
            else ""
        )
        + (
            f"$\lambda={exp_config.estimators.lambda_lasso}$, "
            if exp_config.estimators.lambda_lasso is not None
            else ""
        )
        + f"$\mu={exp_config.estimators.mu_ridge}$, "
        + f"{exp_config.n_seeds} seeds",
        fontsize=11,
    )

    for ax, panel in zip(axes.flat, PANELS):
        title, key, y_scale, agents = panel

        for name in agents:
            if name not in COLORS:
                continue

            # Data extraction
            if key == "cumul_regret":
                data = cumulative_regret_trajectories(results, name, "oracle")
            elif key == "per_ep_regret":
                if name == "oracle":
                    if y_scale == "exp":
                        data = np.ones((len(results), M))
                    else:
                        data = np.zeros((len(results), M))
                else:
                    data = per_episode_regret_trajectories(results, name, "oracle")
            elif key == "episode_cost":
                data = cost_trajectories(results, name)
            else:
                data = aggregate_trajectory(results, name, key)

            if data.size == 0 or np.all(np.isnan(data)):
                continue

            # Plotting lines only where we have valid (non-NaN) data
            valid_mask = ~np.isnan(data[0])
            valid_episodes = episodes[valid_mask]
            if len(valid_episodes) == 0:
                continue

            valid_data = data[:, valid_mask]
            mean, ci_lo, ci_hi = mean_and_ci(valid_data, axis=0)

            ax.plot(
                valid_episodes,
                mean,
                color=COLORS[name],
                label=LABELS[name],
                linewidth=1.6,
            )
            ax.fill_between(
                valid_episodes,
                ci_lo,
                ci_hi,
                color=COLORS[name],
                alpha=0.15,
            )

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Episode", fontsize=8)
        ax.tick_params(labelsize=7)
        if y_scale == "log":
            ax.set_yscale("log", nonpositive="mask")
        elif y_scale == "exp":
            ax.set_yscale(
                "function",
                functions=(
                    lambda x: np.exp(x),
                    lambda x: np.log(np.clip(x, 1.0, None)),
                ),
            )

    axes[0, 0].legend(fontsize=7, loc="upper left")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_basin_entry_comparison(results, exp_config: ExperimentConfig, save_path=None):
    import matplotlib.pyplot as plt

    available = set(results[0].agent_names) if results else set()
    dense_name, sparse_name = "dense_greedy", "sparse_greedy"

    if dense_name not in available or sparse_name not in available:
        return

    # Threshold = max of mean final parameter error across learning agents.
    # We ask "when did each agent first reach the accuracy
    # level that the worst agent achieves at the end of training?"
    learning_agents = [a for a in exp_config.agents if "greedy" in a and a in available]
    final_errors = []
    for name in learning_agents:
        traj = aggregate_trajectory(results, name, "error_joint")
        valid_finals = [float(v) for v in traj[:, -1] if np.isfinite(v)]
        if valid_finals:
            final_errors.append(float(np.mean(valid_finals)))
    threshold = float(max(final_errors)) if final_errors else 0.3

    ratios, median, stats = basin_entry_ratio(
        results, dense_name, sparse_name, threshold=threshold
    )
    theoretical = exp_config.theoretical_speedup

    fig, ax = plt.subplots(figsize=(6, 4))
    valid = ratios[np.isfinite(ratios)]
    if len(valid) > 0:
        ax.hist(valid, bins=20, alpha=0.7, label="Empirical ratios")
        ax.axvline(
            median, color="blue", linestyle="--", label=f"Median: {float(median):.2f}"
        )
    ax.axvline(
        theoretical,
        color="red",
        linestyle="-",
        label=f"Theory: {float(theoretical):.2f}",
    )
    ax.set_xlabel(r"$m_0^{\mathrm{dense}} / m_0^{\mathrm{sparse}}$")
    ax.set_ylabel("Count")
    # Surface the censoring: a high dense-never-entered rate means the median is
    # a conservative lower bound (those seeds are capped at M+1, not dropped).
    ax.set_title(
        f"Basin Entry Speedup ($\\epsilon={threshold:.3f}$)\n"
        f"never entered -- dense: {stats['dense_never_entered']:.0%}, "
        f"sparse: {stats['sparse_never_entered']:.0%}"
    )
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def _build_true_support_mask(d, p, true_supports, block):
    if block == "A":
        mask = np.zeros((d, d), dtype=bool)
        for i, supp in enumerate(true_supports):
            for j in supp:
                if j < d:
                    mask[i, j] = True
    else:
        mask = np.zeros((d, p), dtype=bool)
        for i, supp in enumerate(true_supports):
            for j in supp:
                if j >= d:
                    mask[i, j - d] = True
    return mask


def _draw_support_overlay(ax, mask):
    import matplotlib.patches as mpatches

    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        ax.add_patch(
            mpatches.Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                linewidth=1.2,
                edgecolor="black",
                facecolor="none",
            )
        )


def plot_parameter_evolution(results, exp_config: ExperimentConfig, output_dir):
    """
    For the seed with the best mean final support F1 across learning agents, produce
    two sets of figures per matrix block:

    - params_{block}_best_seed.png : estimated parameter matrices at checkpoints
                                     (RdBu_r, diverging around zero)
    - error_{block}_best_seed.png  : |estimate − truth| at checkpoints
                                     (Reds, starting from zero)
    """
    import matplotlib.pyplot as plt
    import os

    AGENT_LABELS = {
        "dense_greedy": "Dense",
        "dense_excited": "Dense-Ex",
        "sparse_greedy": "Sparse-Gr",
        "sparse_excited": "Sparse-Ex",
    }
    LEARNING_AGENTS = [a for a in exp_config.agents if a in AGENT_LABELS]

    d, p, M = exp_config.system.d, exp_config.system.p, exp_config.max_episodes

    # Select the seed with the highest mean final support F1 across learning agents.
    best_idx, best_score = 0, -np.inf
    for idx, r in enumerate(results):
        scores = []
        for name in LEARNING_AGENTS:
            traj = r.diagnostic_trajectory(name, "support_f1_joint")
            finite = [v for v in traj if np.isfinite(v)]
            if finite:
                scores.append(finite[-1])
        if scores:
            score = float(np.mean(scores))
            if score > best_score:
                best_score, best_idx = score, idx

    n_checkpoints = min(8, M)
    checkpoint_episodes = sorted(
        set(np.round(np.linspace(0, M - 1, n_checkpoints)).astype(int).tolist())
    )

    result = results[best_idx]
    seed = result.seed
    A_true, B_true, supports = result.A_star, result.B_star, result.supports
    vmax = float(np.max(np.abs(np.hstack([A_true, B_true]))))

    mask_A = _build_true_support_mask(d, p, supports, "A")
    mask_B = _build_true_support_mask(d, p, supports, "B")

    shared_suptitle = (
        f"best seed (seed {seed}, F1={best_score:.2f}) — "
        f"d={d}, p={p}, s={exp_config.system.sparsity}, M={M}"
    )

    for block, true_mat, mask, ncols_matrix in [
        ("A", A_true, mask_A, d),
        ("B", B_true, mask_B, p),
    ]:
        n_cols = 1 + len(checkpoint_episodes)
        n_rows = len(LEARNING_AGENTS)
        fig_w = min(n_cols * ncols_matrix * 0.35 + n_cols * 0.15 + 1.5, 28.0)
        fig_h = min(n_rows * d * 0.35 + n_rows * 0.15 + 1.0, 20.0)

        # Each view tuple: (col0_label, col0_mat, est_mat_fn, cmap, vmin, filename, title_prefix)
        VIEWS = [
            (
                "True",
                true_mat,
                lambda est, t=true_mat: est,
                "RdBu_r",
                -vmax,
                f"params_{block}_best_seed.png",
                f"{block} block: estimated parameters",
            ),
            (
                f"|{block}*|",
                np.abs(true_mat),
                lambda est, t=true_mat: np.abs(est - t),
                "Reds",
                0.0,
                f"error_{block}_best_seed.png",
                f"{block} block: |estimate \u2212 truth|",
            ),
        ]

        for (
            col0_label,
            col0_mat,
            est_mat_fn,
            cmap,
            vmin,
            filename,
            title_prefix,
        ) in VIEWS:
            col_titles = [col0_label] + [f"Ep. {m + 1}" for m in checkpoint_episodes]

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(fig_w, fig_h),
                squeeze=False,
                constrained_layout=True,
            )

            for row_idx, agent_name in enumerate(LEARNING_AGENTS):
                for col_idx in range(n_cols):
                    ax = axes[row_idx, col_idx]
                    if col_idx == 0:
                        mat = col0_mat
                    else:
                        ep = result.episodes[agent_name][
                            checkpoint_episodes[col_idx - 1]
                        ]
                        est = ep.diagnostics.get(f"{block}_est", None)
                        if est is None:
                            ax.set_visible(False)
                            continue
                        mat = est_mat_fn(est)

                    ax.imshow(
                        mat,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                        aspect="auto",
                        interpolation="nearest",
                    )
                    _draw_support_overlay(ax, mask)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines[:].set_visible(False)

                    if row_idx == 0:
                        ax.set_title(col_titles[col_idx], fontsize=8, pad=3)
                    if col_idx == 0:
                        ax.set_ylabel(
                            AGENT_LABELS[agent_name],
                            fontsize=8,
                            rotation=90,
                            labelpad=4,
                        )

            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes[:, -1], shrink=0.6, pad=0.02, aspect=20)
            cbar.ax.tick_params(labelsize=7)

            fig.suptitle(f"{title_prefix} — {shared_suptitle}", fontsize=9, y=1.01)
            fig.savefig(
                os.path.join(output_dir, filename), dpi=120, bbox_inches="tight"
            )
            plt.close(fig)


def plot_self_exploration_diagnostics(
    results: list,
    exp_config: ExperimentConfig,
    save_path: str = None,
) -> None:
    """
    Two-panel diagnostic for the self-exploration condition (Basei et al. 2022, Prop 2.1).

    Left:  Scatter of lambda_min(B*^T Q B*) vs final cumulative regret per seed,
           one series per learning agent. Vertical dashed line at zero marks the
           boundary of the sufficient condition for identifiability.

    Right: Histogram of lambda_min across seeds with the same reference line.
    """
    import matplotlib.pyplot as plt
    import os

    COLORS = {
        "dense_greedy": "blue",
        "dense_excited": "purple",
        "sparse_greedy": "red",
        "sparse_excited": "orange",
    }
    LABELS = {
        "dense_greedy": "Dense-Greedy",
        "dense_excited": "Dense-Excitation",
        "sparse_greedy": "Sparse-Greedy",
        "sparse_excited": "Sparse-Excitation",
    }
    MARKERS = {
        "dense_greedy": "o",
        "dense_excited": "s",
        "sparse_greedy": "^",
        "sparse_excited": "D",
    }

    learning_agents = [a for a in exp_config.agents if a != "oracle" and a in COLORS]
    min_eigs = np.array([r.btqb_min_eig for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: scatter min_eig vs final adjusted regret ───────────────
    ax = axes[0]
    for name in learning_agents:
        final_regrets = np.array([r.cumulative_regret(name)[-1] for r in results])
        ax.scatter(
            min_eigs,
            final_regrets,
            color=COLORS[name],
            marker=MARKERS[name],
            label=LABELS[name],
            alpha=0.75,
            s=45,
            zorder=3,
        )
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.6,
        label=r"$\lambda_{\min} = 0$",
    )
    ax.set_xlabel(
        r"$\lambda_{\min}(\mathbf{B}_\star^\top \mathbf{Q} \, \mathbf{B}_\star)$"
    )
    ax.set_ylabel("Final cumulative regret")
    ax.set_title("Self-exploration condition vs regret")
    ax.legend(fontsize=8, framealpha=0.7)

    # ── Right: histogram of min_eig ──────────────────────────────────
    ax = axes[1]
    n_bins = max(10, min(30, len(results) // 3))
    ax.hist(
        min_eigs,
        bins=n_bins,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.6,
        label=r"$\lambda_{\min} = 0$",
    )
    ax.set_xlabel(
        r"$\lambda_{\min}(\mathbf{B}_\star^\top \mathbf{Q} \, \mathbf{B}_\star)$"
    )
    ax.set_ylabel("Count")
    ax.set_title(r"Distribution of $\lambda_{\min}$ across seeds")
    ax.legend(fontsize=8, framealpha=0.7)

    fig.suptitle(
        rf"Self-exploration diagnostics — "
        rf"$d={exp_config.system.d}$, $p={exp_config.system.p}$, "
        rf"$s={exp_config.system.sparsity}$, $N={{{len(results)}}}$ seeds",
        fontsize=11,
    )
    fig.tight_layout()

    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
