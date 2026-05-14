"""
Post-processing: aggregate across seeds, statistical tests, and plotting.
"""

import numpy as np
from scipy import stats
import warnings
import matplotlib

matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{bm}",  # Load the bm package
    }
)


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


def cumulative_regret_trajectories(
    results, agent_name, oracle_name="oracle", adjusted=True
):
    """
    Cumulative regret trajectories, shape (n_seeds, M).

    adjusted=True (default) calls adjusted_cumulative_regret, which subtracts
    the deterministic excitation tax per episode for SparseExcitationAgent.
    Has no effect on other agents (their tax is zero).  Pass adjusted=False
    to retrieve the raw observed regret.
    """
    if adjusted:
        return np.array(
            [r.adjusted_cumulative_regret(agent_name, oracle_name) for r in results]
        )
    return np.array([r.cumulative_regret(agent_name, oracle_name) for r in results])


def cost_trajectories(results, agent_name):
    """
    Per-episode cost trajectories across seeds.

    Returns
    -------
    ndarray, shape (n_seeds, M)
    """
    return np.array([[ep.cost for ep in r.episodes[agent_name]] for r in results])


def per_episode_regret_trajectories(results, agent_name, oracle_name="oracle"):
    """
    Per-episode cost excess r_m = J_m^agent - J_m^oracle, shape (n_seeds, M).

    The cumulative sum of this gives cumulative regret.  Plotting it directly
    shows whether the agent is improving episode by episode.
    """
    return cost_trajectories(results, agent_name) - cost_trajectories(
        results, oracle_name
    )


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


def _final_regret(results, agent_name, oracle_name="oracle", adjusted=True):
    """Per-seed final adjusted (default) cumulative regret."""
    return cumulative_regret_trajectories(
        results, agent_name, oracle_name, adjusted=adjusted
    )[:, -1]


def paired_t_test(results, agent_a, agent_b):
    """Two-sided paired t-test on final adjusted cumulative regret."""
    vals_a = _final_regret(results, agent_a)
    vals_b = _final_regret(results, agent_b)
    diffs = vals_a - vals_b
    t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
    return {"t_stat": t_stat, "p_value": p_val, "mean_diff": float(np.mean(diffs))}


def wilcoxon_test(results, agent_a, agent_b, alternative="greater"):
    """One-sided Wilcoxon signed-rank test on final adjusted cumulative regret."""
    vals_a = _final_regret(results, agent_a)
    vals_b = _final_regret(results, agent_b)
    diffs = vals_a - vals_b
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 2:
        return {"statistic": np.nan, "p_value": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_val = stats.wilcoxon(nonzero, alternative=alternative)
    return {"statistic": stat, "p_value": p_val}


def sign_test(results, agent_a, agent_b):
    """Exact two-sided sign test on final adjusted cumulative regret."""
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
    """Adjusted cumulative regret at episode M/4."""
    trajs = cumulative_regret_trajectories(
        results, agent_name, oracle_name, adjusted=True
    )
    return trajs[:, max(trajs.shape[1] // 4 - 1, 0)]


def episode_average_regret(results, agent_name, oracle_name="oracle"):
    """Episode-averaged adjusted cumulative regret."""
    trajs = cumulative_regret_trajectories(
        results, agent_name, oracle_name, adjusted=True
    )
    return np.mean(trajs, axis=1)


def seed_wins(results, agent_a, agent_b, oracle_name="oracle"):
    """Seeds where agent_b has lower final adjusted regret than agent_a."""
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
    Summary of final-episode statistics.

    Reports both adjusted and raw final cumulative regret for
    sparse_excitation; for all other agents they are identical.
    """
    if agent_names is None:
        agent_names = ["dense_greedy", "sparse_greedy", "sparse_excitation"]

    def _stats(vals):
        n = len(vals)
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        t_crit = float(stats.t.ppf(0.975, df=max(n - 1, 1)))
        return (m, t_crit * se)

    table = {}
    for name in agent_names:
        row = {
            "final_regret": _stats(
                _final_regret(results, name, oracle_name, adjusted=True)
            ),
            "final_regret_raw": _stats(
                _final_regret(results, name, oracle_name, adjusted=False)
            ),
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
            row[key] = _stats(traj[:, -1])
        table[name] = row
    return table


def print_summary(table):
    """Pretty-print summary table with both adjusted and raw regret."""
    header = (
        f"{'Agent':<22} {'Regret(adj)':>14} {'Regret(raw)':>14} "
        f"{'Err(joint)':>12} {'Err(B)':>12} {'F1(joint)':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, row in table.items():

        def fmt(key):
            m, ci = row[key]
            return f"{m:.2f}±{ci:.2f}"

        print(
            f"{name:<22} {fmt('final_regret'):>14} {fmt('final_regret_raw'):>14} "
            f"{fmt('error_joint'):>12} {fmt('error_B'):>12} "
            f"{fmt('support_f1_joint'):>12}"
        )


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────


def plot_trajectories(results, exp_config, save_path=None):
    """
    12-panel (3×4) diagnostic figure; cell (2,3) intentionally empty.

    Row 0: Cumulative Regret | Per-episode Regret | Episode Cost | Gram Min Eig
    Row 1: Error joint (log) | Error A (log)      | Error B (log) | Spectral Abscissa
    Row 2: F1 joint          | F1 A               | F1 B          | (empty)

    For all panels the Oracle line is omitted when it carries no information
    (it is zero for regret panels and undefined for estimation panels).
    """
    import matplotlib.pyplot as plt

    LEARNING_AGENTS = ["dense_greedy", "sparse_greedy", "sparse_excitation"]
    ALL_AGENTS = ["oracle"] + LEARNING_AGENTS
    COLORS = {
        "oracle": "green",
        "dense_greedy": "orange",
        "sparse_greedy": "blue",
        "sparse_excitation": "red",
    }
    LABELS = {
        "oracle": "Oracle",
        "dense_greedy": "Dense-Greedy",
        "sparse_greedy": "Sparse-Greedy",
        "sparse_excitation": "Sparse-Excitation",
    }

    M = len(results[0].episodes["oracle"])
    episodes = np.arange(1, M + 1)

    # ── Panel specification ──────────────────────────────────────────
    # (title, key, log_y, agents_to_plot)
    # key drives the data-fetching branch below.
    PANELS = [
        # Row 0
        (r"Cumulative Regret $R(M)$", "cumul_regret", False, ALL_AGENTS),
        (r"Per-episode Regret $r_m$", "per_ep_regret", False, LEARNING_AGENTS),
        (r"Episode Cost $J(\bm{\pi}_m)$", "episode_cost", False, ALL_AGENTS),
        (
            r"Gram Min Eigenvalue $\min_i \lambda_{\min}((\mathbf{Z}_{S_i})^\top \mathbf{Z}_{S_i}/N_m)$",
            "gram_min_eig",
            False,
            LEARNING_AGENTS,
        ),
        # Row 1
        (
            "Param. Error in $\mathbf{\Theta}$ (log)",
            "error_joint",
            True,
            LEARNING_AGENTS,
        ),
        (r"Param. Error in $\mathbf{A}$ (log)", "error_A", True, LEARNING_AGENTS),
        (r"Param. Error in $\mathbf{B}$ (log)", "error_B", True, LEARNING_AGENTS),
        (
            r"Spectral Abscissa $\max\{\mathrm{Re}(\lambda) : \lambda \in \mathrm{eig}(\mathbf{A}_\star - \mathbf{B}_\star \mathbf{K}_{\widehat{\mathbf{\Theta}}_m}\!(0))\}$",
            "spectral_abscissa_t0",
            False,
            LEARNING_AGENTS,
        ),
        # Row 2
        (
            r"Support F1 in $\mathbf{\Theta}$",
            "support_f1_joint",
            False,
            LEARNING_AGENTS,
        ),
        (r"Support F1 in $\mathbf{A}$", "support_f1_A", False, LEARNING_AGENTS),
        (r"Support F1 in $\mathbf{B}$", "support_f1_B", False, LEARNING_AGENTS),
        None,  # empty cell (2,3)
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
    fig.suptitle(
        f"$d={exp_config.x_dim}$, $p={exp_config.u_dim}$, "
        f"$s={exp_config.sparsity}$, $M={exp_config.max_episodes}$, "
        f"$T={exp_config.T}$, "
        r"$\sigma=$" + f"{exp_config.sigma}, "
        r"$c_\lambda=$" + f"{exp_config.c_lambda}",
        fontsize=11,
    )

    for ax, panel in zip(axes.flat, PANELS):
        if panel is None:
            ax.set_visible(False)
            continue

        title, key, log_y, agents = panel

        for name in agents:
            # ── Data fetch ──────────────────────────────────────────
            if key == "cumul_regret":
                if name == "oracle":
                    data = np.zeros((len(results), M))
                else:
                    # Solid line = adjusted (exploration tax removed)
                    data = cumulative_regret_trajectories(
                        results, name, "oracle", adjusted=True
                    )
            elif key == "per_ep_regret":
                if name == "oracle":
                    data = np.zeros((len(results), M))
                else:
                    raw_ep = per_episode_regret_trajectories(results, name, "oracle")
                    if name == "sparse_excitation":
                        taxes = np.array(
                            [
                                [ep.excitation_tax for ep in r.episodes[name]]
                                for r in results
                            ]
                        )
                        # Solid = adjusted per-episode (tax removed)
                        data = raw_ep - taxes
                    else:
                        data = raw_ep
            elif key == "episode_cost":
                data = cost_trajectories(results, name)
            else:
                data = aggregate_trajectory(results, name, key)

            if data.size == 0:
                continue

            mean, ci_lo, ci_hi = mean_and_ci(data, axis=0)
            ax.plot(
                episodes,
                mean,
                color=COLORS[name],
                label=LABELS[name],
                linewidth=1.6,
            )
            ax.fill_between(
                episodes,
                ci_lo,
                ci_hi,
                color=COLORS[name],
                alpha=0.15,
            )

            # For the excitation agent on both regret panels, overlay the raw
            # trajectory as a dashed line so the exploration tax is visible.
            if key == "cumul_regret" and name == "sparse_excitation":
                raw = cumulative_regret_trajectories(
                    results, name, "oracle", adjusted=False
                )
                raw_mean, _, _ = mean_and_ci(raw, axis=0)
                ax.plot(
                    episodes,
                    raw_mean,
                    color=COLORS[name],
                    linestyle="--",
                    linewidth=1.0,
                    label="Sparse-Excitation (raw)",
                    alpha=0.55,
                )
            elif key == "per_ep_regret" and name == "sparse_excitation":
                raw_ep = per_episode_regret_trajectories(results, name, "oracle")
                raw_mean, _, _ = mean_and_ci(raw_ep, axis=0)
                ax.plot(
                    episodes,
                    raw_mean,
                    color=COLORS[name],
                    linestyle="--",
                    linewidth=1.0,
                    label="Sparse-Excitation (raw)",
                    alpha=0.55,
                )

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Episode", fontsize=8)
        ax.tick_params(labelsize=7)
        if log_y:
            ax.set_yscale("log")

    # Legend on cumulative regret panel (top-left)
    axes[0, 0].legend(fontsize=7, loc="upper left")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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


def _build_true_support_mask(d, p, true_supports, block):
    """Return boolean mask (d, d) for A or (d, p) for B true nonzeros."""
    if block == "A":
        mask = np.zeros((d, d), dtype=bool)
        for i, supp in enumerate(true_supports):
            for j in supp:
                if j < d:
                    mask[i, j] = True
    else:  # B
        mask = np.zeros((d, p), dtype=bool)
        for i, supp in enumerate(true_supports):
            for j in supp:
                if j >= d:
                    mask[i, j - d] = True
    return mask


def _draw_support_overlay(ax, mask):
    """Draw thin black rectangles around each True cell in mask."""
    import matplotlib.patches as mpatches

    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        rect = mpatches.Rectangle(
            (c - 0.5, r - 0.5),
            1,
            1,
            linewidth=1.2,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)


def plot_sparsity_evolution(results, exp_config, output_dir):
    """
    For each seed, save two heatmap figures (A block and B block) showing
    how the estimated sparsity pattern evolves over checkpoint episodes.

    Layout: 3 rows (dense_greedy, sparse_greedy, sparse_excitation),
    first column = true matrix, remaining columns = estimates at
    evenly-spaced checkpoint episodes.

    Colormap: diverging grey (black = large negative, mid-grey = zero,
    white = large positive), symmetric scale from true matrix.

    True nonzero cells are outlined with a thin black rectangle.

    Files saved as:
        {output_dir}/sparsity_A_seed{seed}.png
        {output_dir}/sparsity_B_seed{seed}.png
    """
    import matplotlib.pyplot as plt
    import os

    import warnings

    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:
        pass

    LEARNING_AGENTS = ["dense_greedy", "sparse_greedy", "sparse_excitation"]
    AGENT_LABELS = {
        "dense_greedy": "Dense-Greedy",
        "sparse_greedy": "Sparse-Greedy",
        "sparse_excitation": "Sparse-Excitation",
    }

    d = exp_config.x_dim
    p = exp_config.u_dim
    M = exp_config.max_episodes

    # Checkpoint episode indices (same logic as runner)
    n_checkpoints = min(8, M)
    checkpoint_episodes = sorted(
        set(np.round(np.linspace(0, M - 1, n_checkpoints)).astype(int).tolist())
    )

    for result in results:
        seed = result.seed
        A_true = result.A_star
        B_true = result.B_star
        supports = result.supports

        # Shared colour scale: symmetric about 0, derived from true matrices
        Theta_true = np.hstack([A_true, B_true])
        vmax = float(np.max(np.abs(Theta_true)))
        if vmax < 1e-10:
            vmax = 1.0

        # True support masks for overlay
        mask_A = _build_true_support_mask(d, p, supports, "A")
        mask_B = _build_true_support_mask(d, p, supports, "B")

        for block, true_mat, mask, ncols_matrix in [
            ("A", A_true, mask_A, d),
            ("B", B_true, mask_B, p),
        ]:
            n_cols = 1 + len(checkpoint_episodes)  # true + checkpoints
            n_rows = len(LEARNING_AGENTS)

            # Scale figure so each cell is roughly 0.35 inches square
            cell_size = 0.35
            fig_w = n_cols * ncols_matrix * cell_size + n_cols * 0.15 + 1.5
            fig_h = n_rows * d * cell_size + n_rows * 0.15 + 1.0
            # Cap to reasonable size
            fig_w = min(fig_w, 28.0)
            fig_h = min(fig_h, 20.0)

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(fig_w, fig_h),
                squeeze=False,
                constrained_layout=True,
            )

            # Column titles
            col_titles = ["True"] + [f"Ep. {m + 1}" for m in checkpoint_episodes]

            imshow_kwargs = dict(
                vmin=-vmax,
                vmax=vmax,
                cmap="RdBu_r",
                aspect="auto",
                interpolation="nearest",
            )

            for row_idx, agent_name in enumerate(LEARNING_AGENTS):
                for col_idx in range(n_cols):
                    ax = axes[row_idx, col_idx]

                    if col_idx == 0:
                        # True matrix
                        mat = true_mat
                    else:
                        ep_idx = checkpoint_episodes[col_idx - 1]
                        ep = result.episodes[agent_name][ep_idx]
                        mat = ep.diagnostics.get(f"{block}_est", None)
                        if mat is None:
                            # Snapshot not available: show blank
                            ax.set_visible(False)
                            continue

                    ax.imshow(mat, **imshow_kwargs)
                    _draw_support_overlay(ax, mask)

                    # Axis formatting
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

            # Colourbar on the right
            sm = plt.cm.ScalarMappable(
                cmap="RdBu_r",
                norm=plt.Normalize(vmin=-vmax, vmax=vmax),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes[:, -1], shrink=0.6, pad=0.02, aspect=20)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label("Coefficient value", fontsize=7)

            fig.suptitle(
                f"{block} block — seed {seed} — "
                f"d={d}, p={p}, s={exp_config.sparsity}, "
                f"M={M}",
                fontsize=9,
                y=1.01,
            )
            save_path = os.path.join(output_dir, f"params_{block}_seed{seed}.png")
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig)


def plot_error_evolution(results, exp_config, output_dir):
    """
    For each seed, save two heatmap figures showing how the absolute
    estimation error |A_hat - A_true| and |B_hat - B_true| evolves.

    Layout: 3 rows (dense_greedy, sparse_greedy, sparse_excitation),
    first column = |A_true| or |B_true| as a reference for scale,
    remaining columns = |A_hat_m - A_true| at checkpoint episodes.

    Colormap: Reds, vmin=0, vmax = max absolute coefficient in Theta_true.
    This is shared with plot_sparsity_evolution so magnitudes are comparable.
    True support overlay retained: rectangles mark where the true matrix
    has nonzero entries, i.e. where Lasso shrinkage bias is expected.

    Files saved as:
        {output_dir}/error_A_seed{seed}.png
        {output_dir}/error_B_seed{seed}.png
    """
    import matplotlib.pyplot as plt
    import os

    LEARNING_AGENTS = ["dense_greedy", "sparse_greedy", "sparse_excitation"]
    AGENT_LABELS = {
        "dense_greedy": "Dense-Greedy",
        "sparse_greedy": "Sparse-Greedy",
        "sparse_excitation": "Sparse-Excitation",
    }

    d = exp_config.x_dim
    p = exp_config.u_dim
    M = exp_config.max_episodes

    n_checkpoints = min(8, M)
    checkpoint_episodes = sorted(
        set(np.round(np.linspace(0, M - 1, n_checkpoints)).astype(int).tolist())
    )

    for result in results:
        seed = result.seed
        A_true = result.A_star
        B_true = result.B_star
        supports = result.supports

        # Shared scale: max absolute true coefficient, identical to
        # plot_sparsity_evolution so errors are directly comparable to values.
        Theta_true = np.hstack([A_true, B_true])
        vmax = float(np.max(np.abs(Theta_true)))
        if vmax < 1e-10:
            vmax = 1.0

        mask_A = _build_true_support_mask(d, p, supports, "A")
        mask_B = _build_true_support_mask(d, p, supports, "B")

        for block, true_mat, mask, ncols_matrix in [
            ("A", A_true, mask_A, d),
            ("B", B_true, mask_B, p),
        ]:
            n_cols = 1 + len(checkpoint_episodes)
            n_rows = len(LEARNING_AGENTS)

            cell_size = 0.35
            fig_w = min(n_cols * ncols_matrix * cell_size + n_cols * 0.15 + 1.5, 28.0)
            fig_h = min(n_rows * d * cell_size + n_rows * 0.15 + 1.0, 20.0)

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(fig_w, fig_h),
                squeeze=False,
                constrained_layout=True,
            )

            col_titles = [f"|{block}*|"] + [f"Ep. {m + 1}" for m in checkpoint_episodes]

            # Sequential colormap: white = zero error, deep red = large error
            imshow_kwargs = dict(
                vmin=0,
                vmax=vmax,
                cmap="Reds",
                aspect="auto",
                interpolation="nearest",
            )

            for row_idx, agent_name in enumerate(LEARNING_AGENTS):
                for col_idx in range(n_cols):
                    ax = axes[row_idx, col_idx]

                    if col_idx == 0:
                        # Reference column: absolute true matrix values
                        mat = np.abs(true_mat)
                    else:
                        ep_idx = checkpoint_episodes[col_idx - 1]
                        ep = result.episodes[agent_name][ep_idx]
                        est = ep.diagnostics.get(f"{block}_est", None)
                        if est is None:
                            ax.set_visible(False)
                            continue
                        mat = np.abs(est - true_mat)

                    ax.imshow(mat, **imshow_kwargs)
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
                cmap="Reds",
                norm=plt.Normalize(vmin=0, vmax=vmax),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes[:, -1], shrink=0.6, pad=0.02, aspect=20)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label("Absolute error", fontsize=7)

            fig.suptitle(
                f"|{block}_hat - {block}*| — seed {seed} — "
                f"d={d}, p={p}, s={exp_config.sparsity}, M={M}",
                fontsize=9,
                y=1.01,
            )

            save_path = os.path.join(output_dir, f"error_{block}_seed{seed}.png")
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
