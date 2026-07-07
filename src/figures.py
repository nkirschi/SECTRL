"""
Publication figures, separate from the dev dashboard in dashboard.py.

Each figure is a focused, vector (PDF), median+IQR, colourblind-safe (Okabe-Ito)
plot with log scales where appropriate and a caption-ready layout. Single-config
figures take one (results, config) tuple, while scaling figures take a
{d: (results, config)} sweep.

CLI:  uv run python src/figures.py --focal-d 20
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results_io import load_study, load_point  # unified loaders

# Real LaTeX so the figure maths matches the thesis (Computer Modern + the same
# math packages the document loads).
matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
    }
)


def _save(fig, path):
    """Save `fig` as `path` (a .pdf, the thesis artefact)."""
    fig.savefig(path)  # PDF; format inferred from the extension


OK = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
# agent -> (colour, linestyle, marker, label)
STYLE = {
    "oracle": (OK["black"], ":", None, "oracle"),
    "dense_greedy": (OK["blue"], "-", "o", "dense-greedy"),
    "dense_excited": (OK["sky"], "--", "s", "dense-excited"),
    "sparse_greedy": (OK["vermillion"], "-", "o", "sparse-greedy"),
    "sparse_excited": (OK["orange"], "--", "^", "sparse-excited"),
}
LEARNING = ["dense_greedy", "dense_excited", "sparse_greedy", "sparse_excited"]


def _med_iqr(a, axis=0):
    return (
        np.nanmedian(a, axis),
        np.nanpercentile(a, 25, axis),
        np.nanpercentile(a, 75, axis),
    )


def _final_regret(res, agent):
    return np.array([r.cumulative_regret(agent)[-1] for r in res])


def _traj(res, agent, key):
    return np.array([r.diagnostic_trajectory(agent, key) for r in res], dtype=float)


def _basin_speedup(res, dense="dense_greedy", sparse="sparse_greedy"):
    # V2 basin-entry speedup: per seed, the dense/sparse ratio of the first episode the joint
    # parameter error falls within the *larger* of the two agents' final errors -- a common,
    # always-attainable accuracy, so no seed is censored (unlike a fixed radius, which is
    # unreached at high d). Ratio > 1 means the sparse agent enters the basin from fewer samples.
    ed, es = _traj(res, dense, "error_joint"), _traj(res, sparse, "error_joint")
    out = []
    for d, s in zip(ed, es):
        thr = max(d[-1], s[-1])
        out.append((int(np.argmax(d <= thr)) + 1) / (int(np.argmax(s <= thr)) + 1))
    return np.array(out)


def _per_ep_regret(res, agent):
    a = np.array([[ep.cost for ep in r.episodes[agent]] for r in res])
    o = np.array([[ep.cost for ep in r.episodes["oracle"]] for r in res])
    return a - o


# ----------------------------------------------------------------- A. scaling
def fig_scaling_regret(sweep, outdir, fname="scaling_regret.pdf", suptitle=None):
    # Emits two single-panel PDFs <base>_a.pdf (regret scaling) and <base>_b.pdf (advantage
    # vs d), for assembly with LaTeX subfigures. No in-figure panel titles.
    ds = sorted(sweep)
    base = fname[:-4] if fname.endswith(".pdf") else fname

    # (a) dense-to-sparse regret ratio -- scale-independent, matched per seed via CRN
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for (da, sa), c, lst, mk, lab in [
        (("dense_greedy", "sparse_greedy"), OK["blue"], "-", "o", "greedy"),
        (("dense_excited", "sparse_excited"), OK["orange"], "--", "s", "excited"),
    ]:
        med, lo, hi = [], [], []
        for d in ds:
            rr = _final_regret(sweep[d][0], da) / _final_regret(sweep[d][0], sa)
            med.append(np.median(rr))
            lo.append(np.percentile(rr, 25))
            hi.append(np.percentile(rr, 75))
        ax.plot(ds, med, color=c, ls=lst, marker=mk, ms=5, label=lab)
        ax.fill_between(ds, lo, hi, color=c, alpha=0.15)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel("dense-to-sparse regret ratio")
    ax.set_xticks(ds)
    ax.set_xticklabels(ds)
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, f"{base}_a.pdf"))
    plt.close(fig)

    # (b) basin-entry speedup vs d -- the literal sample-complexity speedup the theory
    #     predicts (cor:basin), more direct than the regret ratio. V2 (final-error) threshold.
    spd_med, lo, hi = [], [], []
    for d in ds:
        r = _basin_speedup(sweep[d][0])
        spd_med.append(np.median(r))
        lo.append(np.percentile(r, 25))
        hi.append(np.percentile(r, 75))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(ds, spd_med, color=OK["vermillion"], marker="o", ms=5, label="empirical")
    ax.fill_between(ds, lo, hi, color=OK["vermillion"], alpha=0.15)
    s0 = sweep[ds[0]][1].system
    spd = np.array(
        [
            (d + sweep[d][1].system.p)
            / ((s0.s_A + s0.s_B) * np.log(d + sweep[d][1].system.p))
            for d in ds
        ]
    )
    cst = float(
        np.dot(np.array(spd_med) - 1.0, spd) / np.dot(spd, spd)
    )  # excess ~ c * speedup
    ax.plot(
        ds,
        1.0 + cst * spd,
        color="grey",
        ls="--",
        lw=1,
        label=r"theory $\propto (d{+}p)/(s\log(d{+}p))$",
    )
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel(r"basin-entry speedup $m_0^{\mathrm{dense}}/m_0^{\mathrm{sparse}}$")
    ax.set_xticks(ds)
    ax.set_xticklabels(ds)
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, f"{base}_b.pdf"))
    plt.close(fig)


# -------------------------------------- exploration ablation (m_explore 0 vs 1)
def fig_exploration_ablation(dirs0, dirs1, outdir, fname="exploration.pdf"):
    """Two panels contrasting pure greedy certainty equivalence (m_explore=0) with one
    initial pure-exploration episode (m_explore=1), across dimension. (a) basin-entry
    speedup: greedy dives below 1 at high d (the advantage reverses), the kick holds and
    widens. (b) control-block restricted-Gram minimum eigenvalue: the greedy design loses
    rank (self-exploration extinction), the kick keeps it excited. dirs0/dirs1 map d -> a
    results directory; points are loaded one at a time to stay light."""
    import gc

    def collect(dirs):
        ds, spd, gram = [], [], []
        for d in sorted(dirs):
            try:
                res, _ = load_point(dirs[d])
            except Exception:
                continue
            ds.append(d)
            spd.append(float(np.median(_basin_speedup(res))))
            gram.append(
                float(
                    np.median(
                        [
                            r.diagnostic_trajectory("sparse_greedy", "gram_min_eig")[-1]
                            for r in res
                        ]
                    )
                )
            )
            del res
            gc.collect()
        return ds, spd, gram

    d0, spd0, g0 = collect(dirs0)
    d1, spd1, g1 = collect(dirs1)
    base = fname[:-4] if fname.endswith(".pdf") else fname
    C0, C1 = OK["vermillion"], OK["blue"]
    L0, L1 = (
        r"$m_{\mathrm{explore}}=0$ (no exploration)",
        r"$m_{\mathrm{explore}}=1$ (first episode exploration)",
    )

    # (a) basin-entry speedup vs d
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(d0, spd0, color=C0, marker="o", ms=5, label=L0)
    ax.plot(d1, spd1, color=C1, marker="s", ms=5, label=L1)
    ax.axhline(
        1.0, color="grey", ls=":", lw=1
    )  # break-even: below = advantage reversed
    ax.set_xscale("log")
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel(r"basin-entry speedup $m_0^{\mathrm{dense}}/m_0^{\mathrm{sparse}}$")
    ax.set_xticks(d0 or d1)
    ax.set_xticklabels(d0 or d1)
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, f"{base}_a.pdf"))
    plt.close(fig)

    # (b) control-block restricted-Gram minimum eigenvalue vs d
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(d0, g0, color=C0, marker="o", ms=5, label=L0)
    ax.plot(d1, g1, color=C1, marker="s", ms=5, label=L1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel(
        r"$\min_i \lambda_{\min}(\mathbf{Z}_{S_i}^\top \mathbf{Z}_{S_i}/N_m)$"
    )
    ax.set_xticks(d0 or d1)
    ax.set_xticklabels(d0 or d1)
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, f"{base}_b.pdf"))
    plt.close(fig)


# ------------------------------------------------ discretisation ablation (vs dt)
def fig_discretisation(study_dir, outdir, fname="discretisation.pdf"):
    """Two panels versus the Euler step dt at fixed d: (a) final regret (dense & sparse);
    (b) the drift/control error split. The estimate is essentially dt-invariant -- finer dt
    buys proportionally more but noisier samples, whose aggregate information cancels -- so the
    drift error is flat; only the control block degrades, and only at the coarsest dt."""
    import gc

    pts = []
    for name in sorted(os.listdir(study_dir)):
        d = os.path.join(study_dir, name)
        if not os.path.isdir(d) or not any(
            f.startswith("seed_") and not f.endswith("_snapshots.npz") for f in os.listdir(d)
        ):
            continue
        res, cfg = load_point(d)
        dg = np.array([r.cumulative_regret("dense_greedy")[-1] for r in res])
        sg = np.array([r.cumulative_regret("sparse_greedy")[-1] for r in res])
        eA = np.array([r.diagnostic_trajectory("sparse_greedy", "error_A")[-1] for r in res])
        eB = np.array([r.diagnostic_trajectory("sparse_greedy", "error_B")[-1] for r in res])
        # each entry: (dt, (med, q25, q75) x {dense reg, sparse reg, err_A, err_B})
        pts.append((cfg.system.dt, _med_iqr(dg), _med_iqr(sg), _med_iqr(eA), _med_iqr(eB)))
        del res
        gc.collect()
    pts.sort()
    dts = [p[0] for p in pts]
    base = fname[:-4] if fname.endswith(".pdf") else fname

    def _band(ax, xs, idx, color, marker, label):
        med = [p[idx][0] for p in pts]
        lo = [p[idx][1] for p in pts]
        hi = [p[idx][2] for p in pts]
        ax.plot(xs, med, color=color, marker=marker, ms=5, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)

    # (a) final regret, dense vs sparse (median, IQR shaded)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _band(ax, dts, 1, STYLE["dense_greedy"][0], "o", "dense-greedy")
    _band(ax, dts, 2, STYLE["sparse_greedy"][0], "o", "sparse-greedy")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"time step $\Delta t$")
    ax.set_ylabel(r"final cumulative regret $R_M$")
    ax.set_xticks(dts)
    ax.set_xticklabels(dts)
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, f"{base}_a.pdf"))
    plt.close(fig)

    # (b) drift vs control error (median, IQR shaded)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _band(ax, dts, 3, OK["green"], "o", r"drift err$_A$")
    _band(ax, dts, 4, OK["vermillion"], "s", r"control err$_B$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"time step $\Delta t$")
    ax.set_ylabel("relative parameter error")
    ax.set_xticks(dts)
    ax.set_xticklabels(dts)
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, f"{base}_b.pdf"))
    plt.close(fig)


# ----------------------------------------------------- B. regret decomposition
def _knee_band(sm, M, d, fac=1.8):
    """Basin-entry knee = the episode of steepest log-log descent of the smoothed median
    per-episode regret, i.e. the visual transition from the transient to the quadratic-basin
    tail. This is defined by the curve's own shape rather than by proximity to the final
    value (which, because the tail keeps improving, lands far too late at low dimension).
    The slope is a central difference over a multiplicative window `fac`; the tail (last
    15%) is excluded so its noise floor cannot masquerade as a knee. Returns a log-symmetric
    band around the knee, widened with d for visibility on the log-x axis, or None if no
    descent clearly steeper than the m^-1 tail exists (e.g. a non-converging curve)."""
    if sm is None or len(sm) < 24:
        return None
    m = np.arange(1, M + 1)
    lr = np.log(np.clip(sm, 1e-12, None))
    lm = np.log(m)
    slopes = np.full(M, np.nan)
    for i in range(M):
        hi = min(M - 1, int(round((i + 1) * fac)) - 1)
        lo = max(0, int(round((i + 1) / fac)) - 1)
        if hi > lo:
            slopes[i] = (lr[hi] - lr[lo]) / (lm[hi] - lm[lo])
    seg = slopes[: int(0.85 * M)]
    if not np.any(np.isfinite(seg)):
        return None
    j = int(np.nanargmin(seg))
    if not (slopes[j] < -0.8):  # nothing steeper than the m^-1 tail => no clear knee
        return None
    m_k = j + 1
    hw = 0.04 + 0.03 * np.log10(max(d, 2))
    return (m_k * 10.0 ** (-hw), m_k * 10.0**hw)


def fig_regret_decomposition(res, cfg, outdir, fname="regret_phases.pdf", legend=True):
    M = cfg.max_episodes
    m = np.arange(1, M + 1)

    def smooth(y, w=5):  # centred rolling median to expose the rate above the noise
        return np.array(
            [np.median(y[max(0, i - w // 2) : i + w // 2 + 1]) for i in range(len(y))]
        )

    fig, ax = plt.subplots(figsize=(6, 4.2))
    sms = {}
    for agent in ("dense_greedy", "sparse_greedy"):
        sm = smooth(np.median(_per_ep_regret(res, agent), axis=0))
        sms[agent] = sm
        c, ls, mk, lab = STYLE[agent]
        pos = sm > 0
        ax.plot(m[pos], sm[pos], color=c, ls=ls, lw=1.6, label=lab)
    sparse_sm = sms["sparse_greedy"]
    # reference slopes anchored to overlay the sparse curve's two regimes
    for exp_, anc, lab, st in [
        (0.5, 5, r"$\propto m^{-1/2}$ (transient)", dict(ls="--", lw=1)),
        (1.0, M // 2, r"$\propto m^{-1}$ (tail)", dict(ls=":", lw=1.2)),
    ]:
        v = sparse_sm[anc - 1]
        ax.plot(m, v * (anc / m) ** exp_, color="grey", label=lab, **st)
    # shade each agent's basin-entry knee (steepest log-log descent) in its own colour
    _d = res[0].A_star.shape[0]
    for _agent in ("dense_greedy", "sparse_greedy"):
        band = _knee_band(sms[_agent], M, _d)
        if band:
            _c, _, _, _lab = STYLE[_agent]
            ax.axvspan(
                *band, color=_c, alpha=0.16, lw=0, zorder=0, label=f"{_lab} transition"
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(1e-2, np.nanmin(sparse_sm[sparse_sm > 0]) * 0.5))
    ax.set_xlabel("episode $m$")
    ax.set_ylabel(r"per-episode regret $r_m$ (median)")
    if legend:
        ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, fname))
    plt.close(fig)


# ----------------------------------------------------------- C. speedup vs d
def fig_speedup_vs_d(sweep, outdir):
    from metrics import basin_entry_ratio

    ds = sorted(sweep)
    emp_med, emp_lo, emp_hi, theo = [], [], [], []
    for d in ds:
        res, cfg = sweep[d]
        # threshold = worst greedy final joint error (same rule as the dashboard)
        finals = [
            np.nanmedian(_traj(res, a, "error_joint")[:, -1])
            for a in ("dense_greedy", "sparse_greedy")
        ]
        thr = float(max(finals))
        ratios, med, _ = basin_entry_ratio(
            res, "dense_greedy", "sparse_greedy", threshold=thr
        )
        r = ratios[np.isfinite(ratios)]
        emp_med.append(med)
        emp_lo.append(np.percentile(r, 25) if len(r) else np.nan)
        emp_hi.append(np.percentile(r, 75) if len(r) else np.nan)
        theo.append(cfg.theoretical_speedup)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(
        ds, emp_med, color=OK["vermillion"], marker="o", ms=5, label="empirical median"
    )
    ax.fill_between(ds, emp_lo, emp_hi, color=OK["vermillion"], alpha=0.15, label="IQR")
    ax.plot(
        ds,
        theo,
        color=OK["black"],
        ls="--",
        marker="x",
        label=r"theory $(d{+}p)/(s\log(d{+}p))$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel(r"basin-entry speedup $m_0^{d}/m_0^{s}$")
    ax.set_xticks(ds)
    ax.set_xticklabels(ds)
    ax.set_title("(c) Basin-entry speedup vs theory")
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "speedup_vs_d.pdf"))
    plt.close(fig)


# ------------------------------------------------------------ D. A/B asymmetry
def fig_ab_asymmetry(res, cfg, outdir):
    M = cfg.max_episodes
    m = np.arange(1, M + 1)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for key, col, lab in [
        (
            "error_A",
            OK["green"],
            r"drift $\|\hat{\mathbf{A}}-\mathbf{A}_\star\|_F/\|\mathbf{A}_\star\|_F$",
        ),
        (
            "error_B",
            OK["vermillion"],
            r"control $\|\hat{\mathbf{B}}-\mathbf{B}_\star\|_F/\|\mathbf{B}_\star\|_F$",
        ),
    ]:
        med, lo, hi = _med_iqr(_traj(res, "sparse_greedy", key))
        ax.plot(m, med, color=col, lw=1.6, label=lab)
        ax.fill_between(m, lo, hi, color=col, alpha=0.15)
        # excited agent (dashed): probing targets the hard control block, so the
        # control error drops below greedy while the easy drift error is unchanged
        mede = np.nanmedian(_traj(res, "sparse_excited", key), axis=0)
        ax.plot(m, mede, color=col, lw=1.4, ls="--")
    ax.plot([], [], color="grey", lw=1.6, label="sparse-greedy")
    ax.plot([], [], color="grey", lw=1.4, ls="--", label="sparse-excited")
    ax.set_yscale("log")
    ax.set_ylim(
        top=1.2
    )  # clip the early transient to expose the late-episode plateau gap
    ax.set_xlabel("episode $m$")
    ax.set_ylabel("relative parameter error")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "ab_asymmetry.pdf"))
    plt.close(fig)


# ------------------------------------------------------ E. excitation restores RE
# ----------------------------------------------------------- F. cond(B^T Q B)
def fig_btqb_cond(sweep, outdir):
    # The artifact that motivates equal-authority scaling: raw random-sparse control
    # matrices become ill-conditioned as p grows. Sampled fresh un-normalised (the
    # systems actually used in the experiments are normalised to kappa ~ 1).
    from system_generator import sample_synthetic_system

    # Raw (un-normalised) sampling is slow at large d (~40s/system at d=500) but succeeds in
    # a single attempt; the conditioning trend is the point, so sweep the full range.
    ds = sorted(sweep)
    med, lo, hi = [], [], []
    for d in ds:
        cfg = sweep[d][1].system
        conds = []
        for seed in range(20):  # 20 seeds is the project default
            _, B, _, _ = sample_synthetic_system(
                d, cfg.p, cfg.s_A, cfg.s_B, seed, 0.1, 1.9, 0.1, 1.9, normalise_B=False
            )
            conds.append(np.linalg.cond(B.T @ B))
        conds = np.array(conds)
        med.append(np.median(conds))
        lo.append(np.percentile(conds, 25))
        hi.append(np.percentile(conds, 75))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(ds, med, color=OK["purple"], marker="o", ms=5, label="median")
    ax.fill_between(ds, lo, hi, color=OK["purple"], alpha=0.15, label="IQR")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel(r"$\kappa(\mathbf{B}_\star^\top \mathbf{Q}\, \mathbf{B}_\star)$")
    ax.set_xticks(ds)
    ax.set_xticklabels(ds)
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "btqb_condition.pdf"))
    plt.close(fig)


# ------------------------------------------- G. regularisation-strength sweep
def fig_hyperparam(study_dir, outdir, chosen=0.02):
    """Sparse-Greedy regret (a) and late-episode parameter error (b) vs the LASSO constant
    c_lambda, one curve per d. The two optima can diverge: at larger d the regret-optimal
    c_lambda drifts upward (more shrinkage stabilises the gain) while the parameter-error
    optimum stays near c_lambda=0.02, so 0.02 is chosen as the estimation-faithful value.
    Reads a c_lambda x d study (results/clambda)."""
    study = load_study(study_dir)
    if not study:
        return

    def _mq(x):  # median and IQR across seeds
        return (
            float(np.median(x)),
            float(np.percentile(x, 25)),
            float(np.percentile(x, 75)),
        )

    by_d_reg, by_d_err = {}, {}
    for res, cfg in study.values():
        cl, d = cfg.estimators.c_lambda, cfg.system.d
        by_d_reg.setdefault(d, []).append(
            (cl, *_mq(_final_regret(res, "sparse_greedy")))
        )
        traj = _traj(res, "sparse_greedy", "error_joint")  # (seeds, M)
        by_d_err.setdefault(d, []).append((cl, *_mq(np.mean(traj[:, -20:], axis=1))))
    palette = (
        OK["green"],
        OK["blue"],
        OK["vermillion"],
        OK["purple"],
        OK["sky"],
        OK["orange"],
    )
    panels = [
        (by_d_reg, "sparse-greedy regret $R_M$", "hyperparameter_a.pdf", True),
        (
            by_d_err,
            r"parameter error $\|\hat{\mathbf{\Theta}}-\mathbf{\Theta}_\star\|_F/\|\mathbf{\Theta}_\star\|_F$",
            "hyperparameter_b.pdf",
            False,
        ),
    ]
    for by_d, ylab, fname, show_legend in panels:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for d, col in zip(sorted(by_d), palette):
            pts = sorted(by_d[d])
            cls = [p[0] for p in pts]
            ax.plot(
                cls, [p[1] for p in pts], color=col, marker="o", ms=4, label=f"$d={d}$"
            )
            ax.fill_between(
                cls, [p[2] for p in pts], [p[3] for p in pts], color=col, alpha=0.15
            )
        ax.axvline(
            chosen,
            color="black",
            ls="--",
            lw=1.2,
            label=rf"chosen $c_\lambda={chosen}$",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"regularisation constant $c_\lambda$")
        ax.set_ylabel(ylab)
        if show_legend:
            ax.legend()
        fig.tight_layout()
        _save(fig, os.path.join(outdir, fname))
        plt.close(fig)


# ------------------------------------------------ K. excitation-strength sweep
def fig_excitation(study_dir, outdir, chosen=0.05):
    """Sparse-Excited final regret vs the excitation scale sigma_u, one curve per d
    (sigma_u=0 reduces the Excited agent to Greedy, so the leftmost point is the
    no-probing baseline). Analogous to fig_hyperparam, but the message is the opposite
    of a sweet spot: at the canonical cost the curves are flat-to-rising in sigma_u, so
    deliberate probing does not pay -- the closed-loop process noise already excites the
    system enough -- and large sigma_u clearly hurts. (Probing only pays when control is
    expensive; see fig_cost.) The marked sigma_u=0.05 is the level the Excited agents use
    in the other studies. Reads an excitation study (results/excitation)."""
    study = load_study(study_dir)
    if not study:
        return
    by_d = {}
    for res, cfg in study.values():
        fr = _final_regret(res, "sparse_excited")
        by_d.setdefault(cfg.system.d, []).append((cfg.excitation.sigma_u, *_med_iqr(fr)))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    palette = (
        OK["green"],
        OK["blue"],
        OK["vermillion"],
        OK["purple"],
        OK["sky"],
        OK["orange"],
    )
    for d, col in zip(sorted(by_d), palette):
        pts = sorted(by_d[d])
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], color=col, marker="o", ms=4, label=f"$d={d}$")
        ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=col, alpha=0.15)
    ax.axvline(
        chosen, color="black", ls="--", lw=1.2, label=rf"chosen $\sigma_u={chosen}$"
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"excitation scale $\sigma_u$")
    ax.set_ylabel("sparse-excited regret $R_M$")
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "excitation.pdf"))
    plt.close(fig)


# ------------------------------------------------- cost-weighting (Q/R) sweep
def fig_cost(study_dir, outdir):
    """Cost-weighting (Q/R) sweep at fixed dimension. (a) dense/sparse final-regret ratio vs
    the control-cost scale r (with q=1; only the ratio q/r is meaningful by LQR scale-
    invariance). (b) per-episode regret at the largest r: greedy certainty-equivalence suffers
    a transient early destabilization that excitation prevents -- so the cumulative ratio in
    (a) at high r is dominated by this early spike, not by steady-state performance (which the
    sparse agent in fact wins). The early estimate matters because timid high-cost gains make a
    cold, slightly-wrong estimate momentarily destabilising."""
    from metrics import per_episode_regret_trajectories

    study = load_study(study_dir)
    if not study:
        return
    pts = []
    for res, cfg in study.values():
        rg = _final_regret(res, "dense_greedy") / _final_regret(res, "sparse_greedy")
        re = _final_regret(res, "dense_excited") / _final_regret(res, "sparse_excited")
        pts.append((cfg.cost.r_scale, _med_iqr(rg), _med_iqr(re)))
    pts.sort()
    rs = [p[0] for p in pts]

    # (a) advantage vs cost weighting (per-seed ratio; median, IQR shaded)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for idx, col, mk, ls_, lab in [
        (1, OK["vermillion"], "o", "-", "greedy"),
        (2, OK["orange"], "^", "--", "excited"),
    ]:
        med = [p[idx][0] for p in pts]
        lo = [p[idx][1] for p in pts]
        hi = [p[idx][2] for p in pts]
        ax.plot(rs, med, color=col, marker=mk, ms=4, ls=ls_, label=lab)
        ax.fill_between(rs, lo, hi, color=col, alpha=0.15)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"control-cost scale $r$ (with $q=1$)")
    ax.set_ylabel("dense / sparse regret ratio")
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "cost_a.pdf"))
    plt.close(fig)

    # (b) per-episode regret at the largest r -- the transient early destabilization
    hi_res, hi_cfg = max(study.values(), key=lambda rc: rc[1].cost.r_scale)
    M = hi_cfg.max_episodes
    m = np.arange(1, M + 1)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for agent in ("sparse_greedy", "dense_greedy", "sparse_excited"):
        col, ls, _, lab = STYLE[agent]
        med = np.median(per_episode_regret_trajectories(hi_res, agent), axis=0)
        ax.plot(m, med, color=col, ls=ls, lw=1.6, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("episode $m$")
    ax.set_ylabel("per-episode regret")
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "cost_b.pdf"))
    plt.close(fig)


# --------------------------------------------- single-system anchor (regret + A/B)
def fig_anchor(res, cfg, outdir, fname="anchor.pdf", suptitle=None):
    # Emits <base>_a.pdf (cumulative regret) and <base>_b.pdf (drift vs control recovery)
    # for LaTeX subfigures.
    M = cfg.max_episodes
    m = np.arange(1, M + 1)
    base = fname[:-4] if fname.endswith(".pdf") else fname

    # (a) cumulative regret vs episode, median + 5-95 percentile band. The anchor system is
    # marginally stable (no heavy-tailed destabilisation), so this near-full-range envelope
    # stays non-overlapping between the dense and sparse groups -- a stronger separation
    # claim than the IQR. (True min-max would overlap: a single outlier seed sets the extreme.)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for agent in LEARNING:
        cum = np.cumsum(_per_ep_regret(res, agent), axis=1)
        med = np.median(cum, axis=0)
        lo, hi = np.percentile(cum, 5, axis=0), np.percentile(cum, 95, axis=0)
        c, ls, _, lab = STYLE[agent]
        ax.plot(m, med, color=c, ls=ls, lw=1.6, label=lab)
        ax.fill_between(m, lo, hi, color=c, alpha=0.12)
    ax.set_xlabel("episode $m$")
    ax.set_ylabel(r"cumulative regret $R_m$")
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, f"{base}_a.pdf"))
    plt.close(fig)

    # (b) drift vs control recovery: dense (solid) vs sparse (dashed), A green / B vermillion
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for agent, ls in (("dense_greedy", "-"), ("sparse_greedy", "--")):
        who = STYLE[agent][3].split("-")[0]
        for key, col, blk in (
            ("error_A", OK["green"], "A"),
            ("error_B", OK["vermillion"], "B"),
        ):
            med = np.nanmedian(_traj(res, agent, key), axis=0)
            ax.plot(m, med, color=col, ls=ls, lw=1.6, label=f"{who} err$_{blk}$")
    ax.set_yscale("log")
    ax.set_xlabel("episode $m$")
    ax.set_ylabel("relative parameter error")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(outdir, f"{base}_b.pdf"))
    plt.close(fig)


# ------------------------------------------------ IEEE-39 topology recovery heatmaps
def fig_ieee39_topology(res, cfg, outdir, fname="ieee39_topology.pdf"):
    """Emits <base>_{a,b,c}.pdf for a LaTeX subfigure row -- log-magnitude heatmaps of
    Theta = [A|B]: (a) the truth, (b/c) the greedy agents' estimates right after the
    exploration episode (checkpoint m=0), when each 87-unknown row regression has seen
    only H=50 samples. The LASSO already shows the grid's wiring; underdetermined least
    squares is full-amplitude noise. Colour scale shared; colorbar on panel (c) only.
    Requires load_point(..., with_snapshots=True). Median seed by sparse final regret.
    No in-figure panel titles."""
    from matplotlib.colors import LogNorm

    base = fname[:-4] if fname.endswith(".pdf") else fname
    fins = [r.cumulative_regret("sparse_greedy")[-1] for r in res]
    r = res[int(np.argsort(fins)[len(fins) // 2])]

    def theta_at(agent, m):
        ep = r.episodes[agent][m]
        return np.hstack([ep.diagnostics["A_est"], ep.diagnostics["B_est"]])

    T_star = np.abs(np.hstack([r.A_star, r.B_star]))
    panels = [
        ("a", T_star, False),
        ("b", np.abs(theta_at("sparse_greedy", 0)), False),
        ("c", np.abs(theta_at("dense_greedy", 0)), True),
    ]
    vmin = 5e-3
    vmax = max(T_star.max(), panels[2][1].max())
    d = T_star.shape[0]

    for tag, mat, cbar in panels:
        fig, ax = plt.subplots(figsize=(4.9, 4.0) if cbar else (4.0, 4.0),
                               constrained_layout=True)
        masked = np.ma.masked_less(mat, vmin)
        im = ax.imshow(masked, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="Greys",
                       aspect="auto", interpolation="nearest")
        ax.axvline(d - 0.5, color=OK["vermillion"], lw=0.9)
        ax.set_xlabel(r"column (state $\mid$ input)")
        ax.tick_params(labelsize=8)
        if tag == "a":
            ax.set_ylabel("row")
        if cbar:
            fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, label=r"$|$entry$|$")
        _save(fig, os.path.join(outdir, f"{base}_{tag}.pdf"))
        plt.close(fig)


# --------------------------------------- single-point diagnostics (2x2, e.g. d=500)
def fig_singlepoint_diagnostics(res, cfg, outdir, fname="singlepoint.pdf"):
    # Emits <base>_{a,b,c,d}.pdf for a 2x2 LaTeX subfigure block: (a) cumulative regret,
    # (b) relative parameter error, (c) restricted Gram min eigenvalue, (d) support F1 --
    # the per-episode single-point view of one sweep point (median, IQR shaded).
    M = cfg.max_episodes
    m = np.arange(1, M + 1)
    base = fname[:-4] if fname.endswith(".pdf") else fname

    def panel(ylabel, yscale, tag, series, legend=False, ylim=None):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for agent in LEARNING:
            med, lo, hi = _med_iqr(series(agent))
            c, ls, _, lab = STYLE[agent]
            ax.plot(m, med, color=c, ls=ls, lw=1.6, label=lab)
            ax.fill_between(m, lo, hi, color=c, alpha=0.15)
        ax.set_yscale(yscale)
        ax.set_xlabel("episode $m$")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        if legend:
            ax.legend()
        fig.tight_layout()
        _save(fig, os.path.join(outdir, f"{base}_{tag}.pdf"))
        plt.close(fig)

    panel(r"cumulative regret $R_m$", "linear", "a",
          lambda ag: np.cumsum(_per_ep_regret(res, ag), axis=1), legend=True)
    panel(r"parameter error $\|\hat{\mathbf{\Theta}}_m-\mathbf{\Theta}_\star\|_F/\|\mathbf{\Theta}_\star\|_F$",
          "log", "b", lambda ag: _traj(res, ag, "error_joint"))
    panel(r"restricted Gram min.\ eigenvalue", "log", "c",
          lambda ag: _traj(res, ag, "gram_min_eig"))
    panel(r"support $F_1$", "linear", "d",
          lambda ag: _traj(res, ag, "support_f1_joint"))


# ------------------------------------------------ J. sparsity sweep (fixed d, vary s)
def fig_sparsity(sweep, outdir):
    # Emits sparsity_a.pdf (regret vs s) and sparsity_b.pdf (advantage vs s) for LaTeX subfigures.
    ss = sorted(sweep)

    # (a) dense-to-sparse regret ratio vs s -- scale-independent, matched per seed
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for (da, sa), c, lst, mk, lab in [
        (("dense_greedy", "sparse_greedy"), OK["blue"], "-", "o", "greedy"),
        (("dense_excited", "sparse_excited"), OK["orange"], "--", "s", "excited"),
    ]:
        med, lo, hi = [], [], []
        for s in ss:
            rr = _final_regret(sweep[s][0], da) / _final_regret(sweep[s][0], sa)
            med.append(np.median(rr))
            lo.append(np.percentile(rr, 25))
            hi.append(np.percentile(rr, 75))
        ax.plot(ss, med, color=c, ls=lst, marker=mk, ms=5, label=lab)
        ax.fill_between(ss, lo, hi, color=c, alpha=0.15)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("row sparsity $s$")
    ax.set_ylabel("dense-to-sparse regret ratio")
    ax.set_xticks(ss)
    ax.set_xticklabels(ss)
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "sparsity_a.pdf"))
    plt.close(fig)

    # (b) basin-entry speedup vs s (V2 final-error threshold), with the 1/(s log(d+p)) shape ref
    med, lo, hi = [], [], []
    for s in ss:
        r = _basin_speedup(sweep[s][0])
        med.append(np.median(r))
        lo.append(np.percentile(r, 25))
        hi.append(np.percentile(r, 75))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(ss, med, color=OK["vermillion"], marker="o", ms=5, label="empirical")
    ax.fill_between(ss, lo, hi, color=OK["vermillion"], alpha=0.15)
    cfg0 = sweep[ss[0]][1]
    dp = cfg0.system.d + cfg0.system.p
    raw = np.array([1.0 / (s * np.log(dp)) for s in ss])
    ref = 1.0 + (med[0] - 1.0) * raw / raw[0]  # excess decays as 1/(s log(d+p))
    ax.plot(
        ss, ref, color="grey", ls="--", lw=1, label=r"theory $\propto 1/(s\log(d+p))$"
    )
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("row sparsity $s$")
    ax.set_ylabel(r"basin-entry speedup $m_0^{\mathrm{dense}}/m_0^{\mathrm{sparse}}$")
    ax.set_xticks(ss)
    ax.set_xticklabels(ss)
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "sparsity_b.pdf"))
    plt.close(fig)


# --------------------------------------------- regret decomposition (theory schematic)
def fig_regret_decomposition_theory(outdir):
    """Single-panel theoretical schematic of the transient O(sqrt(m)) -> logarithmic
    O(log m) regret decomposition, with a basin-entry point m0. No in-figure title."""
    M = 100
    m0 = 25  # basin entry point
    m_transient = np.linspace(0, m0, 100)
    m_log = np.linspace(m0, M, 300)

    c1 = 5
    regret_transient = c1 * np.sqrt(m_transient)
    regret_m0 = c1 * np.sqrt(m0)
    c2 = 12
    regret_log = regret_m0 + c2 * np.log(m_log / m0)

    c_trans, c_log, c_basin = OK["orange"], OK["green"], OK["purple"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(m_transient, regret_transient, color=c_trans, lw=2, label="transient phase")
    ax.plot(m_log, regret_log, color=c_log, lw=2, label="logarithmic phase")

    ax.axvline(x=m0, color=c_basin, ls="--", lw=1.2)
    ax.scatter([m0], [regret_m0], color=c_basin, s=45, zorder=5)
    ax.annotate(
        "basin entry", xy=(m0, regret_m0), xytext=(m0 + 2, regret_m0 - 5), color=c_basin
    )

    ax.axvspan(0, m0, color=c_trans, alpha=0.07)
    ax.axvspan(m0, M, color=c_log, alpha=0.07)
    ax.text(
        m0 / 2,
        regret_m0 * 1.0,
        r"${O}(\sqrt{m_0 \Lambda})$",
        fontsize=13,
        ha="center",
        va="center",
        color=c_trans,
    )
    ax.text(
        (M + m0) / 2,
        regret_m0 + c2 * np.log(M / m0),
        r"${O}(\Lambda \log(M / m_0))$",
        fontsize=13,
        ha="center",
        va="center",
        color=c_log,
    )

    ax.set_xlim(0, M)
    ax.set_ylim(0, np.max(regret_log) + 5)
    ax.set_xlabel("episode $m$")
    ax.set_ylabel("cumulative regret $R_M$")
    ax.set_xticks([0, m0, M])
    ax.set_xticklabels(["$0$", "$m_0$", "$M$"])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "regret_decomposition.pdf"))
    plt.close(fig)


# --------------------------------------------- spring-mass chain illustration
def fig_spring_illustration(outdir):
    """Two single-panel PDFs: (a) the spring-mass chain schematic (spring_chain_a.pdf),
    (b) the [A|B] sparsity pattern (spring_chain_b.pdf). No in-figure titles."""
    from matplotlib.patches import Rectangle, FancyArrowPatch
    from system_generator import sample_spring_chain

    N = 6  # masses for the illustration
    d, p = 2 * N, N // 2
    A, B, supports, _ = sample_spring_chain(d=d, p=p, seed=0)
    act = ((np.arange(p) * N) // p).tolist()

    def spring(ax, x0, x1, y, coils=5, amp=0.13):
        """Draw a zig-zag spring between (x0, y) and (x1, y)."""
        lead = 0.18
        xs = np.linspace(x0 + lead, x1 - lead, 2 * coils + 1)
        ys = y + amp * np.array([0] + [(-1) ** k for k in range(2 * coils - 1)] + [0])
        ax.plot([x0, x0 + lead], [y, y], "k", lw=1.1)
        ax.plot(xs, ys, "k", lw=1.1)
        ax.plot([x1 - lead, x1], [y, y], "k", lw=1.1)

    # (a) physical chain
    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    y = 0.0
    for w in (0.0, N + 1.0):  # walls
        ax1.add_patch(
            Rectangle((w - 0.12, -0.6), 0.12, 1.2, hatch="////", fill=False, lw=1.2)
        )
    xs_mass = np.arange(1, N + 1)
    spring(ax1, 0.0, 1.0, y)  # left wall spring
    for i in range(N - 1):
        spring(ax1, xs_mass[i], xs_mass[i + 1], y)  # inter-mass springs
    spring(ax1, float(N), N + 1.0, y)  # right wall spring
    for i, x in enumerate(xs_mass):
        actuated = i in act
        ax1.add_patch(
            Rectangle(
                (x - 0.26, -0.26),
                0.52,
                0.52,
                facecolor=OK["vermillion"] if actuated else "#cccccc",
                edgecolor="k",
                lw=1.2,
                zorder=3,
            )
        )
        ax1.text(
            x,
            0,
            f"$m_{{{i + 1}}}$",
            ha="center",
            va="center",
            zorder=4,
            color="white" if actuated else "black",
            fontsize=11,
        )
        if actuated:
            # control force acts along the chain axis (masses only translate sideways)
            ax1.add_patch(
                FancyArrowPatch(
                    (x - 0.38, 0.42),
                    (x + 0.38, 0.42),
                    arrowstyle="<->",
                    mutation_scale=13,
                    color=OK["vermillion"],
                    lw=1.6,
                    zorder=4,
                )
            )
            ax1.text(
                x,
                0.53,
                f"$u_{{{act.index(i) + 1}}}$",
                color=OK["vermillion"],
                fontsize=11,
                ha="center",
            )
    ax1.set_xlim(-0.5, N + 1.5)
    ax1.set_ylim(-0.9, 1.25)
    ax1.axis("off")
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "spring_chain_a.pdf"))
    plt.close(fig)

    # (b) sparsity pattern of [A | B]
    d, p = A.shape[0], B.shape[1]
    mask = np.zeros((d, d + p))
    mask[:, :d] = A != 0
    mask[:, d:] = B != 0
    fig, ax2 = plt.subplots(figsize=(4.5, 4.2))
    ax2.imshow(
        mask, cmap="Greys", vmin=0, vmax=1, aspect="equal", interpolation="nearest"
    )
    sep = 0.06 * d  # grey dashed A|B divider, extended a little beyond the matrix
    ax2.plot(
        [d - 0.5, d - 0.5],
        [-0.5 - sep, d - 0.5 + sep],
        color="grey",
        ls="--",
        lw=1.2,
        clip_on=False,
        zorder=3,
    )
    ax2.set_xticks([d / 2 - 0.5, d + p / 2 - 0.5])
    ax2.set_xticklabels([r"$\mathbf{A}_\star$", r"$\mathbf{B}_\star$"])
    ax2.set_yticks([])
    ax2.grid(False)
    for s in ax2.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "spring_chain_b.pdf"))
    plt.close(fig)


# --------------------------------------------- IEEE 39-bus illustration
def fig_ieee39_illustration(outdir):
    """Two single-panel PDFs: (a) the IEEE 39-bus network graph (ieee39_a.pdf),
    (b) the [A|B] sparsity pattern (ieee39_b.pdf). No in-figure titles."""
    import matplotlib.lines as mlines
    from system_generator import sample_ieee39
    from ieee39_data import IEEE39_BRANCHES, IEEE39_ACTUATED_BUSES

    POS = {
        1: [0.351, -0.211],
        2: [0.303, -0.225],
        3: [0.217, -0.086],
        4: [0.199, 0.018],
        5: [0.374, 0.058],
        6: [0.457, 0.543],
        7: [0.666, 0.389],
        8: [0.517, -0.088],
        9: [0.504, -0.143],
        10: [0.061, 0.901],
        11: [0.189, 0.451],
        12: [0.162, 0.413],
        13: [0.134, 0.377],
        14: [0.06, 0.152],
        15: [-0.044, 0.137],
        16: [-0.278, -0.006],
        17: [-0.152, -0.25],
        18: [0.071, -0.119],
        19: [-0.328, 0.093],
        20: [-0.4, 0.25],
        21: [-0.427, -0.068],
        22: [-0.593, -0.064],
        23: [-0.681, -0.253],
        24: [-0.625, -0.264],
        25: [0.116, -0.432],
        26: [0.051, -0.452],
        27: [-0.092, -0.364],
        28: [0.015, -0.434],
        29: [0.044, -0.484],
        30: [0.375, -0.329],
        31: [0.493, 0.616],
        32: [0.059, 1.0],
        33: [-0.463, 0.129],
        34: [-0.484, 0.337],
        35: [-0.735, 0.01],
        36: [-0.749, -0.283],
        37: [0.18, -0.503],
        38: [0.016, -0.619],
        39: [0.436, -0.194],
    }

    GEN = set(IEEE39_ACTUATED_BUSES)
    REF = 39
    VERM, BLUE, GREY = OK["vermillion"], OK["blue"], "#bdbdbd"

    A, B, supports, _ = sample_ieee39(seed=0)

    # (a) grid graph
    # 6 x 4.38 so that, at the LaTeX widths (a: 0.55, b: 0.43 linewidth), the rendered
    # height matches the sparsity panel's 4.5 x 4.2: 0.55*(4.38/6) = 0.43*(4.2/4.5).
    fig, ax1 = plt.subplots(figsize=(6, 4.38))
    for f, t, _ in IEEE39_BRANCHES:
        (x0, y0), (x1, y1) = POS[f], POS[t]
        ax1.plot([x0, x1], [y0, y1], color="#999999", lw=0.8, zorder=1)
    for bus, (x, y) in POS.items():
        if bus == REF:
            c, s, ec = BLUE, 230, "k"
        elif bus in GEN:
            c, s, ec = VERM, 180, "k"
        else:
            c, s, ec = GREY, 70, "none"
        ax1.scatter([x], [y], s=s, c=c, edgecolors=ec, linewidths=0.6, zorder=3)
        if bus in GEN or bus == REF:
            ax1.annotate(
                str(bus),
                (x, y),
                fontsize=7,
                ha="center",
                va="center",
                color="white",
                zorder=4,
                fontweight="bold",
            )
    handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            ls="none",
            mfc=VERM,
            mec="k",
            ms=10,
            label="generator (actuated, $p=9$)",
        ),
        mlines.Line2D(
            [],
            [],
            marker="o",
            ls="none",
            mfc=BLUE,
            mec="k",
            ms=11,
            label="bus 39 (reference, unactuated)",
        ),
        mlines.Line2D(
            [], [], marker="o", ls="none", mfc=GREY, mec="none", ms=7, label="load bus"
        ),
    ]
    ax1.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        fontsize=8,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
    )
    ax1.axis("off")
    # aspect left "auto": the hand-placed layout is schematic, not geographic, so it may
    # stretch to fill the flatter canvas that height-matches the sparsity panel.
    ax1.grid(False)
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "ieee39_a.pdf"))
    plt.close(fig)

    # (b) sparsity pattern of [A | B]
    d, p = A.shape[0], B.shape[1]
    mask = np.zeros((d, d + p))
    mask[:, :d] = A != 0
    mask[:, d:] = B != 0
    fig, ax2 = plt.subplots(figsize=(4.5, 4.2))
    ax2.imshow(
        mask, cmap="Greys", vmin=0, vmax=1, aspect="equal", interpolation="nearest"
    )
    sep = 0.06 * d  # grey dashed A|B divider, extended a little beyond the matrix
    ax2.plot(
        [d - 0.5, d - 0.5],
        [-0.5 - sep, d - 0.5 + sep],
        color="grey",
        ls="--",
        lw=1.2,
        clip_on=False,
        zorder=3,
    )
    ax2.set_xticks([d / 2 - 0.5, d + p / 2 - 0.5])
    ax2.set_xticklabels([r"$\mathbf{A}_\star$", r"$\mathbf{B}_\star$"])
    ax2.set_yticks([])
    ax2.grid(False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "ieee39_b.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal-d", type=int, default=20)
    ap.add_argument("--synthetic-dir", default="results/synthetic")
    ap.add_argument("--spring-dir", default="results/springs")
    ap.add_argument("--ieee39-dir", default="results/ieee39")
    ap.add_argument("--sparsity-dir", default="results/sparsity")
    ap.add_argument("--clambda-dir", default="results/clambda")
    ap.add_argument("--excitation-dir", default="results/excitation")
    ap.add_argument("--cost-dir", default="results/cost")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def by_int(study_dir):
        """Load a study dir into {int(point): (results, cfg)} (e.g. d10->10, s2->2)."""
        if not os.path.isdir(study_dir):
            return {}
        return {
            int("".join(c for c in p if c.isdigit())): rc
            for p, rc in load_study(study_dir).items()
        }

    # Notebook-ported standalone illustrations (no results needed).
    fig_regret_decomposition_theory(args.out)
    print("  [theory] regret decomposition schematic")
    fig_spring_illustration(args.out)
    print("  [illu] spring-mass chain")
    fig_ieee39_illustration(args.out)
    print("  [illu] IEEE 39-bus")

    sweep = by_int(args.synthetic_dir)
    focal = sweep.get(args.focal_d)

    print(f"sweep d={sorted(sweep)}, focal d={args.focal_d}")
    fig_scaling_regret(sweep, args.out)
    print("  [A] scaling regret")
    if focal:
        fig_regret_decomposition(*focal, args.out)
        print("  [B] regret decomposition")
    fig_speedup_vs_d(sweep, args.out)
    print("  [C] speedup vs d")
    if focal:
        fig_ab_asymmetry(*focal, args.out)
        print("  [D] A/B asymmetry")
    fig_btqb_cond(sweep, args.out)
    print("  [F] cond(B^T Q B) vs d")
    if 500 in sweep:
        fig_singlepoint_diagnostics(*sweep[500], args.out, fname="singlepoint_d500.pdf")
        print("  [E] single-point diagnostics d=500")
    if os.path.isdir(args.clambda_dir):
        fig_hyperparam(args.clambda_dir, args.out)
        print("  [G] c_lambda sweep")
    else:
        print("  [G] c_lambda sweep skipped (no results/clambda/)")
    if os.path.isdir(args.excitation_dir):
        fig_excitation(args.excitation_dir, args.out)
        print("  [K] excitation sweep")
    else:
        print("  [K] excitation sweep skipped (no results/excitation/)")
    if os.path.isdir(args.cost_dir):
        fig_cost(args.cost_dir, args.out)
        print("  [cost] cost-weighting sweep")
    else:
        print("  [cost] cost-weighting sweep skipped (no results/cost/)")

    spring = by_int(args.spring_dir)
    if spring:
        fig_scaling_regret(
            spring,
            args.out,
            fname="spring_scaling.pdf",
            suptitle="Spring-mass chain (fixed $s=3$, $p=d/4$)",
        )
        print(f"  [H] spring-chain scaling (d={sorted(spring)})")
    else:
        print("  [H] spring-chain scaling skipped (no results/spring/)")

    if os.path.isdir(args.ieee39_dir) and any(
        f.startswith("seed_") and not f.endswith("_snapshots.npz") for f in os.listdir(args.ieee39_dir)
    ):
        fig_anchor(
            *load_point(args.ieee39_dir),
            args.out,
            fname="ieee39_anchor.pdf",
            suptitle="IEEE 39-bus power grid ($d=78$, $p=9$)",
        )
        print("  [I] IEEE 39-bus anchor")
    else:
        print("  [I] IEEE 39-bus anchor skipped (no results/ieee39/)")

    sparsity = by_int(args.sparsity_dir)
    if sparsity:
        fig_sparsity(sparsity, args.out)
        print(f"  [J] sparsity sweep (s={sorted(sparsity)})")
    else:
        print("  [J] sparsity sweep skipped (no results/sparsity/)")

    # PNGs are written next to each PDF by _save() above — no poppler needed.
    import glob

    n_png = len(glob.glob(os.path.join(args.out, "*.png")))
    print(f"  ({n_png} PNGs written alongside the PDFs)")
    print(f"-> {args.out}/")


if __name__ == "__main__":
    main()
