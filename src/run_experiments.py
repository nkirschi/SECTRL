"""
Top-level script for running all benchmarks from the paper.

Usage:
    python run_experiments.py                    # run all benchmarks
    python run_experiments.py --benchmark main   # run one benchmark
    python run_experiments.py --quick            # fast smoke test
"""

import argparse
import os
import json
import time
import numpy as np

from common import ExperimentConfig
from runner import run_benchmark
from analysis import (
    all_pairwise_tests,
    final_summary_table,
    print_summary,
    plot_trajectories,
    plot_basin_entry_comparison,
    plot_sparsity_evolution,
    basin_entry_ratio,
    seed_wins,
    quarter_horizon_regret,
    episode_average_regret,
    mean_and_ci,
)


# ─────────────────────────────────────────────────────────────────────
# Benchmark configurations
# ─────────────────────────────────────────────────────────────────────


BENCHMARKS = {
    "d10": ExperimentConfig(
        x_dim=10,
        u_dim=3,
        sparsity=3,
        T=1.0,
        dt=0.025,
        max_episodes=120,
        n_seeds=50,
        sigma=0.5,
        lambda_lasso=0.015,
        sigma_u=0.10,
    ),
    "d20": ExperimentConfig(
        x_dim=20,
        u_dim=5,
        sparsity=3,
        T=1.0,
        dt=0.025,
        max_episodes=100,  # 120,
        n_seeds=10,  # 50,
        sigma=0.1,
        # lambda_lasso=0.03,
        c_lambda=0.02,
        sigma_u=0.5 * 0.1,
    ),
    "d50": ExperimentConfig(
        x_dim=50,
        u_dim=13,
        sparsity=3,
        T=1.0,
        dt=0.025,
        max_episodes=150,
        n_seeds=50,
        sigma=0.5,
        lambda_lasso=0.05,
        sigma_u=0.15,
    ),
    "d100": ExperimentConfig(
        x_dim=100,
        u_dim=25,
        sparsity=3,
        T=1.0,
        dt=0.025,
        max_episodes=200,
        n_seeds=50,
        sigma=0.5,
        lambda_lasso=0.08,
        sigma_u=0.15,
    ),
    # Quick smoke test
    "quick": ExperimentConfig(
        x_dim=10,
        u_dim=3,
        sparsity=3,
        T=1.0,
        dt=0.025,
        max_episodes=10,
        n_seeds=5,
        sigma=0.1,
        lambda_lasso=None,
        c_lambda=0.01,
        sigma_u=0.5 * 0.1,
    ),
}


# Sparsity sweep: fixed d=50, varying s
SPARSITY_SWEEP = {
    f"sparsity_s{s}": ExperimentConfig(
        x_dim=50,
        u_dim=13,
        sparsity=s,
        T=1.0,
        dt=0.025,
        max_episodes=150,
        n_seeds=50,
        sigma=0.5,
        lambda_lasso=None,
        sigma_u=0.15,
    )
    for s in [2, 3, 5, 10, 20]
}


# Actuator sweep: fixed d=20, s=3, varying p
ACTUATOR_SWEEP = {
    f"actuator_p{p}": ExperimentConfig(
        x_dim=20,
        u_dim=p,
        sparsity=3,
        T=1.0,
        dt=0.025,
        max_episodes=120,
        n_seeds=50,
        sigma=0.5,
        lambda_lasso=0.03,
        sigma_u=0.15,
    )
    for p in [2, 5, 10, 20]
}


# Excitation sweep: fixed d=10, varying sigma_u
EXCITATION_SWEEP = {
    f"excitation_{su}": ExperimentConfig(
        x_dim=10,
        u_dim=3,
        sparsity=3,
        T=1.0,
        dt=0.025,
        max_episodes=120,
        n_seeds=50,
        sigma=0.5,
        lambda_lasso=0.015,
        sigma_u=su,
    )
    for su in [0.0, 0.05, 0.10, 0.15, 0.20, 0.35]
}


# Horizon sweep: fixed d=10, varying T
HORIZON_SWEEP = {
    f"horizon_T{T}": ExperimentConfig(
        x_dim=10,
        u_dim=3,
        sparsity=3,
        T=T,
        dt=0.025,
        max_episodes=120,
        n_seeds=50,
        sigma=0.5,
        lambda_lasso=0.015,
        sigma_u=0.10,
    )
    for T in [0.5, 1.0, 2.0, 5.0]
}


# ─────────────────────────────────────────────────────────────────────
# Running and reporting
# ─────────────────────────────────────────────────────────────────────


def run_and_report(name, exp_config, output_dir):
    """Run one benchmark and save all outputs."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {name}")
    print(f"  d={exp_config.x_dim}, p={exp_config.u_dim}, s={exp_config.sparsity}")
    print(
        f"  M={exp_config.max_episodes}, H={exp_config.H}, seeds={exp_config.n_seeds}"
    )
    print(f"  Theoretical speedup: {exp_config.theoretical_speedup:.2f}")
    print(
        f"  Theoretical lambda schedule: "
        + ", ".join(f"{l:.3f}" for l in exp_config.theoretical_lambda_schedule)
    )
    print(f"{'=' * 60}")

    t0 = time.time()
    results = run_benchmark(exp_config, verbose=True)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    # Summary table
    table = final_summary_table(results)
    print("\nFinal-episode summary:")
    print_summary(table)

    # Statistical tests
    tests = all_pairwise_tests(results)
    print("\nPaired tests (dense vs sparse):")
    for label, test_dict in tests.items():
        t_test = test_dict["t_test"]
        wilcox = test_dict["wilcoxon"]
        sign = test_dict["sign_test"]
        print(f"  {label}:")
        print(f"    t-test: t={t_test['t_stat']:.3f}, p={t_test['p_value']:.4f}")
        print(f"    Wilcoxon: p={wilcox['p_value']:.4f}")
        print(f"    Sign: wins={sign['wins_b']}/{sign['n']}, p={sign['p_value']:.4f}")

    # Seed wins
    for sp in ["sparse_greedy", "sparse_excitation"]:
        w = seed_wins(results, "dense_greedy", sp)
        print(f"  Seed wins ({sp} < dense): {w}/{len(results)}")

    # Non-endpoint robustness
    # print("\nNon-endpoint robustness:")
    # for sp in ["sparse_greedy", "sparse_excitation"]:
    #     q1 = quarter_horizon_regret(results, sp)
    #     q1_d = quarter_horizon_regret(results, "dense_greedy")
    #     ea = episode_average_regret(results, sp)
    #     ea_d = episode_average_regret(results, "dense_greedy")
    #     q1_red = 1 - np.mean(q1) / max(np.mean(q1_d), 1e-15)
    #     ea_red = 1 - np.mean(ea) / max(np.mean(ea_d), 1e-15)
    #     print(f"  {sp}: Q1 reduction={q1_red:.1%}, episode-avg reduction={ea_red:.1%}")

    # # Basin entry
    ratios, median = basin_entry_ratio(results, threshold=0.15)
    # print(
    #     f"\nBasin entry ratio (median): {median:.2f} "
    #     f"(theory: {exp_config.theoretical_speedup:.2f})"
    # )

    # Save plots
    bench_dir = os.path.join(output_dir, name, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(bench_dir, exist_ok=True)

    plot_trajectories(
        results, exp_config, save_path=os.path.join(bench_dir, "trajectories.png")
    )
    plot_basin_entry_comparison(
        results,
        exp_config,
        save_path=os.path.join(bench_dir, "basin_entry.png"),
    )
    plot_sparsity_evolution(results, exp_config, output_dir=bench_dir)

    # Save numerical results
    save_dict = {
        "config": {
            "x_dim": exp_config.x_dim,
            "u_dim": exp_config.u_dim,
            "sparsity": exp_config.sparsity,
            "max_episodes": exp_config.max_episodes,
            "n_seeds": exp_config.n_seeds,
            "theoretical_speedup": exp_config.theoretical_speedup,
        },
        "summary": {
            name: {k: list(v) for k, v in row.items()} for name, row in table.items()
        },
        "tests": {
            label: {
                test_name: {
                    k: float(v)
                    if isinstance(v, (int, float, np.floating, np.integer))
                    else v
                    for k, v in test_result.items()
                }
                for test_name, test_result in test_dict.items()
            }
            for label, test_dict in tests.items()
        },
        "basin_entry_median": float(median) if np.isfinite(median) else None,
        "elapsed_seconds": elapsed,
    }

    with open(os.path.join(bench_dir, "results.json"), "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"Results saved to {bench_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run sparse continuous-time LQ control experiments."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Run a single benchmark by name (e.g. 'main', 'quick').",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run only the quick smoke test."
    )
    parser.add_argument(
        "--sweeps",
        action="store_true",
        help="Also run sparsity/actuator/excitation/horizon sweeps.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results and plots.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.quick:
        benchmarks = {"quick": BENCHMARKS["quick"]}
    elif args.benchmark:
        if args.benchmark in BENCHMARKS:
            benchmarks = {args.benchmark: BENCHMARKS[args.benchmark]}
        else:
            available = list(BENCHMARKS.keys())
            print(f"Unknown benchmark '{args.benchmark}'. Available: {available}")
            return
    else:
        benchmarks = {k: v for k, v in BENCHMARKS.items() if k != "quick"}

    # Run main benchmarks
    for name, config in benchmarks.items():
        run_and_report(name, config, args.output_dir)

    # Run sweeps if requested
    if args.sweeps and not args.quick:
        all_sweeps = {
            **SPARSITY_SWEEP,
            **ACTUATOR_SWEEP,
            **EXCITATION_SWEEP,
            **HORIZON_SWEEP,
        }
        for name, config in all_sweeps.items():
            run_and_report(name, config, args.output_dir)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
