"""
Top-level script for running benchmarks defined in configs/.

Benchmark configs: configs/benchmarks/<name>.yaml
Sweep configs:     configs/sweeps/<name>.yaml

Usage:
    python main.py                        # all benchmarks
    python main.py --benchmark d20        # single benchmark
    python main.py --sweep excitation     # single sweep
    python main.py --debug                # quick debugging run
"""

import os
import sys

# Avoid multithreading in numpy for cleaner parallelism
if any(a.startswith("--n-workers") for a in sys.argv[1:]):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import dataclasses
from dataclasses import is_dataclass
import json
import time
import glob
from collections import defaultdict
import io
from contextlib import redirect_stdout
import yaml
from common import ExperimentConfig
from runner import (
    run_paired_experiment,
    persist_raw_results,
    load_raw_results,
    find_result_dirs,
)
from analysis import (
    all_pairwise_tests,
    final_summary_table,
    print_summary,
    plot_trajectories,
    plot_basin_entry_comparison,
    plot_parameter_evolution,
    plot_self_exploration_diagnostics,
    basin_entry_ratio,
    seed_wins,
)


CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
)

# Config loading


def load_all_benchmarks() -> dict:
    benchmarks = {}
    pattern = os.path.join(CONFIG_DIR, "benchmarks", "*.yaml")
    for path in sorted(glob.glob(pattern)):
        name = os.path.splitext(os.path.basename(path))[0]
        benchmarks[name] = ExperimentConfig.from_yaml(path)
    return benchmarks


def load_benchmark(name: str) -> ExperimentConfig:
    path = os.path.join(CONFIG_DIR, "benchmarks", f"{name}.yaml")
    return ExperimentConfig.from_yaml(path)


def load_sweep(name: str) -> dict:
    """
    Load a sweep config (Option B: base config + list of override dicts).

    Sweep YAML schema:
        base:         <benchmark name>       # base config to load
        vary:
          - {field1: val1, field2: val2}     # each dict = one override set
          - ...
    """
    path = os.path.join(CONFIG_DIR, "sweeps", f"{name}.yaml")
    with open(path) as f:
        spec = yaml.safe_load(f)

    base = load_benchmark(spec["base"])
    prefix = spec["base"]
    configs = {}

    for override in spec["vary"]:
        suffix = "_".join(f"{k}{v}" for k, v in override.items())
        key = f"{prefix}_{suffix}"
        configs[key] = ExperimentConfig.apply_overrides(base, override)

    return configs


# Execution


def _print_config(name: str, exp_config: ExperimentConfig) -> None:
    """Print benchmark configuration before execution starts."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {name}")
    print(
        f"  d={exp_config.system.d}, p={exp_config.system.p}, "
        f"s={exp_config.system.sparsity}="
        f"{exp_config.system.s_A}+{exp_config.system.s_B}=s_A+s_B"
    )
    print(
        f"  M={exp_config.max_episodes}, H={exp_config.system.H}, "
        f"seeds={exp_config.n_seeds}, m_explore={exp_config.m_explore}"
    )
    print(f"  agents: {list(exp_config.agents)}")
    print(f"  Theoretical speedup: {exp_config.theoretical_speedup:.2f}")
    print(f"{'=' * 60}")


def _run_sequential(exp_config: ExperimentConfig, verbose: bool = False) -> list:
    """Run all seeds for one config sequentially."""
    seeds = list(range(exp_config.n_seeds))
    results = []
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"Seed {i + 1}/{len(seeds)} (seed={seed})")
        results.append(run_paired_experiment(exp_config, seed, verbose))
    return results


def _run_parallel(
    tasks: list[tuple[str, ExperimentConfig, int]],
    n_workers: int,
) -> dict[str, list]:
    """
    Run a list of (name, config, seed) tasks across a worker pool.

    Returns {name: [SeedResult, ...]} with each list sorted by seed.
    Individual failures are caught and reported; a summary error is raised
    after all futures settle so one bad seed does not discard others.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    grouped = defaultdict(list)
    errors = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(run_paired_experiment, cfg, seed): (name, seed)
            for name, cfg, seed in tasks
        }
        for future in as_completed(futures):
            name, seed = futures[future]
            try:
                grouped[name].append(future.result())
            except Exception as exc:
                errors.append((name, seed, exc))
                print(f"Error: seed {seed} of '{name}' failed: {exc}")

    if errors:
        summary = ", ".join(f"{name}[{seed}]" for name, seed, _ in errors)
        raise RuntimeError(f"{len(errors)} task(s) failed: {summary}")

    return {
        name: sorted(results, key=lambda r: r.seed) for name, results in grouped.items()
    }


# Reporting


def _build_summary_text(
    results: list,
    exp_config: ExperimentConfig,
    elapsed: float | None = None,
) -> str:
    """Construct the human-readable summary string."""
    import numpy as np

    table = final_summary_table(results, agent_names=exp_config.agents)
    tests = all_pairwise_tests(results)
    min_eigs = [float(r.btqb_min_eig) for r in results]
    max_eigs = [float(r.btqb_max_eig) for r in results]

    buf = io.StringIO()
    with redirect_stdout(buf):
        if elapsed is not None:
            print(f"Completed in {elapsed:.1f}s")
        print("\nFinal-episode summary:")
        print_summary(table)
        print("\nPaired tests:")
        for label, test_dict in tests.items():
            sign = test_dict["sign_test"]
            print(
                f"  {label}: wins={sign['wins_b']}/{sign['n']}, "
                f"p={sign['p_value']:.4f} (sign test)"
            )
        learning = [n for n in exp_config.agents if n != "oracle"]
        if "dense_greedy" in exp_config.agents:
            for sp in [n for n in learning if n != "dense_greedy"]:
                w = seed_wins(results, "dense_greedy", sp)
                print(f"  Seed wins ({sp} < dense_greedy): {w}/{len(results)}")
        print(f"\nSelf-exploration (B*\u1d40 Q B*):")
        print(
            f"  \u03bb_min : "
            f"min={min(min_eigs):.4f}  "
            f"median={float(np.median(min_eigs)):.4f}  "
            f"max={max(min_eigs):.4f}"
        )
        print(
            f"  \u03bb_max : "
            f"min={min(max_eigs):.4f}  "
            f"median={float(np.median(max_eigs)):.4f}  "
            f"max={max(max_eigs):.4f}"
        )
    return buf.getvalue()


def _build_save_dict(
    results: list,
    exp_config: ExperimentConfig,
    elapsed: float | None = None,
) -> dict:
    """Construct the JSON-serialisable summary dictionary."""
    import numpy as np

    table = final_summary_table(results, agent_names=exp_config.agents)
    tests = all_pairwise_tests(results)
    _, median = basin_entry_ratio(results, threshold=exp_config.support_threshold)
    min_eigs = [float(r.btqb_min_eig) for r in results]
    max_eigs = [float(r.btqb_max_eig) for r in results]

    # Flatten ExperimentConfig (top-level and nested sub-dataclasses) to dicts.
    config_dict = {
        f.name: getattr(exp_config, f.name) for f in dataclasses.fields(exp_config)
    }
    for k, v in config_dict.items():
        if isinstance(v, tuple):
            config_dict[k] = list(v)
        if is_dataclass(v):
            config_dict[k] = {f.name: getattr(v, f.name) for f in dataclasses.fields(v)}

    return {
        "config": config_dict,
        "summary": {
            n: {k: list(v) for k, v in row.items()} for n, row in table.items()
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
        "system_diagnostics": {
            "btqb_min_eig": {
                "mean":     float(np.mean(min_eigs)),
                "median":   float(np.median(min_eigs)),
                "min":      float(min(min_eigs)),
                "max":      float(max(min_eigs)),
                "per_seed": [float(x) for x in min_eigs],
            },
            "btqb_max_eig": {
                "mean":     float(np.mean(max_eigs)),
                "median":   float(np.median(max_eigs)),
                "min":      float(min(max_eigs)),
                "max":      float(max(max_eigs)),
                "per_seed": [float(x) for x in max_eigs],
            },
        },
        "basin_entry_median": float(median) if np.isfinite(median) else None,
        "elapsed_seconds": elapsed,
    }


def _save_summary(
    results: list,
    exp_config: ExperimentConfig,
    output_dir: str,
    elapsed: float | None = None,
    verbose: bool = True,
) -> None:
    """Write summary.txt (human-readable) and results.json (machine-readable)."""
    summary_text = _build_summary_text(results, exp_config, elapsed)
    if verbose:
        print(summary_text, end="")
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary_text)
    save_dict = _build_save_dict(results, exp_config, elapsed)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(save_dict, f, indent=2)


def _generate_plots(
    results: list,
    exp_config: ExperimentConfig,
    output_dir: str,
) -> None:
    """Render and save all PNGs into output_dir."""
    plot_trajectories(
        results, exp_config, save_path=os.path.join(output_dir, "trajectories.png")
    )
    plot_basin_entry_comparison(
        results, exp_config, save_path=os.path.join(output_dir, "basin_entry.png")
    )
    plot_self_exploration_diagnostics(
        results,
        exp_config,
        save_path=os.path.join(output_dir, "self_exploration.png"),
    )
    param_dir = os.path.join(output_dir, "params_evolution")
    os.makedirs(param_dir, exist_ok=True)
    plot_parameter_evolution(results, exp_config, output_dir=param_dir)


def report(
    name: str,
    exp_config: ExperimentConfig,
    results: list,
    output_dir: str,
    elapsed: float | None = None,
    verbose: bool = True,
) -> None:
    """
    Persist raw results, write summary text/JSON, and render plots.

    After this returns, the result directory contains everything needed
    to regenerate every output via `replot()`.
    """
    bench_dir = os.path.join(output_dir, name, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(bench_dir, exist_ok=True)
    persist_raw_results(results, exp_config, bench_dir)
    _save_summary(results, exp_config, bench_dir, elapsed=elapsed, verbose=verbose)
    _generate_plots(results, exp_config, bench_dir)
    print(f"Results saved to {bench_dir}/")


def replot(path: str) -> None:
    """
    Regenerate plots and summary files from previously persisted results.

    `path` may be a single result directory (containing seed_results.pkl)
    or a parent directory (e.g. results/sweeps/) whose subtree is walked
    for all such directories. summary.txt, results.json, and the PNGs are
    overwritten in place; seed_results.pkl is left untouched.
    """
    result_dirs = find_result_dirs(path)
    if not result_dirs:
        raise FileNotFoundError(
            f"No directories containing seed_results.pkl found under {path}"
        )
    for d in result_dirs:
        print(f"\nReplotting {d}")
        results, exp_config = load_raw_results(d)
        _save_summary(results, exp_config, d, elapsed=None, verbose=True)
        _generate_plots(results, exp_config, d)
    print(f"\nReplotted {len(result_dirs)} result director{'y' if len(result_dirs) == 1 else 'ies'}.")


# Entry point


def main():
    parser = argparse.ArgumentParser(
        description="Run sparse continuous-time LQ control experiments."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Run a single benchmark by YAML name (e.g. 'd20').",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Run a single sweep by YAML name (e.g. 'excitation').",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run only the debugging run."
    )
    parser.add_argument(
        "--replot",
        type=str,
        default=None,
        help=(
            "Path to a result directory or a parent containing several. "
            "Regenerates plots and summary in place from seed_results.pkl; "
            "does not re-run any simulation."
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes. "
            "For benchmarks: parallelises across seeds. "
            "For sweeps: parallelises across all (config, seed) pairs. "
        ),
    )
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    bench_dir = os.path.join(args.output_dir, "benchmarks")
    sweep_dir = os.path.join(args.output_dir, "sweeps")
    for d in [bench_dir, sweep_dir]:
        os.makedirs(d, exist_ok=True)

    if args.replot:
        replot(args.replot)

    elif args.debug:
        cfg = load_benchmark("debug")
        _print_config("debug", cfg)
        t0 = time.time()
        results = _run_sequential(cfg, verbose=True)
        report("debug", cfg, results, bench_dir, elapsed=time.time() - t0)

    elif args.benchmark:
        cfg = load_benchmark(args.benchmark)
        _print_config(args.benchmark, cfg)
        tasks = [(args.benchmark, cfg, seed) for seed in range(cfg.n_seeds)]
        t0 = time.time()
        if args.n_workers <= 1:
            results = _run_sequential(cfg, verbose=True)
        else:
            results = _run_parallel(tasks, args.n_workers)[args.benchmark]
        report(args.benchmark, cfg, results, bench_dir, elapsed=time.time() - t0)

    elif args.sweep:
        configs = load_sweep(args.sweep)
        tasks = [
            (name, cfg, seed)
            for name, cfg in configs.items()
            for seed in range(cfg.n_seeds)
        ]
        t0 = time.time()
        if args.n_workers <= 1:
            grouped = {name: _run_sequential(cfg) for name, cfg in configs.items()}
        else:
            print(f"Running {len(tasks)} tasks across {args.n_workers} workers")
            grouped = _run_parallel(tasks, args.n_workers)
        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.1f}s")
        t0 = time.time()
        for name, cfg in configs.items():
            report(name, cfg, grouped[name], sweep_dir, verbose=False)
        print(f"Completed in {time.time() - t0:.1f}s")

    else:
        configs = {k: v for k, v in load_all_benchmarks().items() if k != "debug"}
        for name, cfg in configs.items():
            _print_config(name, cfg)
            tasks = [(name, cfg, seed) for seed in range(cfg.n_seeds)]
            t0 = time.time()
            if args.n_workers <= 1:
                results = _run_sequential(cfg, verbose=True)
            else:
                results = _run_parallel(tasks, args.n_workers)[name]
            report(name, cfg, results, bench_dir, elapsed=time.time() - t0)


if __name__ == "__main__":
    main()
