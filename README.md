# SAMBL: Sparsity-Aware Model-Based Learning

Model-based RL for episodic continuous-time linear-quadratic control where the
dynamics `Θ⋆ = [A⋆ | B⋆]` are row-sparse. 
Our proposed algorithm, **SAMBL** (*Sparsity-Aware Model-Based Learning*), equips a certainty-equivalent agent with a row-wise LASSO dynamics estimator that exploits the sparsity.
The experiments show that SAMBL is more sample efficient; in particular, it attains lower regret than the dense OLS baseline as the dimension grows, on synthetic systems and on two structured benchmarks (a spring-mass chain and the IEEE 39-bus power grid).

The project uses [`uv`](https://docs.astral.sh/uv/). 
Every command below is prefixed with `uv run` so it executes inside the locked environment.

## Directory structure

```
.
├── src/
│   ├── main.py               # CLI entry point: run a benchmark / sweep / debug run
│   ├── runner.py             # per-seed episode loop, agent construction, RNG handling
│   ├── results_io.py         # per-seed NPZ + JSON config persistence and loaders
│   ├── common.py             # config dataclasses (ExperimentConfig, SystemConfig, ...)
│   ├── system_generator.py   # all system samplers (synthetic, spring chain, IEEE 39-bus)
│   ├── ieee39_data.py        # baked IEEE 39-bus topology + inertia data (MATPOWER case39)
│   ├── dynamics.py           # continuous-time SDE simulation
│   ├── agent.py              # generic linear control agent
│   ├── estimator.py          # row-LASSO and OLS estimators (incremental Gram)
│   ├── planner.py            # Riccati solver (Hamiltonian matrix-exponential)
│   ├── diagnostics.py        # per-episode diagnostics (RE, support recovery, ...)
│   ├── figures.py            # regenerate all figures from results/
│   ├── metrics.py            # aggregation + statistics over a result tree
│   └── dashboard.py          # development plots
├── configs/
│   ├── benchmarks/           # single-system configs (one YAML = one study)
│   └── sweeps/               # parameter sweeps (a base + named `vary` points)
├── notebooks/                # analysis + figure-illustration notebooks
├── results/                  # produced result tree (see "Result layout" below)
├── tests/                    # pytest suite
├── job.slurm                 # SLURM wrapper around src/main.py
└── pyproject.toml            # uv project / dependencies
```

## Running experiments

A study is either:
- a **benchmark** (one system, `configs/benchmarks/<name>.yaml`)
- a **sweep** (a base config plus named override points, `configs/sweeps/<name>.yaml`)

### Locally

```bash
# one benchmark (parallelise across seeds)
uv run python src/main.py --benchmark ieee39 --n-workers 20

# one sweep (parallelise across all (point, seed) pairs)
uv run python src/main.py --sweep synthetic --n-workers 20

# quick smoke run (small d, few seeds)
uv run python src/main.py --debug

# regenerate plots/summaries in place from an existing result dir
uv run python src/main.py --replot results/synthetic
```

Available flags: 
- `--n-workers N` (process pool, default 1 = serial)
- `--output-dir DIR`
(default "results/"), 
- `--plots` (also render the dev dashboards)

### On a cluster (SLURM)

`job.slurm` forwards its two positional arguments to `--<mode> <name>`, sets `--n-workers` from `--cpus-per-task`, forwards `--plots`, and puts the user-local TeX Live on `PATH` (so the usetex dashboards render in batch).
For *single-system benchmarks* set `--cpus-per-task` to the seed count so every seed runs in parallel and the wall time is one seed's runtime.

The limits below are the ones that actually worked in our environment. Jobs ran on AMD EPYC 7642 nodes with 48 physical cores (96 threads) and roughly 400 GB RAM. 
The `OMP_NUM_THREADS=1` set in `job.slurm` ensures each of the workers gets one single-threaded BLAS core.
Cost is dominated by the cold-start LASSO of episode 1 and the dense `O(d³)` Riccati solve, so *both time and memory climb steeply with the dimension*.

```bash
# --- single-system benchmark ---
sbatch --cpus-per-task=20 --mem=16GB --time=00:30:00 job.slurm benchmark ieee39

# --- multi-point sweeps ---
sbatch --cpus-per-task=80 --mem=256GB --time=18:00:00 job.slurm sweep synthetic
sbatch --cpus-per-task=80 --mem=256GB --time=18:00:00 job.slurm sweep springs
sbatch --cpus-per-task=80 --mem=256GB --time=18:00:00 job.slurm sweep sparsity
sbatch --cpus-per-task=20 --mem=16GB --time=02:00:00 job.slurm sweep clambda
sbatch --cpus-per-task=20 --mem=16GB --time=02:00:00 job.slurm sweep excitation
sbatch --cpus-per-task=20 --mem=16GB --time=02:00:00 job.slurm sweep cost
sbatch --cpus-per-task=20 --mem=16GB --time=02:00:00 job.slurm sweep discretisation
```

### Available studies

| kind       | name(s)                | what it is                                             |
|------------|------------------------|--------------------------------------------------------|
| benchmark  | `synthetic`            | canonical synthetic system (base for several sweeps)   |
| benchmark  | `springs`              | spring-mass chain (base for the spring sweep)          |
| benchmark  | `ieee39`               | IEEE 39-bus power grid (d=78, p=9)                     |
| sweep      | `synthetic`            | dimension sweep d ∈ {10,20,50,100,200,500}, p=d/2      |
| benchmark  | `synthetic_noexplore`  | dimension sweep with `m_explore=0` (greedy baseline)    |
| sweep      | `springs`              | dimension sweep on the spring chain                    |
| sweep      | `sparsity`             | fixed d=100, row sparsity s = 2..50                    |
| sweep      | `clambda`              | c_λ × d calibration                                    |
| sweep      | `excitation`           | excitation-scale σ_u sweep                             |
| sweep      | `cost`                 | control-cost r sweep at d=100                          |
| sweep      | `discretisation`       | Euler step Δt sweep at d=100                           |

## Result layout

Every run writes a standard per-seed tree (NPZ payload + a JSON config, no pickle):

```
results/<study>/<point>/
    config.json             # the exact ExperimentConfig
    seed_0.npz              # per-agent, per-episode trajectories + diagnostics
    seed_0_snapshots.npz    # heavy A_est/B_est checkpoint arrays (loaded on demand)
    seed_1.npz
    seed_1_snapshots.npz
    ...
    results.json            # aggregated summary
```

A `--benchmark <name>` writes a single point at `results/<name>/`, while a `--sweep <name>` writes one point per `vary` entry at `results/<name>/<point>/`. Runs resume naturally — a point directory just accumulates `seed_*.npz`.

## Generating Figures

Regenerate every figure from the result tree:

```bash
uv run python src/figures.py --out figures/             # -> figures/*.pdf
```

It reads `results/{synthetic,springs,ieee39,sparsity,clambda,excitation,cost}` (override
with `--synthetic-dir`, `--spring-dir`, `--ieee39-dir`, `--sparsity-dir`, `--clambda-dir`,
`--excitation-dir`, `--cost-dir`). System illustrations (spring-mass
schematic, IEEE 39-bus topology) are drawn by `figures.py` as well.

## Tests

The codebase is extensively tested for various theoretical properties and functional specifications.

```bash
uv run pytest -q
```
