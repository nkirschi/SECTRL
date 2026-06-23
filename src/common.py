from __future__ import annotations
from dataclasses import dataclass
import dataclasses
import numpy as np


@dataclass(frozen=True)
class SystemConfig:
    d: int
    p: int
    s_A: int
    s_B: int
    a_min: float
    a_max: float
    b_min: float
    b_max: float
    sigma: float
    dt: float
    T: float

    @property
    def H(self) -> int:
        """Steps per episode."""
        return int(round(self.T / self.dt))

    @property
    def sparsity(self) -> int:
        """Total nonzeros per row = a_nonzeros + b_nonzeros."""
        return self.s_A + self.s_B

    @property
    def sigma_bar(self):
        return self.sigma / np.sqrt(self.dt)


@dataclass(frozen=True)
class CostConfig:
    q_scale: float
    r_scale: float


@dataclass(frozen=True)
class EstimatorConfig:
    mu_ridge: float
    lambda_lasso: float | None
    c_lambda: float
    delta: float
    lasso_max_iter: int
    lasso_tol: float
    lambda_warmup: float = 1e-6


@dataclass(frozen=True)
class ExcitationConfig:
    sigma_u: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Full configuration for one experimental benchmark."""

    max_episodes: int
    n_seeds: int
    agents: tuple[str, ...]
    x0_std: float
    m_explore: float
    action_clip: float
    state_clip: float
    support_threshold: float
    system: SystemConfig
    cost: CostConfig
    estimators: EstimatorConfig
    excitation: ExcitationConfig


    @property
    def theoretical_speedup(self) -> float:
        d_plus_p = self.system.d + self.system.p
        return d_plus_p / (self.system.sparsity * np.log(d_plus_p))

    def theoretical_lambda(self, N):
        d, p, M = self.system.d, self.system.p, self.max_episodes
        log_term = np.log((d + p) * M * d / self.estimators.delta)
        return self.estimators.c_lambda * self.system.sigma_bar * np.sqrt(log_term / N)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)

        sys = raw.get("system", {})
        sim = raw.get("simulation", {})
        cost = raw.get("cost", {})
        est = raw.get("estimation", {})
        exc = raw.get("excitation", {})
        train = raw.get("training", {})

        system_cfg = SystemConfig(
            d=sys["d"],
            p=sys["p"],
            s_A=sys["s_A"],
            s_B=sys["s_B"],
            a_min=sys["a_min"],
            a_max=sys["a_max"],
            b_min=sys["b_min"],
            b_max=sys["b_max"],
            sigma=sim["sigma"],
            dt=sim["dt"],
            T=sim["T"],
        )

        cost_cfg = CostConfig(
            q_scale=cost.get("q_scale", 1.0), r_scale=cost.get("r_scale", 1.0)
        )

        est_cfg = EstimatorConfig(
            mu_ridge=est.get("mu_ridge", 0.01),
            lambda_lasso=est.get("lambda_lasso", None),
            c_lambda=est.get("c_lambda", 2.0),
            delta=est.get("delta", 0.05),
            lasso_max_iter=est.get("lasso_max_iter", 5000),
            lasso_tol=float(est.get("lasso_tol", 1e-4)),
            lambda_warmup=float(est.get("lambda_warmup", 1e-6)),
        )

        exc_cfg = ExcitationConfig(sigma_u=exc.get("sigma_u", 0.1))

        agents = raw.get(
            "agents",
            (
                "oracle",
                "dense_greedy",
                "dense_excited",
                "sparse_greedy",
                "sparse_excited",
            ),
        )

        return cls(
            max_episodes=train.get("max_episodes", 100),
            n_seeds=train.get("n_seeds", 50),
            agents=tuple(agents),
            x0_std=sim.get("x0_std", 0.0),
            m_explore=train.get("m_explore", 0.0),
            action_clip=sim.get("action_clip", 10.0),
            state_clip=sim.get("state_clip", 100.0),
            support_threshold=est.get("support_threshold", 0.05),
            system=system_cfg,
            cost=cost_cfg,
            estimators=est_cfg,
            excitation=exc_cfg,
        )

    @classmethod
    def apply_overrides(
        cls, base: "ExperimentConfig", overrides: dict
    ) -> "ExperimentConfig":
        sub_configs = {
            "system":     base.system,
            "cost":       base.cost,
            "estimators": base.estimators,
            "excitation": base.excitation,
        }
        sub_overrides = {name: {} for name in sub_configs}
        top_overrides = {}

        for key, val in overrides.items():
            for sub_name, sub_cfg in sub_configs.items():
                if key in {f.name for f in dataclasses.fields(sub_cfg)}:
                    sub_overrides[sub_name][key] = val
                    break
            else:
                top_overrides[key] = val

        new_sub = {}
        for sub_name, sub_cfg in sub_configs.items():
            if sub_overrides[sub_name]:
                fields = {f.name: getattr(sub_cfg, f.name) for f in dataclasses.fields(sub_cfg)}
                fields.update(sub_overrides[sub_name])
                new_sub[sub_name] = type(sub_cfg)(**fields)
            else:
                new_sub[sub_name] = sub_cfg

        kwargs = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
        kwargs.update(new_sub)
        kwargs.update(top_overrides)
        return cls(**kwargs)