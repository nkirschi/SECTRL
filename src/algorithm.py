from dynamics import ContinuousLQREnv
from estimator import ContinuousLeastSquaresEstimator
from planner import RiccatiODESolver

import numpy as np


def learn(
    A_true, B_true, sigma, x0, Q, R, config, cycles=5, m0=1, exploration_scale=0.1
):
    env = ContinuousLQREnv(A_true, B_true, sigma, x0, config)
    estimator = ContinuousLeastSquaresEstimator(
        config
    )  # note: using all past trajectories as opposed to only current cycle's to decrease variance, even though this now correlates trajectories across epochs
    riccati = RiccatiODESolver(config, Q, R)

    # shadow simulation to obtain "what-if-i-had-behaved-optimally" cost
    env_shadow = ContinuousLQREnv(A_true, B_true, sigma, x0, config)
    riccati_opt = RiccatiODESolver(config, Q, R)
    riccati_opt.solve(A_true, B_true)

    def cost_function(x, u):
        return x.T @ Q @ x + u.T @ R @ u

    # zero initialisation of parameters (leads to u = 0 in the first cycle)
    A_hat = np.zeros_like(A_true)
    B_hat = np.zeros_like(B_true)

    print(f"{'Cycle':<5} | {'Episodes':<8} | {'Param Error':<12} | {'Avg Cost':<10}")
    print("-" * 45)

    results = {
        "costs": [],
        "regrets": [],
        "thetas": [],
        "m_l": [],
    }

    for l in range(cycles):
        m_l = m0 * (2**l)  # Theory requires duration T_l to grow geometrically.

        # 1. Update Policy
        # Solve Riccati using CURRENT best estimates
        riccati.solve(A_hat, B_hat)

        # 2. Schedule Exploration (Dithering)
        # As our estimates get better (l increases), we reduce noise.
        # But we never turn it off completely (to maintain rank).
        exploration_std = exploration_scale / np.sqrt(l + 1)

        cycle_costs = []
        cycle_regrets = []

        # Step 2: Execute m_l episodes
        for _ in range(m_l):
            state = env.reset()
            _ = env_shadow.reset()
            state_shadow = state.copy()
            trajectory = []
            episode_cost = 0.0
            episode_regret = 0.0

            # Simulate Episode
            while env.t < config.T:
                # Compute Control u = K(t)x
                u = riccati.get_K(env.t, B_hat) @ state

                # b. Add Exploration Noise (Excitation)
                u += np.random.normal(0, exploration_std, size=config.u_dim)

                cost_learner = cost_function(state, u)

                # Environment step
                shared_noise = np.random.normal(size=config.x_dim)
                new_state, dx, dt = env.step(u, noise_override=shared_noise)

                # Record Data
                trajectory.append((state.copy(), u, dx, dt))
                state = new_state

                # Shadow simulation
                u_opt = riccati_opt.get_K(env_shadow.t, B_true) @ state_shadow
                cost_opt = cost_function(state_shadow, u_opt)
                state_shadow, _, _ = env_shadow.step(u_opt, noise_override=shared_noise)

                # Accumulate cost and regret
                episode_cost += cost_learner * dt
                episode_regret += (cost_learner - cost_opt) * dt

            estimator.add_trajectory(trajectory)

            cycle_costs.append(episode_cost.item())
            cycle_regrets.append(episode_regret.item())

        # Step 3: Update Estimation
        A_hat, B_hat = estimator.estimate()

        results["costs"].extend(cycle_costs)
        results["regrets"].extend(cycle_regrets)
        results["thetas"].append((A_hat, B_hat))
        results["m_l"].append(m_l)

        # Metrics
        err = np.linalg.norm(A_hat - A_true) + np.linalg.norm(B_hat - B_true)
        print(f"{l:<5} | {m_l:<8} | {err:<12.4f} | {np.mean(cycle_costs):<10.2f}")

    return results
