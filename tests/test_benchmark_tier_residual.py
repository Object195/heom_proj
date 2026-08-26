"""Tier-resolved MLP residual diagnostics."""

import unittest

import numpy as np
import torch

from benchmark.benchmark_mlp import compute_tier_residual_statistics
from heom.heom_rep import heom_state
from model import HEOMMLP, HEOMPINNLoss, state_and_time_derivative


def make_problem(depth=2):
    h_system = np.array([[0.5, 0.1], [0.1, -0.5]], dtype=np.complex128)
    h_coupling = np.diag([1.0, -1.0]).astype(np.complex128)
    hierarchy = heom_state(
        K=0,
        L=depth,
        H_s=h_system,
        H_c=h_coupling,
        C_list=np.array([0.2 + 0.1j]),
        gamma_list=np.array([0.3 + 1.0j]),
    )
    rho0 = np.diag([1.0, 0.0]).astype(np.complex128)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    return hierarchy, rho0, liouvillian


class TierResidualStatisticsTests(unittest.TestCase):
    def test_statistics_match_direct_tier_formula(self):
        hierarchy, rho0, liouvillian = make_problem()
        model = HEOMMLP(
            hierarchy,
            hidden_sizes=(6,),
            rho0=rho0,
            t_start=0.0,
            t_stop=0.2,
            dtype=torch.float64,
        )
        t_eval = np.linspace(0.0, 0.2, 5)

        statistics = compute_tier_residual_statistics(
            model,
            hierarchy,
            liouvillian,
            t_eval,
            batch_size=2,
        )

        objective = HEOMPINNLoss(
            hierarchy,
            liouvillian=liouvillian,
            dtype=torch.float64,
        )
        times = torch.as_tensor(t_eval, dtype=torch.float64)
        state, derivative = state_and_time_derivative(
            model,
            times,
            create_graph=False,
        )
        residual = derivative - objective.rhs(state)
        residual_u, residual_v = residual.split(model.state_size, dim=-1)
        energy = (
            residual_u.reshape(-1, model.n_ados, model.system_size).square()
            + residual_v.reshape(
                -1,
                model.n_ados,
                model.system_size,
            ).square()
        )

        by_tier = {item.tier: item for item in statistics}
        self.assertEqual(
            {tier: item.n_ados for tier, item in by_tier.items()},
            {0: 1, 1: 2, 2: 3},
        )
        for tier, item in by_tier.items():
            indices = torch.as_tensor(
                [
                    index
                    for index, node in enumerate(hierarchy.idx_to_node)
                    if hierarchy._tier(node) == tier
                ],
                dtype=torch.long,
            )
            tier_energy = energy.index_select(1, indices)
            time_error = tier_energy.mean(dim=(1, 2))
            self.assertAlmostEqual(
                item.mean_squared_residual,
                time_error.mean().item(),
            )
            self.assertAlmostEqual(
                item.max_time_squared_residual,
                time_error.max().item(),
            )
            self.assertAlmostEqual(
                item.max_component_residual,
                tier_energy.max().sqrt().item(),
            )

        weighted_tier_mean = sum(
            item.n_ados * item.mean_squared_residual for item in statistics
        ) / hierarchy.nADO
        global_mean = objective(model, times).detach().item()
        self.assertAlmostEqual(weighted_tier_mean, global_mean)


if __name__ == "__main__":
    unittest.main()
