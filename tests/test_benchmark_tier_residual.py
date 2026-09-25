"""Tier-resolved MLP residual diagnostics."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark.benchmark_mlp import (
    compute_tier_ado_statistics,
    compute_tier_downward_forcing_statistics,
    compute_tier_residual_statistics,
    plot_tier_diagnostics,
)
from heom.heom_rep import heom_state
from heom.heom_solver import solve_heom
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

        weighted_tier_mean = sum(
            item.n_ados * item.mean_squared_residual for item in statistics
        ) / hierarchy.nADO
        global_mean = objective(model, times).detach().item()
        self.assertAlmostEqual(weighted_tier_mean, global_mean)

    def test_relative_residual_averages_per_ado_ratios(self):
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
        reference = solve_heom(
            hierarchy,
            rho0,
            t_eval,
            liouvillian=liouvillian,
        ).y
        statistics = compute_tier_residual_statistics(
            model,
            hierarchy,
            liouvillian,
            t_eval,
            reference_state=reference,
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
        residual_energy = (
            residual_u.reshape(-1, model.n_ados, model.system_size).square()
            + residual_v.reshape(
                -1,
                model.n_ados,
                model.system_size,
            ).square()
        ).detach().numpy()
        exact_derivative = liouvillian @ reference
        dynamical_energy = np.abs(
            exact_derivative.T.reshape(
                t_eval.size,
                hierarchy.nADO,
                hierarchy.system_size,
            )
        ) ** 2
        tier_to_indices = {
            tier: [
                index
                for index, node in enumerate(hierarchy.idx_to_node)
                if hierarchy._tier(node) == tier
            ]
            for tier in range(hierarchy.L + 1)
        }
        ado_residual_mean = residual_energy.mean(axis=(0, 2))
        ado_dynamical_mean = dynamical_energy.mean(axis=(0, 2))

        for item in statistics:
            indices = tier_to_indices[item.tier]
            self.assertAlmostEqual(
                item.mean_squared_dynamical_activity,
                ado_dynamical_mean[indices].mean(),
            )
            self.assertAlmostEqual(
                item.mean_relative_rms_residual,
                np.sqrt(
                    ado_residual_mean[indices]
                    / ado_dynamical_mean[indices]
                ).mean(),
            )

    def test_ado_statistics_match_direct_complex_state_formula(self):
        hierarchy, rho0, liouvillian = make_problem()
        t_eval = np.linspace(0.0, 0.2, 5)
        reference = solve_heom(
            hierarchy,
            rho0,
            t_eval,
            liouvillian=liouvillian,
        ).y
        perturbation = np.linspace(
            1e-4,
            2e-3,
            reference.size,
        ).reshape(reference.shape)
        predicted = reference + perturbation * (1.0 + 0.5j)

        statistics = compute_tier_ado_statistics(
            hierarchy,
            reference,
            predicted,
        )

        reference_energy = np.abs(
            reference.T.reshape(
                t_eval.size,
                hierarchy.nADO,
                hierarchy.system_size,
            )
        ) ** 2
        error_energy = np.abs(
            (predicted - reference).T.reshape(
                t_eval.size,
                hierarchy.nADO,
                hierarchy.system_size,
            )
        ) ** 2
        tier_to_indices = {
            tier: [
                index
                for index, node in enumerate(hierarchy.idx_to_node)
                if hierarchy._tier(node) == tier
            ]
            for tier in range(hierarchy.L + 1)
        }
        ado_activity = reference_energy.mean(axis=(0, 2))
        ado_error = error_energy.mean(axis=(0, 2))

        for item in statistics:
            indices = tier_to_indices[item.tier]
            self.assertAlmostEqual(
                item.mean_squared_activity,
                ado_activity[indices].mean(),
            )
            self.assertAlmostEqual(
                item.mean_squared_error,
                ado_error[indices].mean(),
            )
            self.assertAlmostEqual(
                item.mean_relative_rms_error,
                np.sqrt(ado_error[indices] / ado_activity[indices]).mean(),
            )

    def test_per_ado_relative_error_is_invariant_to_ado_rescaling(self):
        hierarchy, rho0, liouvillian = make_problem()
        t_eval = np.linspace(0.0, 0.2, 5)
        reference = solve_heom(
            hierarchy,
            rho0,
            t_eval,
            liouvillian=liouvillian,
        ).y
        predicted = reference + (2e-4 + 1e-4j)
        baseline = compute_tier_ado_statistics(
            hierarchy,
            reference,
            predicted,
        )

        ado_scales = np.geomspace(1e-3, 1e3, hierarchy.nADO)
        component_scales = np.repeat(ado_scales, hierarchy.system_size)
        scaled = compute_tier_ado_statistics(
            hierarchy,
            component_scales[:, None] * reference,
            component_scales[:, None] * predicted,
        )

        np.testing.assert_allclose(
            [item.mean_relative_rms_error for item in baseline],
            [item.mean_relative_rms_error for item in scaled],
            rtol=1e-12,
            atol=0.0,
        )

    def test_downward_forcing_matches_liouvillian_tier_block(self):
        hierarchy, rho0, liouvillian = make_problem()
        t_eval = np.linspace(0.0, 0.2, 5)
        reference = solve_heom(
            hierarchy,
            rho0,
            t_eval,
            liouvillian=liouvillian,
        ).y
        predicted = reference + (2e-4 + 1e-4j)

        statistics = compute_tier_downward_forcing_statistics(
            hierarchy,
            liouvillian,
            reference,
            predicted,
        )

        tier_to_ados = {
            tier: [
                index
                for index, node in enumerate(hierarchy.idx_to_node)
                if hierarchy._tier(node) == tier
            ]
            for tier in range(hierarchy.L + 1)
        }
        self.assertEqual(
            [item.source_tier for item in statistics],
            [1, 2],
        )
        for item in statistics:
            source_ados = tier_to_ados[item.source_tier]
            target_ados = tier_to_ados[item.target_tier]
            source_components = np.concatenate(
                [
                    np.arange(
                        index * hierarchy.system_size,
                        (index + 1) * hierarchy.system_size,
                    )
                    for index in source_ados
                ]
            )
            target_components = np.concatenate(
                [
                    np.arange(
                        index * hierarchy.system_size,
                        (index + 1) * hierarchy.system_size,
                    )
                    for index in target_ados
                ]
            )
            block = liouvillian[np.ix_(target_components, source_components)]
            exact = np.asarray(block @ reference[source_components, :])
            error = np.asarray(
                block @ (predicted - reference)[source_components, :]
            )
            force_shape = (
                len(target_ados),
                hierarchy.system_size,
                t_eval.size,
            )
            target_force = np.abs(exact.reshape(force_shape)) ** 2
            target_error = np.abs(error.reshape(force_shape)) ** 2
            target_force_mean = target_force.mean(axis=(1, 2))
            target_error_mean = target_error.mean(axis=(1, 2))
            self.assertEqual(item.n_source_ados, len(source_ados))
            self.assertEqual(item.n_target_ados, len(target_ados))
            self.assertAlmostEqual(
                item.mean_squared_forcing,
                target_force_mean.mean(),
            )
            self.assertAlmostEqual(
                item.mean_squared_error,
                target_error_mean.mean(),
            )
            self.assertAlmostEqual(
                item.mean_relative_rms_error,
                np.sqrt(target_error_mean / target_force_mean).mean(),
            )

    def test_downward_forcing_ratio_is_similarity_scale_invariant(self):
        hierarchy, rho0, liouvillian = make_problem()
        t_eval = np.linspace(0.0, 0.2, 5)
        reference = solve_heom(
            hierarchy,
            rho0,
            t_eval,
            liouvillian=liouvillian,
        ).y
        predicted = reference + (2e-4 + 1e-4j)
        baseline = compute_tier_downward_forcing_statistics(
            hierarchy,
            liouvillian,
            reference,
            predicted,
        )

        ado_scales = np.geomspace(1e-3, 1e3, hierarchy.nADO)
        component_scales = np.repeat(ado_scales, hierarchy.system_size)
        scaled_liouvillian = (
            component_scales[:, None]
            * liouvillian.toarray()
            / component_scales[None, :]
        )
        scaled = compute_tier_downward_forcing_statistics(
            hierarchy,
            scaled_liouvillian,
            component_scales[:, None] * reference,
            component_scales[:, None] * predicted,
        )

        np.testing.assert_allclose(
            [item.mean_relative_rms_error for item in baseline],
            [item.mean_relative_rms_error for item in scaled],
            rtol=1e-12,
            atol=0.0,
        )

    def test_tier_diagnostic_plot_groups_statistics_on_log_axes(self):
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
        reference = solve_heom(
            hierarchy,
            rho0,
            t_eval,
            liouvillian=liouvillian,
        ).y
        with torch.no_grad():
            prediction = model(torch.as_tensor(t_eval)).numpy()
        predicted_state = (
            prediction[:, : model.state_size]
            + 1j * prediction[:, model.state_size :]
        ).T
        residual_statistics = compute_tier_residual_statistics(
            model,
            hierarchy,
            liouvillian,
            t_eval,
            reference_state=reference,
            batch_size=2,
        )
        ado_statistics = compute_tier_ado_statistics(
            hierarchy,
            reference,
            predicted_state,
        )
        downward_forcing_statistics = (
            compute_tier_downward_forcing_statistics(
                hierarchy,
                liouvillian,
                reference,
                predicted_state,
            )
        )

        axes = plot_tier_diagnostics(
            residual_statistics,
            ado_statistics,
            downward_forcing_statistics,
            show=False,
        )

        self.assertEqual(axes.shape, (2,))
        self.assertTrue(all(axis.get_yscale() == "log" for axis in axes))
        self.assertEqual(len(axes[0].lines), 3)
        self.assertEqual(len(axes[1].lines), 6)
        plt.close(axes[0].figure)


if __name__ == "__main__":
    unittest.main()
