"""Numerical checks for time-resolved hierarchy activity and derivatives."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from benchmark.benchmark_lindbladian import (
    compute_tier_time_diagnostics,
    plot_tier_time_diagnostics,
    select_plot_tiers,
)
from heom.heom_rep import heom_state
from heom.heom_solver import solve_heom


def make_hierarchy(depth=2):
    return heom_state(
        K=0, L=depth,
        H_s=np.array([[0.5, 0.1], [0.1, -0.5]], dtype=complex),
        H_c=np.diag([1.0, -1.0]).astype(complex),
        C_list=np.array([0.2]),
        gamma_list=np.array([0.3 + 1j]),
    )


class LindbladianTierDiagnosticsTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_rms_uses_complex_magnitudes_and_averages_over_components(self):
        hierarchy = make_hierarchy()
        state = np.zeros((hierarchy.nADO * hierarchy.system_size, 3), complex)
        # One populated component per ADO gives RMS 5 / sqrt(4) = 2.5,
        # independently of the number of ADOs at each tier.
        state[::hierarchy.system_size] = np.array([0, 3 + 4j, 6 + 8j])
        diagnostic = compute_tier_time_diagnostics(
            hierarchy, hierarchy.build_Liouvillian(), state,
        )
        np.testing.assert_array_equal(diagnostic.n_ados, [1, 2, 3])
        np.testing.assert_allclose(diagnostic.ado_rms, [[0, 2.5, 5]] * 3)

    def test_opposite_source_ados_cancel_before_derivative_norm(self):
        hierarchy = make_hierarchy(depth=1)
        liouvillian = hierarchy.build_Liouvillian()
        state = np.zeros((hierarchy.nADO * hierarchy.system_size, 2), complex)
        sources = [i for i, node in enumerate(hierarchy.idx_to_node)
                   if hierarchy._tier(node) == 1]
        # An off-diagonal component has nonzero commutator with sigma_z.
        state[sources[0] * hierarchy.system_size + 1] = 1j
        state[sources[1] * hierarchy.system_size + 1] = [-1j, 1j]
        diagnostic = compute_tier_time_diagnostics(hierarchy, liouvillian, state)
        np.testing.assert_allclose(diagnostic.ado_rms[1], 0.5)
        np.testing.assert_allclose(diagnostic.derivative_rms[0], [0, 2])

    def test_initial_derivative_includes_root_dynamics_and_upward_excitation(self):
        hierarchy = make_hierarchy()
        state = hierarchy.build_initial_state(np.diag([1.0, 0.0]), as_sparse=False)
        diagnostic = compute_tier_time_diagnostics(
            hierarchy, hierarchy.build_Liouvillian(), state[:, None],
        )
        # Root coherences have derivatives +/-0.1j. Each tier-one ADO has
        # one derivative component of magnitude 0.2 despite its zero state.
        np.testing.assert_allclose(
            diagnostic.derivative_rms[:, 0], [0.1 / np.sqrt(2), 0.1, 0],
        )
        np.testing.assert_allclose(diagnostic.ado_rms[1:], 0)

    def test_root_derivative_agrees_between_normalization_conventions(self):
        hierarchy = make_hierarchy()
        rho0 = np.diag([1.0, 0.0]).astype(complex)
        times = np.linspace(0, 0.5, 11)
        diagnostics = []
        roots = []
        for normalized in (False, True):
            liouvillian = hierarchy.build_Liouvillian(normalized=normalized)
            solution = solve_heom(
                hierarchy, rho0, times, liouvillian=liouvillian,
                rtol=1e-10, atol=1e-12,
            )
            diagnostics.append(compute_tier_time_diagnostics(
                hierarchy, liouvillian, solution.y,
            ))
            roots.append(solution.primary_ados)
        np.testing.assert_allclose(roots[0], roots[1], atol=1e-9)
        np.testing.assert_allclose(
            diagnostics[0].derivative_rms[0], diagnostics[1].derivative_rms[0],
            atol=1e-9,
        )
        # Tier amplitudes themselves do change under ADO rescaling.
        self.assertFalse(np.allclose(
            diagnostics[0].ado_rms[1:], diagnostics[1].ado_rms[1:],
        ))

    def test_derivative_time_mean_matches_mlp_dynamical_activity(self):
        import torch

        from benchmark.benchmark_mlp import compute_tier_residual_statistics
        from model import HEOMMLP

        hierarchy = make_hierarchy()
        liouvillian = hierarchy.build_Liouvillian(normalized=True)
        times = np.linspace(0.0, 0.2, 5)
        rho0 = np.diag([1.0, 0.0]).astype(complex)
        solution = solve_heom(hierarchy, rho0, times, liouvillian=liouvillian)
        diagnostic = compute_tier_time_diagnostics(hierarchy, liouvillian, solution.y)
        model = HEOMMLP(
            hierarchy, hidden_sizes=(6,), rho0=rho0,
            t_start=times[0], t_stop=times[-1], dtype=torch.float64,
        )
        mlp_statistics = compute_tier_residual_statistics(
            model, hierarchy, liouvillian, times, reference_state=solution.y,
        )
        np.testing.assert_allclose(
            np.mean(diagnostic.derivative_rms ** 2, axis=1),
            [item.mean_squared_dynamical_activity for item in mlp_statistics],
            rtol=1e-12, atol=1e-15,
        )

    def test_many_tiers_remain_in_heatmap_when_traces_are_selected(self):
        hierarchy = make_hierarchy(depth=20)
        times = np.array([0.0, 0.01, 0.5])
        state = np.zeros((hierarchy.nADO * hierarchy.system_size, len(times)))
        diagnostic = compute_tier_time_diagnostics(
            hierarchy, hierarchy.build_Liouvillian(), state,
        )
        selected = select_plot_tiers(diagnostic.tiers)
        self.assertLessEqual(len(selected), 7)
        self.assertTrue({0, 1, 2, 20}.issubset(selected))
        axes = plot_tier_time_diagnostics(
            times, diagnostic, calculation_name="test", selected_tiers=[0, 5, 20],
        )
        self.assertEqual(axes[0, 0].collections[0].get_array().size, 21 * 3)
        self.assertEqual(axes[1, 0].collections[0].get_array().size, 21 * 3)
        self.assertEqual(len(axes[0, 1].lines), 3)
        self.assertEqual(len(axes[1, 1].lines), 3)
        self.assertTrue(axes[0, 0].collections[0].get_array().mask.all())
        axes[0, 0].figure.canvas.draw()

    def test_root_only_hierarchy_and_invalid_selections(self):
        hierarchy = make_hierarchy(depth=0)
        state = np.tile(hierarchy.build_initial_state(np.diag([1.0, 0.0]),
                                                      as_sparse=False)[:, None], (1, 2))
        diagnostic = compute_tier_time_diagnostics(
            hierarchy, hierarchy.build_Liouvillian(), state,
        )
        self.assertEqual(diagnostic.derivative_rms.shape, (1, 2))
        np.testing.assert_allclose(diagnostic.derivative_rms, 0.1 / np.sqrt(2))
        axes = plot_tier_time_diagnostics(
            [0, 1], diagnostic, calculation_name="root only",
        )
        axes[0, 0].figure.canvas.draw()
        with self.assertRaises(ValueError):
            select_plot_tiers(diagnostic.tiers, [1])


if __name__ == "__main__":
    unittest.main()
