"""Benchmark time-grid and reference warm-up checks."""

from dataclasses import replace
import unittest

import numpy as np

from benchmark.benchmark_mlp import (
    _build_lindbladian_solver_grid,
    build_benchmark_time_grids,
    mean_absolute_error,
    resolve_default_model_path,
    run_lindbladian,
)
from experiment_parameters import MLP_MODEL_PATH, PSEUDOMODE


class BenchmarkTimeGridTests(unittest.TestCase):
    def test_mean_absolute_error_averages_complex_component_magnitudes(self):
        reference = np.array([[1.0 + 1.0j, 2.0], [3.0, 4.0 - 1.0j]])
        approximation = reference + np.array(
            [[3.0 + 4.0j, -2.0], [0.0, 1.0j]]
        )

        self.assertAlmostEqual(
            mean_absolute_error(reference, approximation),
            (5.0 + 2.0 + 0.0 + 1.0) / 4.0,
        )

    def test_mean_absolute_error_rejects_different_shapes(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            mean_absolute_error(np.zeros(2), np.zeros((2, 1)))

    def test_default_model_path_uses_mlp_folder_when_dialog_is_off(self):
        self.assertEqual(
            resolve_default_model_path(folder_dialog=False),
            MLP_MODEL_PATH,
        )

    def test_default_model_path_uses_folder_selected_by_dialog(self):
        selected_folder = MLP_MODEL_PATH.parent.parent / "selected_run"
        observed_initial_directories = []

        def select_folder(initial_directory):
            observed_initial_directories.append(initial_directory)
            return selected_folder

        self.assertEqual(
            resolve_default_model_path(
                folder_dialog=True,
                folder_selector=select_folder,
            ),
            selected_folder / MLP_MODEL_PATH.name,
        )
        self.assertEqual(
            observed_initial_directories,
            [MLP_MODEL_PATH.parent.parent],
        )

    def test_default_model_path_rejects_cancelled_dialog(self):
        with self.assertRaisesRegex(ValueError, "no model folder"):
            resolve_default_model_path(
                folder_dialog=True,
                folder_selector=lambda _initial_directory: None,
            )

    def test_default_grid_matches_training_interval(self):
        parameters = replace(
            PSEUDOMODE,
            t_start=0.0,
            t_stop=10.0,
            n_times=11,
        )

        t_eval, reference_t_eval, reference_offset = (
            build_benchmark_time_grids(parameters)
        )

        np.testing.assert_allclose(t_eval, np.linspace(0.0, 10.0, 11))
        self.assertIs(reference_t_eval, t_eval)
        self.assertEqual(reference_offset, 0)

    def test_later_interval_adds_initial_time_for_reference_warmup(self):
        parameters = replace(PSEUDOMODE, t_start=0.0, t_stop=10.0)

        t_eval, reference_t_eval, reference_offset = (
            build_benchmark_time_grids(
                parameters,
                t_start=10.0,
                t_stop=20.0,
                n_times=5,
            )
        )

        np.testing.assert_allclose(t_eval, np.linspace(10.0, 20.0, 5))
        np.testing.assert_allclose(
            reference_t_eval,
            np.concatenate(([0.0], t_eval)),
        )
        self.assertEqual(reference_offset, 1)

    def test_invalid_custom_intervals_are_rejected(self):
        parameters = replace(PSEUDOMODE, t_start=1.0, t_stop=10.0)
        invalid_options = (
            {"t_start": 0.0, "t_stop": 5.0},
            {"t_start": 2.0, "t_stop": 2.0},
            {"t_start": 2.0, "t_stop": float("inf")},
            {"t_start": 2.0, "t_stop": 5.0, "n_times": 1},
        )

        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    build_benchmark_time_grids(parameters, **options)

    def test_lindbladian_later_start_is_warmed_up_from_physical_zero(self):
        parameters = replace(PSEUDOMODE, cavity_dimension=3)
        later_times = np.array([5.0, 5.1])

        later_result = run_lindbladian(later_times, parameters)
        full_result = run_lindbladian(
            np.concatenate(([0.0], later_times)),
            parameters,
        )

        np.testing.assert_allclose(later_result, full_result[1:], atol=1e-12)

    def test_lindbladian_grid_caps_gaps_and_preserves_requested_times(self):
        requested = np.array([0.0, 0.25, 4.5, 5.0])

        solver_times, requested_indices = _build_lindbladian_solver_grid(
            requested,
            max_output_step=0.5,
        )

        self.assertLessEqual(np.diff(solver_times).max(), 0.5)
        np.testing.assert_allclose(solver_times[requested_indices], requested)

        later_requested = np.array([4.5, 5.0])
        later_solver_times, later_indices = _build_lindbladian_solver_grid(
            later_requested,
            max_output_step=0.5,
        )
        self.assertEqual(later_solver_times[0], 0.0)
        self.assertLessEqual(np.diff(later_solver_times).max(), 0.5)
        np.testing.assert_allclose(
            later_solver_times[later_indices],
            later_requested,
        )


if __name__ == "__main__":
    unittest.main()
