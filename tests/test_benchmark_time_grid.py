"""Benchmark time-grid and reference warm-up checks."""

from dataclasses import replace
import unittest

import numpy as np

from benchmark.benchmark_mlp import build_benchmark_time_grids
from experiment_parameters import PSEUDOMODE


class BenchmarkTimeGridTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
