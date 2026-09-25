"""Restart timing, whole-tier replacement, numerical accuracy, and checkpoint CLI."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
import torch

from benchmark.benchmark_mlp import build_normalized_hard_heom, run_mlp_solver
from benchmark.benchmark_mlp_restart import (
    build_hybrid_state,
    build_restart_time_grids,
    main,
    plot_restart_diagnostic,
    run_restart_diagnostic,
)
from experiment_parameters import MLP, MLP_MODEL_PATH, PSEUDOMODE
from model import HEOMMLP, solve_mlp
from training_sequence import (
    TrainingSequence, TrainingSession, save_training_metadata,
    training_incomplete_marker_path,
)


def make_problem(*, positive=False, time_switch="linear"):
    parameters = replace(
        PSEUDOMODE, heom_depth=2, t_start=0.3, t_stop=0.5, n_times=7,
        g=0.4, v=0.3, rtol=1e-10, atol=1e-12,
    )
    mlp_parameters = replace(
        MLP, hidden_sizes=(6,), device="cpu", dtype="float64",
        positive_rdm_ansatz=positive, time_switch=time_switch,
        ansatz_scale_normalization=False,
    )
    hierarchy, prepared, liouvillian = build_normalized_hard_heom(parameters)
    # The reference must honor the model buffer, not reconstruct the default
    # product-state run. Deliberately choose a different valid root anchor.
    anchor = prepared.copy()
    anchor[:4] = np.array([0.7, 0.05 - 0.1j, 0.05 + 0.1j, 0.3])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = HEOMMLP(
            hierarchy, hidden_sizes=mlp_parameters.hidden_sizes,
            initial_heom_state=anchor, t_start=parameters.t_start,
            t_stop=parameters.t_stop, positive_rdm_ansatz=positive,
            time_switch=time_switch, dtype=torch.float64, device="cpu",
        )
    return model, hierarchy, liouvillian, parameters, mlp_parameters


class RestartDiagnosticTests(unittest.TestCase):
    def test_default_and_custom_grids_keep_physical_anchor(self):
        parameters = replace(PSEUDOMODE, t_start=10.0, t_stop=12.0, n_times=5)
        times, reference_times, offset = build_restart_time_grids(parameters)
        np.testing.assert_array_equal(times, np.linspace(12.0, 14.0, 5))
        np.testing.assert_array_equal(reference_times, np.r_[10.0, times])
        self.assertEqual(offset, 1)
        times, reference_times, offset = build_restart_time_grids(
            parameters, restart_time=10.0, t_stop=11.0, n_times=3,
        )
        np.testing.assert_array_equal(times, [10.0, 10.5, 11.0])
        self.assertIs(times, reference_times)
        self.assertEqual(offset, 0)
        for options in (
            {"restart_time": 9.0}, {"restart_time": float("nan")},
            {"t_stop": 12.0}, {"t_stop": float("inf")}, {"n_times": 1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_restart_time_grids(parameters, **options)

    def test_hybrid_copies_whole_tiers_without_mutating_inputs(self):
        _, hierarchy, _, _, _ = make_problem()
        size = hierarchy.nADO * hierarchy.system_size
        reference = np.arange(size) + 1j * np.arange(size)[::-1]
        predicted = -reference - 1.0j
        reference_copy, predicted_copy = reference.copy(), predicted.copy()
        for cutoff in range(hierarchy.L + 1):
            hybrid = build_hybrid_state(hierarchy, reference, predicted, cutoff)
            for index, node in enumerate(hierarchy.idx_to_node):
                block = slice(index * hierarchy.system_size, (index + 1) * hierarchy.system_size)
                source = reference if hierarchy._tier(node) <= cutoff else predicted
                np.testing.assert_array_equal(hybrid[block], source[block])
            self.assertFalse(np.shares_memory(hybrid, reference))
            self.assertFalse(np.shares_memory(hybrid, predicted))
        np.testing.assert_array_equal(reference, reference_copy)
        np.testing.assert_array_equal(predicted, predicted_copy)
        for cutoff in (-1, 3, 0.5, True):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                build_hybrid_state(hierarchy, reference, predicted, cutoff)
        with self.assertRaisesRegex(ValueError, "shape"):
            build_hybrid_state(hierarchy, reference[:-1], predicted, 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            build_hybrid_state(hierarchy, reference, predicted * np.nan, 0)

    def test_continuations_match_matrix_exponentials_and_metrics(self):
        problem = make_problem()
        model, _, liouvillian, parameters, _ = problem
        with patch(
            "benchmark.benchmark_mlp_restart.run_mlp_solver", wraps=run_mlp_solver,
        ) as inference:
            result = run_restart_diagnostic(*problem, lower_tier_cutoff=0)
        self.assertEqual(inference.call_count, 1)
        np.testing.assert_array_equal(inference.call_args.args[1], [parameters.t_stop])
        dense = liouvillian.toarray()
        anchor = model.complex_initial_state().detach().numpy()
        expected_reference = np.column_stack([
            expm(dense * (time - parameters.t_start)) @ anchor for time in result.t
        ])
        np.testing.assert_allclose(result.reference_state, expected_reference, atol=3e-9, rtol=0)
        np.testing.assert_allclose(
            result.reference_sigma_z, (expected_reference[0] - expected_reference[3]).real,
            atol=3e-9, rtol=0,
        )
        expected_prediction = solve_mlp(model, result.t[:1]).y[:, 0]
        np.testing.assert_array_equal(result.predicted_state, expected_prediction)
        a, b = result.trajectories
        self.assertEqual((a.mode, b.mode), ("A", "B"))
        np.testing.assert_array_equal(a.initial_state, expected_prediction)
        np.testing.assert_array_equal(b.initial_state[:4], result.reference_state[:4, 0])
        np.testing.assert_array_equal(b.initial_state[4:], expected_prediction[4:])
        self.assertAlmostEqual(b.sigma_z[0], result.reference_sigma_z[0])
        self.assertGreater(a.full_state_mae, 1e-6)
        self.assertGreater(b.full_state_mae, 1e-6)
        for trajectory in result.trajectories:
            expected = np.column_stack([
                expm(dense * (time - result.t[0])) @ trajectory.initial_state
                for time in result.t
            ])
            np.testing.assert_allclose(trajectory.state, expected, atol=3e-9, rtol=0)
            self.assertAlmostEqual(
                trajectory.full_state_mae,
                float(np.abs(trajectory.state - result.reference_state).mean()),
            )
            self.assertAlmostEqual(
                trajectory.sigma_z_mae,
                float(np.abs(trajectory.sigma_z - result.reference_sigma_z).mean()),
            )

    def test_exact_restart_control_and_restart_at_initial_time(self):
        problem = make_problem()
        result = run_restart_diagnostic(*problem, mode="B", lower_tier_cutoff=2)
        self.assertEqual(len(result.trajectories), 1)
        self.assertLess(result.trajectories[0].sigma_z_mae, 3e-9)
        self.assertLess(result.trajectories[0].full_state_mae, 3e-9)
        model, _, _, parameters, _ = problem
        result = run_restart_diagnostic(
            *problem, mode="A", restart_time=parameters.t_start,
        )
        np.testing.assert_allclose(
            result.reference_state[:, 0], model.complex_initial_state().detach().numpy(),
        )
        self.assertLess(result.trajectories[0].full_state_mae, 3e-9)

    def test_plot_has_metrics_and_saves_both_panels(self):
        problem = make_problem()
        result = run_restart_diagnostic(*problem)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "restart.png"
            figure = plot_restart_diagnostic(result, problem[3], output=output, show=False)
            try:
                self.assertTrue(output.is_file())
                self.assertEqual(len(figure.axes), 2)
                for axis, trajectory in zip(figure.axes, result.trajectories):
                    self.assertIn(f"E_z={trajectory.sigma_z_mae:.1e}", axis.get_title())
                    self.assertIn(f"E_H={trajectory.full_state_mae:.1e}", axis.get_title())
                    np.testing.assert_array_equal(axis.lines[0].get_ydata(), result.reference_sigma_z)
                    np.testing.assert_array_equal(axis.lines[1].get_ydata(), trajectory.sigma_z)
                self.assertIn("L=2", figure._suptitle.get_text())
            finally:
                plt.close(figure)

    def test_checkpoint_cli_defaults_picker_and_positive_exponential_model(self):
        for positive, switch in ((False, "linear"), (True, "exponential")):
            with self.subTest(positive=positive), TemporaryDirectory() as directory:
                model, hierarchy, liouvillian, parameters, mlp_parameters = make_problem(
                    positive=positive, time_switch=switch,
                )
                path = Path(directory) / MLP_MODEL_PATH.name
                sequence = TrainingSequence(
                    pseudomode=parameters, base_mlp=mlp_parameters,
                    sessions=(TrainingSession("test", mlp_parameters),),
                )
                torch.save(model.state_dict(), path)
                save_training_metadata(sequence, path)
                extra = Path(directory) / "extra" / "copy.png"
                log = StringIO()
                with patch(
                    "benchmark.benchmark_mlp_restart.resolve_default_model_path", return_value=path,
                ) as picker, redirect_stdout(log):
                    result = main(["--no-show", "--output", str(extra)])
                picker.assert_called_once_with(folder_dialog=True)
                self.assertTrue((path.parent / "restart_trajectory.png").is_file())
                self.assertTrue(extra.is_file())
                self.assertIn("Mean absolute <sigma_z> error", log.getvalue())
                self.assertIn("normalized-HEOM state-component error", log.getvalue())
                self.assertEqual(result.t[0], parameters.t_stop)
                self.assertAlmostEqual(result.t[-1], 0.7)
                expected = expm(liouvillian.toarray() * 0.2) @ model.complex_initial_state().numpy()
                np.testing.assert_allclose(result.reference_state[:, 0], expected, atol=3e-9, rtol=0)
                with redirect_stdout(StringIO()):
                    control = main([
                        "--model-path", str(path.parent), "--mode", "B",
                        "--lower-tier-cutoff", str(hierarchy.L), "--no-show", "--device", "cpu",
                    ])
                self.assertLess(control.trajectories[0].full_state_mae, 3e-9)

    def test_missing_metadata_and_incomplete_checkpoint_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / MLP_MODEL_PATH.name
            path.touch()
            for incomplete in (False, True):
                if incomplete:
                    training_incomplete_marker_path(path).touch()
                log = StringIO()
                with redirect_stderr(log), self.assertRaises(SystemExit) as error:
                    main(["--model-path", str(path), "--no-show"])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("incomplete" if incomplete else "no training metadata", log.getvalue())


if __name__ == "__main__":
    unittest.main()
