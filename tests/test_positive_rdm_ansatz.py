"""Physical constraints, derivatives, and checkpoint semantics of the PSD root."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmark.benchmark_mlp import (
    compute_tier_ado_statistics,
    compute_tier_residual_statistics,
    load_mlp,
    run_mlp_solver,
)
from experiment_parameters import MLP, PSEUDOMODE
from model import (
    HEOMMLP, HEOMPINNLoss, compute_heom_dynamical_scales,
    state_and_time_derivative,
)
from model.train_mlp_model import (
    _validate_pretrained_config,
    _validate_resume_config,
    build_training_problem,
    run_training_sequence,
)
from training_sequence import (
    TrainingSequence,
    TrainingSession,
    load_training_metadata,
    load_training_sequence,
    save_training_metadata,
)
from tests.test_mlp_baseline import make_hierarchy, make_rho0


def make_model(rho0=None, **options):
    settings = dict(
        rho0=make_rho0() if rho0 is None else rho0,
        hidden_sizes=(6,),
        t_start=0.4,
        t_stop=2.4,
        dtype=torch.float64,
        device="cpu",
        positive_rdm_ansatz=True,
    )
    settings.update(options)
    return HEOMMLP(make_hierarchy(depth=2), **settings)


def mixed_root():
    return np.array([[0.7, 0.1 + 0.2j], [0.1 - 0.2j, 0.3]])


class PositiveRDMTests(unittest.TestCase):
    def assert_density_matrices(self, root):
        torch.testing.assert_close(root, root.mH, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            root.diagonal(dim1=-2, dim2=-1).sum(-1),
            torch.ones(root.shape[0], dtype=root.dtype, device=root.device),
            atol=1e-12,
            rtol=0,
        )
        self.assertGreaterEqual(torch.linalg.eigvalsh(root).min().item(), -1e-12)

    def test_root_matches_unconstrained_factor_formula_for_both_switches(self):
        for rho0 in (make_rho0(), mixed_root()):
            for switch in ("linear", "exponential"):
                with self.subTest(rho0=rho0, switch=switch):
                    model = make_model(
                        rho0, time_switch=switch, switch_time_constant=0.7,
                        correction_scale=0.23,
                    )
                    times = [0.4, 0.7, 1.3, 2.4, 4.0]
                    raw = model.raw_output(times)[:, 0].detach().numpy()
                    # Independent column-major reconstruction of complex B.
                    b = np.array([
                        (row[:4] + 1j * row[4:]).reshape(2, 2, order="F")
                        for row in raw
                    ])
                    values, vectors = np.linalg.eigh(rho0)
                    sqrt_root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.conj().T
                    s = model.switching_function(times).detach().numpy()
                    a = sqrt_root + s[:, None, None] * 0.23 * b
                    gram = a @ a.conj().transpose(0, 2, 1)
                    expected = gram / np.trace(gram, axis1=1, axis2=2)[:, None, None]
                    actual = model.root_density_matrices(times)
                    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-13)
                    np.testing.assert_allclose(actual[0].detach().numpy(), rho0, atol=1e-13)
                    self.assert_density_matrices(actual)

    def test_nonroot_ados_keep_original_ansatz_and_partner_symmetry(self):
        for switch in ("linear", "exponential"):
            positive = make_model(time_switch=switch, correction_scale=0.3)
            legacy = make_model(
                positive_rdm_ansatz=False, time_switch=switch, correction_scale=0.3
            )
            legacy.load_state_dict(positive.state_dict())
            times = [0.4, 0.9, 2.4, 3.0]
            actual = positive.complex_states(times)
            torch.testing.assert_close(actual[:, 4:], legacy.complex_states(times)[:, 4:])
            matrices = actual.reshape(-1, positive.n_ados, 2, 2).transpose(-2, -1)
            torch.testing.assert_close(
                matrices, matrices[:, positive.conjugate_indices].mH
            )

    def test_jvp_matches_finite_difference_and_supports_loss_backpropagation(self):
        for switch in ("linear", "exponential"):
            model = make_model(mixed_root(), time_switch=switch, correction_scale=0.2)
            times = torch.tensor([0.4, 0.8, 1.7], dtype=torch.float64)
            _, derivative = state_and_time_derivative(model, times)
            step = 1e-5
            finite_difference = (model(times + step) - model(times - step)) / (2 * step)
            torch.testing.assert_close(derivative, finite_difference, rtol=2e-6, atol=1e-9)
            root_trace_derivative = derivative[:, 0] + derivative[:, 3]
            torch.testing.assert_close(root_trace_derivative, torch.zeros(3, dtype=torch.float64), atol=1e-13, rtol=0)
            objective = HEOMPINNLoss(model.hierarchy, tier_normalized=True)
            loss = objective(model, times)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()), 0)

    def test_parameter_gradient_including_time_derivative_matches_finite_difference(self):
        model = make_model(mixed_root(), correction_scale=0.4)
        objective = HEOMPINNLoss(model.hierarchy, tier_normalized=True)
        times = [0.4, 0.9, 1.5]
        objective(model, times).backward()
        parameter = model.network[-1].bias
        expected = parameter.grad[0].item()
        original = parameter[0].item()
        step = 1e-5
        with torch.no_grad():
            parameter[0] = original + step
        plus = objective(model, times).item()
        with torch.no_grad():
            parameter[0] = original - step
        minus = objective(model, times).item()
        np.testing.assert_allclose(expected, (plus - minus) / (2 * step), rtol=1e-5, atol=1e-8)

    def test_zero_factor_falls_back_to_initial_state_with_finite_derivatives(self):
        model = make_model()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.network[-1].bias[0] = -1.0
        # s(t_stop)=1, a_s=1, B=-sqrt(rho_init), hence A=0 exactly.
        state, derivative = state_and_time_derivative(model, [model.t_stop])
        root = model.root_density_matrices([model.t_stop])
        np.testing.assert_allclose(root.detach().numpy()[0], make_rho0())
        self.assert_density_matrices(root)
        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue(torch.isfinite(derivative).all())
        HEOMPINNLoss(model.hierarchy)(model, [model.t_stop]).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_gram_rescaling_avoids_overflow_for_large_finite_outputs(self):
        model = make_model()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.network[-1].bias.copy_(
                torch.tensor([1., 2., -3., 4., 5., -6., 7., 8.], dtype=torch.float64) * 1e200
            )
        root = model.root_density_matrices([0.4, 0.5, 2.4])
        self.assertTrue(torch.isfinite(root).all())
        self.assert_density_matrices(root)

    def test_invalid_initial_rdms_are_rejected(self):
        invalid = (
            (np.diag([1.1, -0.1]), "positive-semidefinite"),
            (np.diag([1., 1.]), "unit-trace"),
            (np.array([[1., 0.1], [0., 0.]]), "Hermitian"),
            (np.array([[np.nan, 0.], [0., 0.]]), "finite"),
        )
        for rho0, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                make_model(rho0)
        with self.assertRaisesRegex(TypeError, "positive_rdm_ansatz must be a boolean"):
            make_model(positive_rdm_ansatz="true")

    def test_zero_network_has_same_constant_loss_normalization(self):
        for switch in ("linear", "exponential"):
            model = make_model(time_switch=switch)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            initial = model.complex_initial_state().numpy()
            liouvillian = model.hierarchy.build_Liouvillian(normalized=True)
            scales = compute_heom_dynamical_scales(
                model.hierarchy, initial, liouvillian,
                t_start=model.t_start, t_stop=model.t_stop,
                time_switch=switch, tier_normalized=True,
                lower_tier_cutoff=0, lower_tier_weight=0.75,
            )
            model.set_correction_scale(scales.correction_scale)
            objective = HEOMPINNLoss(
                model.hierarchy, liouvillian=liouvillian,
                tier_normalized=True, lower_tier_cutoff=0, lower_tier_weight=0.75,
                normalization_loss=scales.effective_constant_loss,
            )
            times = [0.4, 0.9, 2.4]
            self.assertAlmostEqual(objective(model, times).item(), 1.0, places=12)
            np.testing.assert_allclose(
                model.complex_states(times).detach().numpy(),
                np.broadcast_to(initial, (3, initial.size)), atol=1e-14,
            )

    def test_float32_forward_and_backward_are_supported(self):
        model = make_model(mixed_root(), dtype=torch.float32)
        objective = HEOMPINNLoss(model.hierarchy, dtype=torch.float32)
        objective(model, [0.4, 0.9, 2.4]).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        root = model.root_density_matrices([0.4, 0.9, 2.4])
        self.assertGreaterEqual(torch.linalg.eigvalsh(root).min().item(), -1e-6)
        np.testing.assert_allclose(
            root.diagonal(dim1=-2, dim2=-1).sum(-1).detach().numpy(), 1., atol=1e-6
        )

    def test_roundoff_repair_is_shared_by_initial_state_and_factor(self):
        model = make_model(np.diag([1. + 1e-15, -1e-15]))
        np.testing.assert_allclose(model.complex_initial_state().numpy()[:4], [1., 0., 0., 0.])
        np.testing.assert_allclose(model.root_density_matrices([0.4]).detach().numpy()[0], make_rho0())

    def test_checkpoint_load_rebuilds_factor_from_saved_anchor(self):
        source = make_model(mixed_root())
        destination = make_model()
        destination.load_state_dict(source.state_dict())
        times = [0.4, 0.8, 2.4]
        torch.testing.assert_close(source(times), destination(times))
        self.assertFalse(any("root_sqrt" in key for key in source.state_dict()))
        invalid = {key: value.clone() for key, value in source.state_dict().items()}
        invalid["initial_state"][:4] = torch.tensor([1.1, 0., 0., -0.1], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "positive-semidefinite"):
            destination.load_state_dict(invalid)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_residual_and_double_backward_remain_finite(self):
        model = make_model(mixed_root(), device="cuda")
        loss = HEOMPINNLoss(model.hierarchy, device="cuda")(model, [0.4, 0.9])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assert_density_matrices(model.root_density_matrices([0.4, 0.9, 3.]))


def small_sequence(positive=True, switch="linear"):
    base = replace(
        MLP, device="cpu", dtype="float64", hidden_sizes=(6,),
        positive_rdm_ansatz=positive, time_switch=switch,
        epochs=2, collocation_points=6, batch_size=3,
        optimizer="adam", lbfgs_max_iter=2, lbfgs_max_eval=5, log_every=2,
    )
    return TrainingSequence(
        pseudomode=replace(PSEUDOMODE, heom_depth=2, t_start=0.0, t_stop=0.2, n_times=5),
        base_mlp=base,
        sessions=(
            TrainingSession("adam", base),
            TrainingSession("lbfgs", replace(base, optimizer="lbfgs", epochs=1)),
        ),
    )


class PositiveRDMPipelineTests(unittest.TestCase):
    def test_toml_option_is_boolean_and_base_only(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sequence.toml"
            path.write_text("[mlp]\npositive_rdm_ansatz = true\n", encoding="utf-8")
            sequence = load_training_sequence(path)
            self.assertTrue(sequence.base_mlp.positive_rdm_ansatz)
            self.assertTrue(sequence.sessions[0].mlp.positive_rdm_ansatz)
            for text in (
                "[mlp]\npositive_rdm_ansatz = 'true'\n",
                "[[sessions]]\nname = 'bad'\n[sessions.mlp]\npositive_rdm_ansatz = true\n",
            ):
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "positive_rdm_ansatz"):
                    load_training_sequence(path)
        self.assertFalse(MLP.positive_rdm_ansatz)

    def test_v8_roundtrip_and_old_versions_force_legacy_default(self):
        sequence = small_sequence()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            sidecar = save_training_metadata(sequence, path)
            self.assertEqual(load_training_metadata(path), sequence)
            current = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(current["format_version"], 8)
            for version in range(1, 8):
                document = json.loads(json.dumps(current))
                document["format_version"] = version
                if version < 7:
                    document.pop("initialize_from")
                if version < 4:
                    document.pop("name")
                values = [document["base_mlp"], *(s["mlp"] for s in document["sessions"])]
                for value in values:
                    value.pop("positive_rdm_ansatz")
                sidecar.write_text(json.dumps(document), encoding="utf-8")
                with patch("training_sequence.MLP", replace(MLP, positive_rdm_ansatz=True)):
                    migrated = load_training_metadata(path)
                self.assertFalse(migrated.base_mlp.positive_rdm_ansatz)
                self.assertTrue(all(not s.mlp.positive_rdm_ansatz for s in migrated.sessions))
            current["base_mlp"].pop("positive_rdm_ansatz")
            sidecar.write_text(json.dumps(current), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive_rdm_ansatz"):
                load_training_metadata(path)

    def test_resume_transfer_and_benchmark_reject_opposite_ansatz(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            for positive in (True, False):
                saved = small_sequence(positive)
                requested = small_sequence(not positive)
                save_training_metadata(saved, path)
                for validate in (_validate_resume_config, _validate_pretrained_config):
                    with self.assertRaisesRegex(ValueError, "positive_rdm_ansatz"):
                        validate(requested, path)
                h, initial, l = build_training_problem(saved.pseudomode)
                with self.assertRaisesRegex(ValueError, "positive_rdm_ansatz"):
                    load_mlp(h, initial, liouvillian=l, model_path=path,
                             mlp_parameters=requested.base_mlp,
                             pseudomode_parameters=requested.pseudomode)

    def test_missing_metadata_cannot_enable_positive_ansatz_on_old_weights(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            positive = small_sequence()
            for validate in (_validate_resume_config, _validate_pretrained_config):
                with self.assertRaisesRegex(ValueError, "metadata is missing"):
                    validate(positive, path)

    def test_train_save_resume_and_benchmark_with_tier_diagnostics(self):
        for switch in ("linear", "exponential"):
            with self.subTest(switch=switch), TemporaryDirectory() as directory:
                sequence = small_sequence(switch=switch)
                path = Path(directory) / "model.pt"
                with redirect_stdout(StringIO()):
                    trained, results = run_training_sequence(sequence, model_path=path)
                self.assertEqual(len(results), 2)
                self.assertTrue(all(np.isfinite(r.final.loss) for r in results))
                saved = load_training_metadata(path)
                self.assertTrue(saved.base_mlp.positive_rdm_ansatz)
                h, initial, l = build_training_problem(saved.pseudomode)
                loaded = load_mlp(
                    h, initial, liouvillian=l, model_path=path,
                    mlp_parameters=saved.base_mlp, pseudomode_parameters=saved.pseudomode,
                )
                times = np.linspace(0., 0.2, 5)
                torch.testing.assert_close(loaded(times), trained(times))
                sigma_z, solution = run_mlp_solver(loaded, times, return_solution=True)
                self.assertEqual(solution.y.shape, (h.nADO * h.system_size, 5))
                self.assertTrue(np.all(np.abs(sigma_z) <= 1. + 1e-12))
                self.assertEqual(len(compute_tier_ado_statistics(h, solution.y, solution.y)), 3)
                self.assertEqual(len(compute_tier_residual_statistics(
                    loaded, h, l, times, reference_state=solution.y, batch_size=2,
                )), 3)
                with redirect_stdout(StringIO()):
                    resumed, _ = run_training_sequence(sequence, model_path=path, resume=True)
                self.assertTrue(resumed.positive_rdm_ansatz)
                self.assertTrue(torch.isfinite(resumed(times)).all())


if __name__ == "__main__":
    unittest.main()
