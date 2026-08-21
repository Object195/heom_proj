"""Configuration resolution and multi-session training orchestration tests."""

from contextlib import redirect_stderr
from dataclasses import FrozenInstanceError, replace
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, call, patch

import torch

from experiment_parameters import (
    MLP,
    MLP_MODEL_PATH,
    PSEUDOMODE,
    MLPParameters,
    PseudomodeParameters,
)
from model import EpochRecord, TrainingResult
from model.train_mlp_model import (
    _validate_resume_config,
    build_argument_parser,
    build_training_config,
    main,
    run_training_sequence,
)
from training_sequence import (
    TrainingSequence,
    TrainingSession,
    default_training_sequence,
    load_training_metadata,
    load_training_sequence,
    save_training_metadata,
    training_metadata_path,
)


class TrainingSequenceResolutionTests(unittest.TestCase):
    def load_toml(self, contents: str) -> TrainingSequence:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sequence.toml"
            path.write_text(contents, encoding="utf-8")
            return load_training_sequence(path)

    def test_default_sequence_matches_experiment_parameters(self):
        sequence = default_training_sequence()

        self.assertEqual(sequence.pseudomode, PSEUDOMODE)
        self.assertEqual(sequence.base_mlp, MLP)
        self.assertIsInstance(sequence.sessions, tuple)
        self.assertEqual(len(sequence.sessions), 1)
        self.assertEqual(sequence.sessions[0].mlp, MLP)

    def test_sparse_overrides_rebase_each_session_on_base_parameters(self):
        sequence = self.load_toml(
            """
[pseudomode]
g = 0.25

[mlp]
device = "cpu"
epochs = 7

[[sessions]]
name = "warmup"
[sessions.mlp]
optimizer = "adam"
learning_rate = 0.02

[[sessions]]
name = "polish"
[sessions.mlp]
epochs = 2
"""
        )

        expected_base = replace(MLP, device="cpu", epochs=7)
        self.assertEqual(sequence.pseudomode, replace(PSEUDOMODE, g=0.25))
        self.assertEqual(sequence.base_mlp, expected_base)
        self.assertEqual(
            sequence.sessions,
            (
                TrainingSession(
                    "warmup",
                    replace(
                        expected_base,
                        optimizer="adam",
                        learning_rate=0.02,
                    ),
                ),
                TrainingSession(
                    "polish",
                    replace(expected_base, epochs=2),
                ),
            ),
        )
        self.assertEqual(
            sequence.sessions[1].mlp.learning_rate,
            expected_base.learning_rate,
        )

    def test_toml_arrays_are_normalized_to_immutable_tuples(self):
        sequence = self.load_toml(
            """
[pseudomode]
qutip_depths = [2, 4]

[mlp]
hidden_sizes = [8, 4]
"""
        )

        self.assertEqual(sequence.pseudomode.qutip_depths, (2, 4))
        self.assertIsInstance(sequence.pseudomode.qutip_depths, tuple)
        self.assertEqual(sequence.base_mlp.hidden_sizes, (8, 4))
        self.assertIsInstance(sequence.base_mlp.hidden_sizes, tuple)
        self.assertEqual(sequence.sessions[0].mlp.hidden_sizes, (8, 4))
        with self.assertRaises(FrozenInstanceError):
            sequence.base_mlp.epochs = 10

    def test_base_only_file_creates_one_session_from_resolved_base(self):
        sequence = self.load_toml(
            """
[pseudomode]
heom_depth = 3

[mlp]
optimizer = "adam"
epochs = 4
"""
        )

        self.assertEqual(sequence.pseudomode.heom_depth, 3)
        self.assertEqual(len(sequence.sessions), 1)
        self.assertEqual(sequence.sessions[0].mlp, sequence.base_mlp)
        self.assertEqual(sequence.base_mlp.optimizer, "adam")
        self.assertEqual(sequence.base_mlp.epochs, 4)

    def test_loading_does_not_mutate_module_defaults(self):
        self.load_toml(
            """
[pseudomode]
g = 0.5

[mlp]
epochs = 3
"""
        )

        self.assertEqual(PSEUDOMODE, PseudomodeParameters())
        self.assertEqual(MLP, MLPParameters())

    def test_lbfgs_session_requires_float64_base_dtype(self):
        float32_base = replace(
            MLP,
            dtype="float32",
            optimizer="adam",
        )
        with self.assertRaisesRegex(ValueError, "float64"):
            TrainingSequence(
                pseudomode=PSEUDOMODE,
                base_mlp=float32_base,
                sessions=(
                    TrainingSession(
                        "polish",
                        replace(float32_base, optimizer="lbfgs"),
                    ),
                ),
            )

        adam_only = TrainingSequence(
            pseudomode=PSEUDOMODE,
            base_mlp=float32_base,
            sessions=(TrainingSession("warmup", float32_base),),
        )
        self.assertEqual(adam_only.base_mlp.dtype, "float32")

    def test_invalid_files_are_rejected_during_resolution(self):
        invalid_documents = {
            "unknown top-level table": "[unexpected]\nvalue = 1\n",
            "unknown pseudomode field": "[pseudomode]\ngamam = 1.0\n",
            "unknown mlp field": "[mlp]\nepohs = 2\n",
            "invalid optimizer": "[mlp]\noptimizer = 'sgd'\n",
            "invalid numeric value": "[mlp]\nepochs = 0\n",
            "invalid value type": "[mlp]\nepochs = 'two'\n",
            "missing session name": """
[[sessions]]
[sessions.mlp]
epochs = 2
""",
            "empty session name": """
[[sessions]]
name = ""
[sessions.mlp]
epochs = 2
""",
            "empty explicit sequence": "sessions = []\n",
            "duplicate session names": """
[[sessions]]
name = "repeat"

[[sessions]]
name = "repeat"
""",
            "session architecture change": """
[[sessions]]
name = "resize"
[sessions.mlp]
hidden_sizes = [4]
""",
            "session pseudomode change": """
[[sessions]]
name = "different-physics"
[sessions.pseudomode]
g = 0.2
""",
        }

        for label, document in invalid_documents.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self.load_toml(document)


class TrainingMetadataTests(unittest.TestCase):
    @staticmethod
    def make_sequence() -> TrainingSequence:
        base_mlp = replace(
            MLP,
            hidden_sizes=(8, 4),
            device="cpu",
            optimizer="adam",
            epochs=3,
        )
        return TrainingSequence(
            pseudomode=replace(
                PSEUDOMODE,
                g=0.25,
                qutip_depths=(2, 4),
            ),
            base_mlp=base_mlp,
            sessions=(
                TrainingSession(
                    "warmup",
                    replace(base_mlp, learning_rate=0.02),
                ),
                TrainingSession(
                    "polish",
                    replace(base_mlp, optimizer="lbfgs", epochs=1),
                ),
            ),
        )

    def test_metadata_path_is_adjacent_to_checkpoint(self):
        self.assertEqual(
            training_metadata_path(Path("models") / "state.pt"),
            Path("models") / "state.pt.config.json",
        )
        self.assertEqual(
            training_metadata_path("model.pt"),
            Path("model.pt.config.json"),
        )
        with self.assertRaises(ValueError):
            training_metadata_path("")

    def test_metadata_save_and_load_round_trip(self):
        sequence = self.make_sequence()
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "nested" / "model.pt"
            metadata_path = save_training_metadata(sequence, model_path)

            self.assertEqual(
                metadata_path,
                training_metadata_path(model_path),
            )
            self.assertTrue(metadata_path.is_file())
            self.assertEqual(load_training_metadata(model_path), sequence)
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(document["format_version"], 1)
            self.assertIsInstance(document["base_mlp"]["hidden_sizes"], list)
            self.assertIsInstance(
                document["pseudomode"]["qutip_depths"],
                list,
            )

    def test_metadata_save_failure_preserves_previous_sidecar(self):
        sequence = self.make_sequence()
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            metadata_path = training_metadata_path(model_path)
            metadata_path.write_text("previous metadata\n", encoding="utf-8")

            def fail_after_partial_write(document, stream, **kwargs):
                del document, kwargs
                stream.write('{"partial":')
                raise OSError("simulated write failure")

            with (
                patch(
                    "training_sequence.json.dump",
                    side_effect=fail_after_partial_write,
                ),
                self.assertRaisesRegex(OSError, "simulated write failure"),
            ):
                save_training_metadata(sequence, model_path)

            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"),
                "previous metadata\n",
            )
            self.assertEqual(
                list(
                    metadata_path.parent.glob(
                        f".{metadata_path.name}.*.tmp"
                    )
                ),
                [],
            )

    def test_invalid_metadata_and_version_are_rejected(self):
        sequence = self.make_sequence()
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            metadata_path = save_training_metadata(sequence, model_path)
            valid = json.loads(metadata_path.read_text(encoding="utf-8"))

            metadata_path.write_text("{not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid.*JSON"):
                load_training_metadata(model_path)

            unsupported = dict(valid, format_version=2)
            metadata_path.write_text(
                json.dumps(unsupported),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "format_version"):
                load_training_metadata(model_path)

            missing_sessions = dict(valid)
            del missing_sessions["sessions"]
            metadata_path.write_text(
                json.dumps(missing_sessions),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing key"):
                load_training_metadata(model_path)

    def test_resume_sidecar_checks_only_checkpoint_compatibility_fields(self):
        saved = self.make_sequence()
        compatible_base = replace(
            saved.base_mlp,
            optimizer="lbfgs",
            epochs=20,
            learning_rate=0.5,
        )
        compatible = TrainingSequence(
            pseudomode=saved.pseudomode,
            base_mlp=compatible_base,
            sessions=(TrainingSession("continue", compatible_base),),
        )
        incompatible = replace(
            compatible,
            pseudomode=replace(compatible.pseudomode, g=0.5),
        )

        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"

            # Legacy checkpoints without a sidecar remain resumable.
            _validate_resume_config(compatible, model_path)

            save_training_metadata(saved, model_path)
            _validate_resume_config(compatible, model_path)
            with self.assertRaisesRegex(ValueError, "pseudomode.g"):
                _validate_resume_config(incompatible, model_path)


class TrainingSequenceEntrypointTests(unittest.TestCase):
    def test_parser_tracks_explicit_optimizer_and_sequence(self):
        parser = build_argument_parser()

        defaults = parser.parse_args([])
        self.assertEqual(defaults.optimizer, MLP.optimizer)
        self.assertFalse(defaults.optimizer_explicit)
        self.assertIsNone(defaults.sequence)
        self.assertEqual(defaults.model_path, MLP_MODEL_PATH)

        optimizer = parser.parse_args(["--optimizer", "adam"])
        self.assertEqual(optimizer.optimizer, "adam")
        self.assertTrue(optimizer.optimizer_explicit)

        sequence = parser.parse_args(["--sequence", "schedule.toml"])
        self.assertEqual(sequence.sequence, Path("schedule.toml"))
        self.assertFalse(sequence.optimizer_explicit)

        alias = parser.parse_args(["--config", "schedule.toml"])
        self.assertEqual(alias.sequence, Path("schedule.toml"))

        checkpoint = parser.parse_args(
            ["--model-path", "artifacts/custom.pt"]
        )
        self.assertEqual(
            checkpoint.model_path,
            Path("artifacts/custom.pt"),
        )

    def test_main_rejects_explicit_optimizer_with_sequence(self):
        stderr = StringIO()
        with (
            patch("model.train_mlp_model.run_training_sequence") as run,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            main(
                [
                    "--sequence",
                    "schedule.toml",
                    "--optimizer",
                    "adam",
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("cannot be combined", stderr.getvalue())
        run.assert_not_called()

    def test_training_config_applies_optimizer_specific_batch_behavior(self):
        pseudomode = replace(PSEUDOMODE, t_start=1.0, t_stop=3.0)
        adam = replace(
            MLP,
            optimizer="adam",
            epochs=3,
            collocation_points=12,
            batch_size=4,
            resample_each_epoch=True,
        )
        lbfgs = replace(adam, optimizer="lbfgs")

        adam_config = build_training_config(pseudomode, adam)
        lbfgs_config = build_training_config(pseudomode, lbfgs)

        self.assertEqual(adam_config.t_start, 1.0)
        self.assertEqual(adam_config.t_stop, 3.0)
        self.assertEqual(adam_config.batch_size, 4)
        self.assertTrue(adam_config.resample_each_epoch)
        self.assertEqual(lbfgs_config.batch_size, 12)
        self.assertFalse(lbfgs_config.resample_each_epoch)

    def test_runner_reuses_model_and_rebuilds_stage_training_objects(self):
        base_mlp = replace(MLP, device="cpu", optimizer="adam", epochs=2)
        warmup = TrainingSession(
            "warmup",
            replace(base_mlp, learning_rate=0.02),
        )
        polish = TrainingSession(
            "polish",
            replace(base_mlp, optimizer="lbfgs", epochs=1),
        )
        sequence = TrainingSequence(
            pseudomode=replace(PSEUDOMODE, heom_depth=1),
            base_mlp=base_mlp,
            sessions=(warmup, polish),
        )
        hierarchy = object()
        rho0 = object()
        liouvillian = object()
        model = Mock(name="model")
        model.state_dict.return_value = {"weight": "state"}
        objective = object()
        adam_optimizer = object()
        lbfgs_optimizer = object()
        first_result = TrainingResult(
            history=(EpochRecord(epoch=2, loss=2.0),),
            elapsed_seconds=0.1,
        )
        second_result = TrainingResult(
            history=(EpochRecord(epoch=1, loss=1.0),),
            elapsed_seconds=0.2,
        )

        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "checkpoints" / "model.pt"
            with (
                patch(
                    "model.train_mlp_model.build_training_problem",
                    return_value=(hierarchy, rho0, liouvillian),
                ) as build_problem,
                patch(
                    "model.train_mlp_model.HEOMMLP",
                    return_value=model,
                ) as model_type,
                patch(
                    "model.train_mlp_model.HEOMPINNLoss",
                    return_value=objective,
                ) as objective_type,
                patch(
                    "model.train_mlp_model.load_saved_model"
                ) as load_saved,
                patch(
                    "model.train_mlp_model.build_optimizer",
                    side_effect=(adam_optimizer, lbfgs_optimizer),
                ) as build_optimizer,
                patch(
                    "model.train_mlp_model.train_mlp",
                    side_effect=(first_result, second_result),
                ) as train,
                patch("model.train_mlp_model.save_model") as save_model,
                patch(
                    "model.train_mlp_model.save_training_metadata",
                    return_value=training_metadata_path(model_path),
                ) as save_metadata,
                patch("builtins.print"),
            ):
                actual_model, results = run_training_sequence(
                    sequence,
                    resume=True,
                    model_path=model_path,
                )

        self.assertIs(actual_model, model)
        self.assertEqual(results, (first_result, second_result))
        build_problem.assert_called_once_with(sequence.pseudomode)
        model_type.assert_called_once_with(
            hierarchy,
            hidden_sizes=base_mlp.hidden_sizes,
            rho0=rho0,
            t_start=sequence.pseudomode.t_start,
            t_stop=sequence.pseudomode.t_stop,
            activation=base_mlp.activation,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        objective_type.assert_called_once_with(
            hierarchy,
            liouvillian=liouvillian,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        load_saved.assert_called_once_with(
            model,
            model_path,
            torch.device("cpu"),
        )
        self.assertEqual(
            build_optimizer.call_args_list,
            [
                call(model, "adam", warmup.mlp),
                call(model, "lbfgs", polish.mlp),
            ],
        )
        self.assertEqual(train.call_count, 2)
        for invocation in train.call_args_list:
            self.assertIs(invocation.args[0], model)
            self.assertIs(invocation.args[1], objective)
        self.assertEqual(
            train.call_args_list[0].args[2],
            build_training_config(sequence.pseudomode, warmup.mlp),
        )
        self.assertEqual(
            train.call_args_list[1].args[2],
            build_training_config(sequence.pseudomode, polish.mlp),
        )
        self.assertEqual(
            save_model.call_args_list,
            [
                call(model, model_path),
                call(model, model_path),
            ],
        )
        self.assertEqual(
            save_metadata.call_args_list,
            [
                call(sequence, model_path),
                call(sequence, model_path),
            ],
        )


if __name__ == "__main__":
    unittest.main()
