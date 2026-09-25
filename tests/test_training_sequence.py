"""Configuration resolution and multi-session training orchestration tests."""

from contextlib import redirect_stderr, redirect_stdout
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
from model import EpochRecord, HEOMDynamicalScales, TrainingResult
from model.train_mlp_model import (
    _save_training_stage,
    _validate_pretrained_config,
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
    require_complete_training_checkpoint,
    resolve_model_path,
    resolve_pretrained_model_path,
    save_training_metadata,
    training_incomplete_marker_path,
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
        self.assertIsNone(sequence.name)
        self.assertIsNone(sequence.initialize_from)

    def test_optional_run_name_is_trimmed_and_selects_its_own_folder(self):
        sequence = self.load_toml('name = "  experiment_a  "\n')

        self.assertEqual(sequence.name, "experiment_a")
        self.assertEqual(
            resolve_model_path(sequence),
            MLP_MODEL_PATH.parent.parent
            / "experiment_a"
            / MLP_MODEL_PATH.name,
        )
        explicit = Path("artifacts") / "custom.pt"
        self.assertEqual(resolve_model_path(sequence, explicit), explicit)
        self.assertEqual(
            resolve_model_path(default_training_sequence()),
            MLP_MODEL_PATH,
        )

    def test_optional_pretrained_folder_is_loaded_and_trimmed(self):
        sequence = self.load_toml(
            'initialize_from = "  depth_5  "\n'
        )

        self.assertEqual(sequence.initialize_from, "depth_5")

    def test_pretrained_folder_resolves_inside_saved_models(self):
        self.assertEqual(resolve_pretrained_model_path(), MLP_MODEL_PATH)
        self.assertEqual(
            resolve_pretrained_model_path("depth_5"),
            MLP_MODEL_PATH.parent.parent
            / "depth_5"
            / MLP_MODEL_PATH.name,
        )
        for invalid in ("", "..", "nested/folder", "CON"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_pretrained_model_path(invalid)

    def test_sparse_overrides_rebase_each_session_on_base_parameters(self):
        sequence = self.load_toml(
            """
[pseudomode]
g = 0.25

[mlp]
device = "cpu"
epochs = 7
tier_normalized_loss = true
lower_tier_loss_cutoff = 5
lower_tier_loss_weight = 0.6
time_switch = "exponential"
switch_time_constant = 0.75

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

        expected_base = replace(
            MLP,
            device="cpu",
            epochs=7,
            tier_normalized_loss=True,
            lower_tier_loss_cutoff=5,
            lower_tier_loss_weight=0.6,
            time_switch="exponential",
            switch_time_constant=0.75,
        )
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

    def test_power_tier_loss_and_raw_coordinates_are_configurable(self):
        sequence = self.load_toml(
            """
[mlp]
tier_normalized_loss = true
tier_loss_power = 1.5
normalize_hierarchy_coordinates = false
"""
        )

        self.assertEqual(sequence.base_mlp.tier_loss_power, 1.5)
        self.assertFalse(
            sequence.base_mlp.normalize_hierarchy_coordinates
        )

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
            "empty run name": "name = ''\n",
            "non-string run name": "name = 3\n",
            "parent run name": "name = '..'\n",
            "nested run name": "name = '../escape'\n",
            "reserved run name": "name = 'CON'\n",
            "unsafe pretrained folder": (
                "initialize_from = '../escape'\n"
            ),
            "unknown pseudomode field": "[pseudomode]\ngamam = 1.0\n",
            "unknown mlp field": "[mlp]\nepohs = 2\n",
            "invalid optimizer": "[mlp]\noptimizer = 'sgd'\n",
            "invalid tier loss flag": (
                "[mlp]\ntier_normalized_loss = 'yes'\n"
            ),
            "incomplete lower tier weighting": (
                "[mlp]\nlower_tier_loss_cutoff = 5\n"
            ),
            "lower tier weighting without tier normalization": (
                "[mlp]\nlower_tier_loss_cutoff = 5\n"
                "lower_tier_loss_weight = 0.6\n"
                "tier_normalized_loss = false\n"
            ),
            "invalid lower tier cutoff": (
                "[mlp]\nlower_tier_loss_cutoff = -1\n"
                "lower_tier_loss_weight = 0.6\n"
            ),
            "invalid lower tier weight": (
                "[mlp]\nlower_tier_loss_cutoff = 5\n"
                "lower_tier_loss_weight = 1.1\n"
            ),
            "lower tier cutoff reaches maximum tier": (
                "[pseudomode]\nheom_depth = 5\n"
                "[mlp]\nlower_tier_loss_cutoff = 5\n"
                "lower_tier_loss_weight = 0.6\n"
            ),
            "power weighting without tier normalization": (
                "[mlp]\ntier_loss_power = 1.0\n"
                "tier_normalized_loss = false\n"
            ),
            "power and two-group weighting together": (
                "[mlp]\ntier_loss_power = 1.0\n"
                "lower_tier_loss_cutoff = 5\n"
                "lower_tier_loss_weight = 0.6\n"
            ),
            "non-finite tier loss power": (
                "[mlp]\ntier_loss_power = inf\n"
            ),
            "invalid hierarchy coordinate normalization": (
                "[mlp]\nnormalize_hierarchy_coordinates = 'no'\n"
            ),
            "invalid time switch": "[mlp]\ntime_switch = 'quadratic'\n",
            "invalid switch time constant": (
                "[mlp]\nswitch_time_constant = 0.0\n"
            ),
            "invalid numeric value": "[mlp]\nepochs = 0\n",
            "invalid value type": "[mlp]\nepochs = 'two'\n",
            "negative training start": "[pseudomode]\nt_start = -1.0\n",
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
            "session loss change": """
[[sessions]]
name = "different-loss"
[sessions.mlp]
tier_normalized_loss = true
""",
            "session time switch change": """
[[sessions]]
name = "different-switch"
[sessions.mlp]
time_switch = "exponential"
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
            constant_loss_normalization=True,
            ansatz_scale_normalization=True,
            normalization_floor=1e-9,
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
            name="roundtrip",
            initialize_from="depth_5",
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
        self.assertEqual(
            training_incomplete_marker_path("model.pt"),
            Path(".model.pt.incomplete"),
        )
        with self.assertRaises(ValueError):
            training_metadata_path("")
        with self.assertRaises(ValueError):
            training_incomplete_marker_path("")

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
            self.assertEqual(document["format_version"], 8)
            self.assertEqual(document["name"], "roundtrip")
            self.assertEqual(document["initialize_from"], "depth_5")
            self.assertIsInstance(document["base_mlp"]["hidden_sizes"], list)
            self.assertIsInstance(
                document["pseudomode"]["qutip_depths"],
                list,
            )

    def test_version_one_metadata_defaults_to_global_loss(self):
        sequence = self.make_sequence()
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            metadata_path = save_training_metadata(sequence, model_path)
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            document["format_version"] = 1
            document.pop("name")
            document.pop("initialize_from")
            document["base_mlp"].pop("tier_normalized_loss")
            document["base_mlp"].pop("time_switch")
            document["base_mlp"].pop("switch_time_constant")
            document["base_mlp"].pop("constant_loss_normalization")
            document["base_mlp"].pop("ansatz_scale_normalization")
            document["base_mlp"].pop("normalization_floor")
            document["base_mlp"].pop("lower_tier_loss_cutoff")
            document["base_mlp"].pop("lower_tier_loss_weight")
            document["base_mlp"].pop("tier_loss_power")
            document["base_mlp"].pop("normalize_hierarchy_coordinates")
            for session in document["sessions"]:
                session["mlp"].pop("tier_normalized_loss")
                session["mlp"].pop("time_switch")
                session["mlp"].pop("switch_time_constant")
                session["mlp"].pop("constant_loss_normalization")
                session["mlp"].pop("ansatz_scale_normalization")
                session["mlp"].pop("normalization_floor")
                session["mlp"].pop("lower_tier_loss_cutoff")
                session["mlp"].pop("lower_tier_loss_weight")
                session["mlp"].pop("tier_loss_power")
                session["mlp"].pop("normalize_hierarchy_coordinates")
            metadata_path.write_text(json.dumps(document), encoding="utf-8")

            loaded = load_training_metadata(model_path)

        self.assertFalse(loaded.base_mlp.tier_normalized_loss)
        self.assertEqual(loaded.base_mlp.time_switch, "linear")
        self.assertEqual(loaded.base_mlp.switch_time_constant, 1.0)
        self.assertFalse(loaded.base_mlp.constant_loss_normalization)
        self.assertFalse(loaded.base_mlp.ansatz_scale_normalization)
        self.assertEqual(
            loaded.base_mlp.normalization_floor,
            MLP.normalization_floor,
        )
        self.assertIsNone(loaded.name)
        self.assertTrue(
            all(
                not session.mlp.tier_normalized_loss
                for session in loaded.sessions
            )
        )

    def test_version_two_metadata_defaults_to_linear_switch(self):
        sequence = self.make_sequence()
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            metadata_path = save_training_metadata(sequence, model_path)
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            document["format_version"] = 2
            document.pop("name")
            document.pop("initialize_from")
            document["base_mlp"].pop("time_switch")
            document["base_mlp"].pop("switch_time_constant")
            document["base_mlp"].pop("constant_loss_normalization")
            document["base_mlp"].pop("ansatz_scale_normalization")
            document["base_mlp"].pop("normalization_floor")
            document["base_mlp"].pop("lower_tier_loss_cutoff")
            document["base_mlp"].pop("lower_tier_loss_weight")
            document["base_mlp"].pop("tier_loss_power")
            document["base_mlp"].pop("normalize_hierarchy_coordinates")
            for session in document["sessions"]:
                session["mlp"].pop("time_switch")
                session["mlp"].pop("switch_time_constant")
                session["mlp"].pop("constant_loss_normalization")
                session["mlp"].pop("ansatz_scale_normalization")
                session["mlp"].pop("normalization_floor")
                session["mlp"].pop("lower_tier_loss_cutoff")
                session["mlp"].pop("lower_tier_loss_weight")
                session["mlp"].pop("tier_loss_power")
                session["mlp"].pop("normalize_hierarchy_coordinates")
            metadata_path.write_text(json.dumps(document), encoding="utf-8")

            loaded = load_training_metadata(model_path)

        self.assertEqual(loaded.base_mlp.time_switch, "linear")
        self.assertEqual(loaded.base_mlp.switch_time_constant, 1.0)
        self.assertEqual(
            loaded.base_mlp.tier_normalized_loss,
            sequence.base_mlp.tier_normalized_loss,
        )
        self.assertFalse(loaded.base_mlp.constant_loss_normalization)
        self.assertFalse(loaded.base_mlp.ansatz_scale_normalization)
        self.assertIsNone(loaded.name)

    def test_version_three_metadata_disables_new_normalizations(self):
        sequence = self.make_sequence()
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            metadata_path = save_training_metadata(sequence, model_path)
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            document["format_version"] = 3
            document.pop("name")
            document.pop("initialize_from")
            for values in (
                document["base_mlp"],
                *(session["mlp"] for session in document["sessions"]),
            ):
                values.pop("constant_loss_normalization")
                values.pop("ansatz_scale_normalization")
                values.pop("normalization_floor")
                values.pop("lower_tier_loss_cutoff")
                values.pop("lower_tier_loss_weight")
                values.pop("tier_loss_power")
                values.pop("normalize_hierarchy_coordinates")
            metadata_path.write_text(json.dumps(document), encoding="utf-8")

            loaded = load_training_metadata(model_path)

        self.assertFalse(loaded.base_mlp.constant_loss_normalization)
        self.assertFalse(loaded.base_mlp.ansatz_scale_normalization)
        self.assertEqual(
            loaded.base_mlp.normalization_floor,
            MLP.normalization_floor,
        )
        self.assertIsNone(loaded.name)
        self.assertTrue(
            all(
                not session.mlp.constant_loss_normalization
                and not session.mlp.ansatz_scale_normalization
                for session in loaded.sessions
            )
        )

    def test_version_four_metadata_defaults_newer_tier_and_input_options(self):
        sequence = self.make_sequence()
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            metadata_path = save_training_metadata(sequence, model_path)
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            document["format_version"] = 4
            document.pop("initialize_from")
            for values in (
                document["base_mlp"],
                *(session["mlp"] for session in document["sessions"]),
            ):
                values.pop("lower_tier_loss_cutoff")
                values.pop("lower_tier_loss_weight")
                values.pop("tier_loss_power")
                values.pop("normalize_hierarchy_coordinates")
            metadata_path.write_text(json.dumps(document), encoding="utf-8")

            loaded = load_training_metadata(model_path)

        self.assertIsNone(loaded.base_mlp.lower_tier_loss_cutoff)
        self.assertIsNone(loaded.base_mlp.lower_tier_loss_weight)
        self.assertIsNone(loaded.base_mlp.tier_loss_power)
        self.assertTrue(loaded.base_mlp.normalize_hierarchy_coordinates)
        self.assertTrue(
            all(
                session.mlp.lower_tier_loss_cutoff is None
                and session.mlp.lower_tier_loss_weight is None
                for session in loaded.sessions
            )
        )

    def test_version_five_metadata_defaults_power_and_input_options(self):
        sequence = self.make_sequence()
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            metadata_path = save_training_metadata(sequence, model_path)
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            document["format_version"] = 5
            document.pop("initialize_from")
            for values in (
                document["base_mlp"],
                *(session["mlp"] for session in document["sessions"]),
            ):
                values.pop("tier_loss_power")
                values.pop("normalize_hierarchy_coordinates")
            metadata_path.write_text(json.dumps(document), encoding="utf-8")

            loaded = load_training_metadata(model_path)

        self.assertIsNone(loaded.base_mlp.tier_loss_power)
        self.assertTrue(loaded.base_mlp.normalize_hierarchy_coordinates)
        self.assertTrue(
            all(
                session.mlp.tier_loss_power is None
                and session.mlp.normalize_hierarchy_coordinates
                for session in loaded.sessions
            )
        )

    def test_version_six_metadata_defaults_pretrained_source_to_none(self):
        sequence = self.make_sequence()
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            metadata_path = save_training_metadata(sequence, model_path)
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            document["format_version"] = 6
            document.pop("initialize_from")
            metadata_path.write_text(json.dumps(document), encoding="utf-8")

            loaded = load_training_metadata(model_path)

        self.assertIsNone(loaded.initialize_from)

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

            unsupported = dict(valid, format_version=9)
            metadata_path.write_text(
                json.dumps(unsupported),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "format_version"):
                load_training_metadata(model_path)

            missing_name = dict(valid)
            del missing_name["name"]
            metadata_path.write_text(
                json.dumps(missing_name),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing key"):
                load_training_metadata(model_path)

            unsafe_name = dict(valid, name="../escape")
            metadata_path.write_text(
                json.dumps(unsafe_name),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "directory name"):
                load_training_metadata(model_path)

            unsafe_source = dict(valid, initialize_from="../escape")
            metadata_path.write_text(
                json.dumps(unsafe_source),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "directory name"):
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
        incompatible_coordinate_base = replace(
            compatible.base_mlp,
            normalize_hierarchy_coordinates=False,
        )
        incompatible_coordinates = replace(
            compatible,
            base_mlp=incompatible_coordinate_base,
            sessions=(
                TrainingSession("continue", incompatible_coordinate_base),
            ),
        )
        transfer = replace(
            compatible,
            pseudomode=replace(
                compatible.pseudomode,
                heom_depth=compatible.pseudomode.heom_depth + 1,
            ),
        )

        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"

            with self.assertRaisesRegex(ValueError, "metadata is missing"):
                _validate_resume_config(compatible, model_path)

            # Unscaled legacy checkpoints remain resumable without a sidecar.
            legacy_base = replace(
                compatible.base_mlp,
                ansatz_scale_normalization=False,
            )
            legacy = replace(
                compatible,
                base_mlp=legacy_base,
                sessions=(TrainingSession("legacy", legacy_base),),
            )
            _validate_resume_config(legacy, model_path)

            save_training_metadata(saved, model_path)
            _validate_resume_config(compatible, model_path)
            _validate_pretrained_config(transfer, model_path)
            with self.assertRaisesRegex(ValueError, "pseudomode.g"):
                _validate_resume_config(incompatible, model_path)
            with self.assertRaisesRegex(
                ValueError,
                "mlp.normalize_hierarchy_coordinates",
            ):
                _validate_resume_config(incompatible_coordinates, model_path)
            with self.assertRaisesRegex(
                ValueError,
                "mlp.normalize_hierarchy_coordinates",
            ):
                _validate_pretrained_config(
                    incompatible_coordinates,
                    model_path,
                )

            marker_path = training_incomplete_marker_path(model_path)
            marker_path.write_text("incomplete\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                _validate_resume_config(compatible, model_path)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                require_complete_training_checkpoint(model_path)

    def test_loss_floor_can_change_when_ansatz_scaling_is_disabled(self):
        saved_base = replace(
            MLP,
            ansatz_scale_normalization=False,
            normalization_floor=1e-12,
        )
        saved = TrainingSequence(
            PSEUDOMODE,
            saved_base,
            (TrainingSession("saved", saved_base),),
        )
        current_base = replace(saved_base, normalization_floor=1e-8)
        current = TrainingSequence(
            PSEUDOMODE,
            current_base,
            (TrainingSession("current", current_base),),
        )

        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            save_training_metadata(saved, model_path)
            _validate_resume_config(current, model_path)


class TrainingSequenceEntrypointTests(unittest.TestCase):
    def test_parser_tracks_explicit_optimizer_and_sequence(self):
        parser = build_argument_parser()

        defaults = parser.parse_args([])
        self.assertEqual(defaults.optimizer, MLP.optimizer)
        self.assertFalse(defaults.optimizer_explicit)
        self.assertIsNone(defaults.sequence)
        self.assertIsNone(defaults.model_path)

        optimizer = parser.parse_args(["--optimizer", "adam"])
        self.assertEqual(optimizer.optimizer, "adam")
        self.assertTrue(optimizer.optimizer_explicit)

        sequence = parser.parse_args(["--sequence", "schedule.toml"])
        self.assertEqual(sequence.sequence, Path("schedule.toml"))
        self.assertFalse(sequence.optimizer_explicit)

        alias = parser.parse_args(["--config", "schedule.toml"])
        self.assertEqual(alias.sequence, Path("schedule.toml"))

        self.assertIsNone(defaults.initialize_from)
        initialize_default = parser.parse_args(["--initialize-from"])
        self.assertEqual(initialize_default.initialize_from, "mlp")
        initialize_named = parser.parse_args(
            ["--initialize-from", "depth_5"]
        )
        self.assertEqual(initialize_named.initialize_from, "depth_5")

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

    def test_main_rejects_resume_with_pretrained_initialization(self):
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["--resume", "--initialize-from"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("cannot be combined", stderr.getvalue())

    def test_main_uses_pretrained_folder_from_sequence_file(self):
        sequence = replace(
            default_training_sequence(),
            initialize_from="depth_5",
        )
        trained_model = object()
        final_result = object()
        with TemporaryDirectory() as directory:
            pretrained_path = Path(directory) / "mlp_state_dict.pt"
            pretrained_path.write_bytes(b"checkpoint")
            with (
                patch(
                    "model.train_mlp_model.load_training_sequence",
                    return_value=sequence,
                ),
                patch(
                    "model.train_mlp_model.resolve_pretrained_model_path",
                    return_value=pretrained_path,
                ) as resolve_pretrained,
                patch(
                    "model.train_mlp_model._validate_pretrained_config"
                ) as validate_pretrained,
                patch(
                    "model.train_mlp_model.run_training_sequence",
                    return_value=(trained_model, (final_result,)),
                ) as run,
            ):
                actual_model, actual_result = main(
                    ["--sequence", "schedule.toml"]
                )

        self.assertIs(actual_model, trained_model)
        self.assertIs(actual_result, final_result)
        resolve_pretrained.assert_called_once_with("depth_5")
        validate_pretrained.assert_called_once_with(
            sequence,
            pretrained_path,
        )
        run.assert_called_once_with(
            sequence,
            resume=False,
            plot_loss=False,
            model_path=None,
            pretrained_model_path=pretrained_path,
        )

    def test_command_line_pretrained_folder_overrides_sequence_file(self):
        sequence = replace(
            default_training_sequence(),
            initialize_from="depth_5",
        )
        with TemporaryDirectory() as directory:
            pretrained_path = Path(directory) / "mlp_state_dict.pt"
            pretrained_path.write_bytes(b"checkpoint")
            with (
                patch(
                    "model.train_mlp_model.load_training_sequence",
                    return_value=sequence,
                ),
                patch(
                    "model.train_mlp_model.resolve_pretrained_model_path",
                    return_value=pretrained_path,
                ) as resolve_pretrained,
                patch(
                    "model.train_mlp_model._validate_pretrained_config"
                ),
                patch(
                    "model.train_mlp_model.run_training_sequence",
                    return_value=(object(), (object(),)),
                ) as run,
            ):
                main(
                    [
                        "--sequence",
                        "schedule.toml",
                        "--initialize-from",
                        "depth_10",
                    ]
                )

        overridden_sequence = run.call_args.args[0]
        self.assertEqual(overridden_sequence.initialize_from, "depth_10")
        resolve_pretrained.assert_called_once_with("depth_10")

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
        base_mlp = replace(
            MLP,
            device="cpu",
            optimizer="adam",
            epochs=2,
            constant_loss_normalization=True,
            ansatz_scale_normalization=True,
            lower_tier_loss_cutoff=0,
            lower_tier_loss_weight=0.75,
            normalize_hierarchy_coordinates=False,
        )
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
        initial_heom_state = object()
        liouvillian = object()
        model = Mock(name="model")
        model.state_dict.return_value = {"weight": "state"}
        anchored_state = object()
        complex_state = model.complex_initial_state.return_value
        cpu_state = complex_state.detach.return_value.cpu.return_value
        cpu_state.numpy.return_value = anchored_state
        scales = HEOMDynamicalScales(
            constant_loss=2.0,
            effective_constant_loss=2.0,
            global_constant_loss=3.0,
            effective_global_constant_loss=3.0,
            initial_switch_slope=0.1,
            correction_scale=4.0,
        )
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
            save_training_metadata(sequence, model_path)
            with (
                patch(
                    "model.train_mlp_model.build_training_problem",
                    return_value=(
                        hierarchy,
                        initial_heom_state,
                        liouvillian,
                    ),
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
                    "model.train_mlp_model.compute_heom_dynamical_scales",
                    return_value=scales,
                ) as compute_scales,
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
                self.assertTrue(
                    (model_path.parent / "training.log").is_file()
                )
                self.assertFalse(
                    training_incomplete_marker_path(model_path).exists()
                )

        self.assertIs(actual_model, model)
        self.assertEqual(results, (first_result, second_result))
        build_problem.assert_called_once_with(sequence.pseudomode)
        model_type.assert_called_once_with(
            hierarchy,
            hidden_sizes=base_mlp.hidden_sizes,
            initial_heom_state=initial_heom_state,
            t_start=sequence.pseudomode.t_start,
            t_stop=sequence.pseudomode.t_stop,
            activation=base_mlp.activation,
            time_switch=base_mlp.time_switch,
            switch_time_constant=base_mlp.switch_time_constant,
            normalize_hierarchy_coordinates=(
                base_mlp.normalize_hierarchy_coordinates
            ),
            positive_rdm_ansatz=base_mlp.positive_rdm_ansatz,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        objective_type.assert_called_once_with(
            hierarchy,
            liouvillian=liouvillian,
            tier_normalized=base_mlp.tier_normalized_loss,
            lower_tier_cutoff=base_mlp.lower_tier_loss_cutoff,
            lower_tier_weight=base_mlp.lower_tier_loss_weight,
            tier_loss_power=base_mlp.tier_loss_power,
            normalization_loss=scales.effective_constant_loss,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        compute_scales.assert_called_once_with(
            hierarchy,
            anchored_state,
            liouvillian,
            t_start=sequence.pseudomode.t_start,
            t_stop=sequence.pseudomode.t_stop,
            tier_normalized=base_mlp.tier_normalized_loss,
            lower_tier_cutoff=base_mlp.lower_tier_loss_cutoff,
            lower_tier_weight=base_mlp.lower_tier_loss_weight,
            tier_loss_power=base_mlp.tier_loss_power,
            time_switch=base_mlp.time_switch,
            switch_time_constant=base_mlp.switch_time_constant,
            normalization_floor=base_mlp.normalization_floor,
        )
        model.set_correction_scale.assert_called_once_with(
            scales.correction_scale
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

    def test_failed_metadata_update_leaves_an_incomplete_marker(self):
        sequence = default_training_sequence(
            mlp_defaults=replace(MLP, device="cpu")
        )
        model = Mock()

        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "run" / "model.pt"
            model_path.parent.mkdir(parents=True)
            with (
                patch("model.train_mlp_model.save_model") as save_model,
                patch(
                    "model.train_mlp_model.save_training_metadata",
                    side_effect=OSError("metadata write failed"),
                ),
                self.assertRaisesRegex(OSError, "metadata write failed"),
            ):
                _save_training_stage(model, sequence, model_path)

            save_model.assert_called_once_with(model, model_path)
            marker_path = training_incomplete_marker_path(model_path)
            self.assertTrue(marker_path.is_file())
            with self.assertRaisesRegex(ValueError, "incomplete"):
                require_complete_training_checkpoint(model_path)

    def test_training_log_is_replaced_for_fresh_runs_and_appended_on_resume(self):
        sequence = default_training_sequence(
            mlp_defaults=replace(MLP, device="cpu")
        )
        returned = (object(), ())

        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "run" / "model.pt"
            log_path = model_path.parent / "training.log"
            model_path.parent.mkdir(parents=True)
            log_path.write_text("stale record\n", encoding="utf-8")

            def fresh_body(*args, **kwargs):
                del args, kwargs
                print("Epoch fresh: loss=1.0")
                return returned

            with (
                patch(
                    "model.train_mlp_model._run_training_sequence_body",
                    side_effect=fresh_body,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(
                    run_training_sequence(sequence, model_path=model_path),
                    returned,
                )

            fresh_log = log_path.read_text(encoding="utf-8")
            self.assertNotIn("stale record", fresh_log)
            self.assertIn("Epoch fresh: loss=1.0", fresh_log)
            save_training_metadata(sequence, model_path)

            def resumed_body(*args, **kwargs):
                del args, kwargs
                print("Epoch resumed: loss=0.5")
                return returned

            with (
                patch(
                    "model.train_mlp_model._run_training_sequence_body",
                    side_effect=resumed_body,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(
                    run_training_sequence(
                        sequence,
                        resume=True,
                        model_path=model_path,
                    ),
                    returned,
                )

            resumed_log = log_path.read_text(encoding="utf-8")
            self.assertIn("Epoch fresh: loss=1.0", resumed_log)
            self.assertIn("Epoch resumed: loss=0.5", resumed_log)


if __name__ == "__main__":
    unittest.main()
