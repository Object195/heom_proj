"""Load layered, immutable MLP training sequences from TOML.

``experiment_parameters`` remains the source of defaults.  A TOML file may
override a sparse subset of those defaults once for the experiment and then
define named training sessions whose MLP overrides are independently layered
on the same resolved base configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
import tomllib

from experiment_parameters import (
    MLP,
    MLP_MODEL_PATH,
    PSEUDOMODE,
    MLPParameters,
    PseudomodeParameters,
)


_TOP_LEVEL_KEYS = frozenset(
    {"name", "initialize_from", "pseudomode", "mlp", "sessions"}
)
_SESSION_KEYS = frozenset({"name", "mlp"})
_LEGACY_METADATA_KEYS = frozenset(
    {"format_version", "pseudomode", "base_mlp", "sessions"}
)
_NAMED_METADATA_KEYS = _LEGACY_METADATA_KEYS | {"name"}
_METADATA_KEYS = _NAMED_METADATA_KEYS | {"initialize_from"}
_PSEUDOMODE_FIELDS = frozenset(
    field.name for field in fields(PseudomodeParameters)
)
_MLP_FIELDS = frozenset(field.name for field in fields(MLPParameters))
_SESSION_MLP_FIELDS = frozenset(
    {
        "epochs",
        "collocation_points",
        "batch_size",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "lbfgs_learning_rate",
        "lbfgs_max_iter",
        "lbfgs_max_eval",
        "lbfgs_history_size",
        "lbfgs_tolerance_grad",
        "lbfgs_tolerance_change",
        "lbfgs_line_search",
        "gradient_clip_norm",
        "resample_each_epoch",
        "seed",
        "log_every",
    }
)
_BASE_ONLY_MLP_FIELDS = _MLP_FIELDS.difference(_SESSION_MLP_FIELDS)


@dataclass(frozen=True)
class TrainingSession:
    """One named training stage with fully resolved MLP parameters."""

    name: str
    mlp: MLPParameters

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("training session names must be non-empty strings")
        if not isinstance(self.mlp, MLPParameters):
            raise TypeError("TrainingSession.mlp must be an MLPParameters object")


@dataclass(frozen=True)
class TrainingSequence:
    """Resolved experiment parameters and its ordered training sessions."""

    pseudomode: PseudomodeParameters
    base_mlp: MLPParameters
    sessions: tuple[TrainingSession, ...]
    name: str | None = None
    initialize_from: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _run_name(self.name, "name"))
        object.__setattr__(
            self,
            "initialize_from",
            _run_name(self.initialize_from, "initialize_from"),
        )
        if not isinstance(self.pseudomode, PseudomodeParameters):
            raise TypeError(
                "TrainingSequence.pseudomode must be a "
                "PseudomodeParameters object"
            )
        if not isinstance(self.base_mlp, MLPParameters):
            raise TypeError(
                "TrainingSequence.base_mlp must be an MLPParameters object"
            )
        if not isinstance(self.sessions, tuple) or not self.sessions:
            raise ValueError(
                "a training sequence must contain at least one session"
            )
        if not all(
            isinstance(session, TrainingSession) for session in self.sessions
        ):
            raise TypeError(
                "TrainingSequence.sessions must contain TrainingSession objects"
            )
        names = tuple(session.name for session in self.sessions)
        if len(set(names)) != len(names):
            raise ValueError("training session names must be unique")

        _validate_pseudomode(self.pseudomode, "pseudomode")
        _validate_mlp(self.base_mlp, "base_mlp")
        if (
            self.base_mlp.lower_tier_loss_cutoff is not None
            and self.base_mlp.lower_tier_loss_cutoff
            >= self.pseudomode.heom_depth
        ):
            raise ValueError(
                "base_mlp.lower_tier_loss_cutoff must be less than "
                "pseudomode.heom_depth"
            )
        for session in self.sessions:
            _validate_mlp(session.mlp, f"session {session.name!r}.mlp")
            changed = sorted(
                name
                for name in _BASE_ONLY_MLP_FIELDS
                if getattr(session.mlp, name) != getattr(self.base_mlp, name)
            )
            if changed:
                rendered = ", ".join(repr(name) for name in changed)
                raise ValueError(
                    f"session {session.name!r} changes base-only MLP "
                    f"field(s): {rendered}"
                )
        if (
            any(session.mlp.optimizer == "lbfgs" for session in self.sessions)
            and self.base_mlp.dtype != "float64"
        ):
            raise ValueError(
                "base_mlp.dtype must be 'float64' when any training session "
                "uses L-BFGS"
            )


def _unknown_keys(
    values: Mapping[str, object],
    allowed: frozenset[str],
    location: str,
) -> None:
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        rendered = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"unknown key(s) in {location}: {rendered}")


def _exact_keys(
    values: Mapping[str, object],
    required: frozenset[str],
    location: str,
) -> None:
    _unknown_keys(values, required, location)
    missing = sorted(required.difference(values))
    if missing:
        rendered = ", ".join(repr(key) for key in missing)
        raise ValueError(f"missing key(s) in {location}: {rendered}")


def _table(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a TOML table")
    return value


def _json_object(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a JSON object")
    return value


_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_WINDOWS_INVALID_NAME_CHARACTERS = frozenset('<>:"/\\|?*')


def _run_name(value: object, location: str) -> str | None:
    """Return a safe optional single-directory run name."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string or omitted")
    name = value.strip()
    if name in {".", ".."} or name[-1] in {".", " "}:
        raise ValueError(f"{location} must be a safe directory name")
    if any(
        character in _WINDOWS_INVALID_NAME_CHARACTERS
        or ord(character) < 32
        for character in name
    ):
        raise ValueError(
            f"{location} must be one directory name without path separators"
        )
    device_stem = name.split(".", 1)[0].upper()
    if device_stem in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"{location} uses a reserved Windows device name")
    return name


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{location} must be a finite number")
    return numeric


def _positive_number(value: object, location: str) -> None:
    if _finite_number(value, location) <= 0.0:
        raise ValueError(f"{location} must be greater than zero")


def _nonnegative_number(value: object, location: str) -> None:
    if _finite_number(value, location) < 0.0:
        raise ValueError(f"{location} must be non-negative")


def _integer(value: object, location: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{location} must be {qualifier}")


def _normalize_tuple_fields(
    defaults: object,
    overrides: Mapping[str, object],
    location: str,
    *,
    array_description: str = "TOML array",
) -> dict[str, object]:
    normalized = dict(overrides)
    tuple_fields = {
        field.name
        for field in fields(defaults)
        if isinstance(getattr(defaults, field.name), tuple)
    }
    for name in tuple_fields.intersection(normalized):
        value = normalized[name]
        if not isinstance(value, list):
            raise ValueError(
                f"{location}.{name} must be a {array_description}"
            )
        normalized[name] = tuple(value)
    return normalized


def _validate_pseudomode(
    parameters: PseudomodeParameters,
    location: str,
) -> None:
    for name in ("w0", "delta", "v", "g", "gamma", "t_start", "t_stop"):
        _finite_number(getattr(parameters, name), f"{location}.{name}")
    if parameters.gamma < 0:
        raise ValueError(f"{location}.gamma must be non-negative")
    _integer(
        parameters.cavity_dimension,
        f"{location}.cavity_dimension",
    )
    _integer(parameters.heom_depth, f"{location}.heom_depth")
    if (
        not isinstance(parameters.qutip_depths, tuple)
        or not parameters.qutip_depths
    ):
        raise ValueError(f"{location}.qutip_depths must be a non-empty tuple")
    for index, depth in enumerate(parameters.qutip_depths):
        _integer(depth, f"{location}.qutip_depths[{index}]")
    if parameters.t_stop <= parameters.t_start:
        raise ValueError(f"{location}.t_stop must be greater than t_start")
    if parameters.t_start < 0.0:
        raise ValueError(f"{location}.t_start must be non-negative")
    _integer(parameters.n_times, f"{location}.n_times", minimum=2)
    _positive_number(parameters.rtol, f"{location}.rtol")
    _positive_number(parameters.atol, f"{location}.atol")


def _validate_mlp(parameters: MLPParameters, location: str) -> None:
    if (
        not isinstance(parameters.hidden_sizes, tuple)
        or not parameters.hidden_sizes
    ):
        raise ValueError(f"{location}.hidden_sizes must be a non-empty tuple")
    for index, width in enumerate(parameters.hidden_sizes):
        _integer(width, f"{location}.hidden_sizes[{index}]")

    if not isinstance(parameters.activation, str) or parameters.activation not in {
        "tanh",
        "gelu",
        "silu",
        "relu",
    }:
        raise ValueError(
            f"{location}.activation must be one of tanh, gelu, silu, or relu"
        )
    if not isinstance(parameters.dtype, str) or parameters.dtype not in {
        "float32",
        "float64",
    }:
        raise ValueError(f"{location}.dtype must be float32 or float64")
    if not isinstance(parameters.device, str) or not parameters.device.strip():
        raise ValueError(f"{location}.device must be a non-empty string")
    if not isinstance(parameters.tier_normalized_loss, bool):
        raise ValueError(
            f"{location}.tier_normalized_loss must be a boolean"
        )
    cutoff = parameters.lower_tier_loss_cutoff
    weight = parameters.lower_tier_loss_weight
    power = parameters.tier_loss_power
    if (cutoff is None) != (weight is None):
        raise ValueError(
            f"{location}.lower_tier_loss_cutoff and "
            "lower_tier_loss_weight must be set together"
        )
    if cutoff is not None and power is not None:
        raise ValueError(
            f"{location} cannot combine lower-tier two-group weighting "
            "with tier_loss_power"
        )
    if power is not None:
        if not parameters.tier_normalized_loss:
            raise ValueError(
                f"{location}.tier_loss_power requires "
                "tier_normalized_loss = true"
            )
        _finite_number(power, f"{location}.tier_loss_power")
    if cutoff is not None:
        if not parameters.tier_normalized_loss:
            raise ValueError(
                f"{location}.lower-tier weighting requires "
                "tier_normalized_loss = true"
            )
        _integer(
            cutoff,
            f"{location}.lower_tier_loss_cutoff",
            minimum=0,
        )
        weight = _finite_number(
            weight,
            f"{location}.lower_tier_loss_weight",
        )
        if not 0.0 <= weight <= 1.0:
            raise ValueError(
                f"{location}.lower_tier_loss_weight must be between 0 and 1"
            )
    for name in (
        "normalize_hierarchy_coordinates",
        "positive_rdm_ansatz",
        "constant_loss_normalization",
        "ansatz_scale_normalization",
    ):
        if not isinstance(getattr(parameters, name), bool):
            raise ValueError(f"{location}.{name} must be a boolean")
    _positive_number(
        parameters.normalization_floor,
        f"{location}.normalization_floor",
    )
    if (
        not isinstance(parameters.time_switch, str)
        or parameters.time_switch not in {"linear", "exponential"}
    ):
        raise ValueError(
            f"{location}.time_switch must be linear or exponential"
        )
    _positive_number(
        parameters.switch_time_constant,
        f"{location}.switch_time_constant",
    )
    if not isinstance(parameters.optimizer, str) or parameters.optimizer not in {
        "adam",
        "lbfgs",
    }:
        raise ValueError(f"{location}.optimizer must be adam or lbfgs")
    if parameters.lbfgs_line_search != "strong_wolfe":
        raise ValueError(
            f"{location}.lbfgs_line_search must be strong_wolfe"
        )

    for name in (
        "epochs",
        "collocation_points",
        "batch_size",
        "lbfgs_max_iter",
        "lbfgs_max_eval",
        "lbfgs_history_size",
        "log_every",
        "inference_batch_size",
    ):
        _integer(getattr(parameters, name), f"{location}.{name}")
    _integer(parameters.seed, f"{location}.seed", minimum=0)
    if parameters.seed > 2**63 - 1:
        raise ValueError(f"{location}.seed must not exceed 2**63 - 1")

    for name in ("learning_rate", "lbfgs_learning_rate"):
        _positive_number(getattr(parameters, name), f"{location}.{name}")
    for name in (
        "weight_decay",
        "lbfgs_tolerance_grad",
        "lbfgs_tolerance_change",
    ):
        _nonnegative_number(getattr(parameters, name), f"{location}.{name}")
    if parameters.gradient_clip_norm is not None:
        _positive_number(
            parameters.gradient_clip_norm,
            f"{location}.gradient_clip_norm",
        )
    if not isinstance(parameters.resample_each_epoch, bool):
        raise ValueError(f"{location}.resample_each_epoch must be a boolean")


def _resolve_parameters(
    defaults: PseudomodeParameters | MLPParameters,
    overrides: Mapping[str, object],
    location: str,
    *,
    allowed_fields: frozenset[str] | None = None,
    array_description: str = "TOML array",
) -> PseudomodeParameters | MLPParameters:
    dataclass_fields = frozenset(field.name for field in fields(defaults))
    permitted = dataclass_fields if allowed_fields is None else allowed_fields
    _unknown_keys(overrides, permitted, location)
    normalized = _normalize_tuple_fields(
        defaults,
        overrides,
        location,
        array_description=array_description,
    )
    parameters = replace(defaults, **normalized) if normalized else defaults
    if isinstance(parameters, PseudomodeParameters):
        _validate_pseudomode(parameters, location)
    else:
        _validate_mlp(parameters, location)
    return parameters


def default_training_sequence(
    *,
    pseudomode_defaults: PseudomodeParameters = PSEUDOMODE,
    mlp_defaults: MLPParameters = MLP,
) -> TrainingSequence:
    """Return the legacy one-session configuration without mutating defaults."""
    _validate_pseudomode(pseudomode_defaults, "pseudomode")
    _validate_mlp(mlp_defaults, "mlp")
    return TrainingSequence(
        pseudomode=pseudomode_defaults,
        base_mlp=mlp_defaults,
        sessions=(TrainingSession("default", mlp_defaults),),
    )


def load_training_sequence(
    path: str | Path,
    *,
    pseudomode_defaults: PseudomodeParameters = PSEUDOMODE,
    mlp_defaults: MLPParameters = MLP,
) -> TrainingSequence:
    """Load and strictly validate a sparse TOML training sequence."""
    path = Path(path)
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid training sequence TOML {path}: {error}") from error

    _unknown_keys(document, _TOP_LEVEL_KEYS, "top level")
    run_name = _run_name(document.get("name"), "name")
    initialize_from = _run_name(
        document.get("initialize_from"),
        "initialize_from",
    )
    pseudomode_overrides = _table(document.get("pseudomode", {}), "pseudomode")
    mlp_overrides = _table(document.get("mlp", {}), "mlp")
    pseudomode = _resolve_parameters(
        pseudomode_defaults,
        pseudomode_overrides,
        "pseudomode",
    )
    base_mlp = _resolve_parameters(mlp_defaults, mlp_overrides, "mlp")
    assert isinstance(pseudomode, PseudomodeParameters)
    assert isinstance(base_mlp, MLPParameters)

    if "sessions" not in document:
        return TrainingSequence(
            pseudomode=pseudomode,
            base_mlp=base_mlp,
            sessions=(TrainingSession("default", base_mlp),),
            name=run_name,
            initialize_from=initialize_from,
        )

    raw_sessions = document["sessions"]
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError("sessions must be a non-empty TOML array of tables")

    sessions = []
    names = set()
    for index, raw_session in enumerate(raw_sessions, start=1):
        location = f"sessions[{index}]"
        session_table = _table(raw_session, location)
        _unknown_keys(session_table, _SESSION_KEYS, location)

        name = session_table.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{location}.name must be a non-empty string")
        name = name.strip()
        if name in names:
            raise ValueError(f"duplicate training session name: {name!r}")
        names.add(name)

        session_overrides = _table(
            session_table.get("mlp", {}),
            f"{location}.mlp",
        )
        session_mlp = _resolve_parameters(
            base_mlp,
            session_overrides,
            f"{location}.mlp",
            allowed_fields=_SESSION_MLP_FIELDS,
        )
        assert isinstance(session_mlp, MLPParameters)
        sessions.append(TrainingSession(name=name, mlp=session_mlp))

    return TrainingSequence(
        pseudomode=pseudomode,
        base_mlp=base_mlp,
        sessions=tuple(sessions),
        name=run_name,
        initialize_from=initialize_from,
    )


def resolve_model_path(
    sequence: TrainingSequence,
    explicit: str | Path | None = None,
) -> Path:
    """Resolve the checkpoint path for a training sequence.

    An explicit path takes precedence. Otherwise a named sequence gets its
    own directory below ``saved_models``; unnamed sequences retain the legacy
    ``saved_models/mlp`` destination.
    """
    if not isinstance(sequence, TrainingSequence):
        raise TypeError("sequence must be a TrainingSequence object")
    if explicit is not None:
        return Path(explicit)
    if sequence.name is None:
        return MLP_MODEL_PATH
    return MLP_MODEL_PATH.parent.parent / sequence.name / MLP_MODEL_PATH.name


def resolve_pretrained_model_path(folder: str = "mlp") -> Path:
    """Resolve one safe folder below ``saved_models`` to its checkpoint."""
    folder = _run_name(folder, "pretrained model folder")
    if folder is None:
        raise ValueError("pretrained model folder must not be omitted")
    return MLP_MODEL_PATH.parent.parent / folder / MLP_MODEL_PATH.name


def training_metadata_path(model_path: str | Path) -> Path:
    """Return the JSON sidecar path associated with a model checkpoint."""
    model_path = Path(model_path)
    if not model_path.name:
        raise ValueError("model_path must identify a checkpoint file")
    return model_path.with_name(f"{model_path.name}.config.json")


def training_incomplete_marker_path(model_path: str | Path) -> Path:
    """Return the marker used while replacing a checkpoint/sidecar pair."""
    model_path = Path(model_path)
    if not model_path.name:
        raise ValueError("model_path must identify a checkpoint file")
    return model_path.with_name(f".{model_path.name}.incomplete")


def require_complete_training_checkpoint(model_path: str | Path) -> None:
    """Reject a checkpoint whose paired metadata update was interrupted."""
    marker_path = training_incomplete_marker_path(model_path)
    if marker_path.is_file():
        raise ValueError(
            "checkpoint update is incomplete; rerun training before loading "
            f"{Path(model_path)} (marker: {marker_path})"
        )


def _metadata_document(sequence: TrainingSequence) -> dict[str, object]:
    if not isinstance(sequence, TrainingSequence):
        raise TypeError("sequence must be a TrainingSequence object")
    sequence.__post_init__()
    return {
        "format_version": 8,
        "name": sequence.name,
        "initialize_from": sequence.initialize_from,
        "pseudomode": asdict(sequence.pseudomode),
        "base_mlp": asdict(sequence.base_mlp),
        "sessions": [
            {"name": session.name, "mlp": asdict(session.mlp)}
            for session in sequence.sessions
        ],
    }


def save_training_metadata(
    sequence: TrainingSequence,
    model_path: str | Path,
) -> Path:
    """Atomically save a fully resolved training configuration sidecar."""
    metadata_path = training_metadata_path(model_path)
    document = _metadata_document(sequence)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            dir=metadata_path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(
                document,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        temporary_path.replace(metadata_path)
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return metadata_path


def load_training_metadata(model_path: str | Path) -> TrainingSequence:
    """Load and strictly validate a checkpoint's training sidecar."""
    metadata_path = training_metadata_path(model_path)
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(
            f"invalid training metadata JSON {metadata_path}: {error}"
        ) from error

    metadata = _json_object(document, "training metadata")
    _unknown_keys(metadata, _METADATA_KEYS, "training metadata")
    if "format_version" not in metadata:
        raise ValueError(
            "missing key(s) in training metadata: 'format_version'"
        )
    format_version = metadata["format_version"]
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version not in {1, 2, 3, 4, 5, 6, 7, 8}
    ):
        raise ValueError(
            "unsupported training metadata format_version "
            f"{format_version!r}; expected 1, 2, 3, 4, 5, 6, 7, or 8"
        )
    _exact_keys(
        metadata,
        _METADATA_KEYS
        if format_version >= 7
        else _NAMED_METADATA_KEYS
        if format_version >= 4
        else _LEGACY_METADATA_KEYS,
        "training metadata",
    )
    run_name = _run_name(
        metadata.get("name"),
        "training metadata.name",
    )
    initialize_from = _run_name(
        metadata.get("initialize_from"),
        "training metadata.initialize_from",
    )

    pseudomode_values = _json_object(
        metadata["pseudomode"],
        "training metadata.pseudomode",
    )
    _exact_keys(
        pseudomode_values,
        _PSEUDOMODE_FIELDS,
        "training metadata.pseudomode",
    )
    pseudomode = _resolve_parameters(
        PSEUDOMODE,
        pseudomode_values,
        "training metadata.pseudomode",
        array_description="JSON array",
    )
    assert isinstance(pseudomode, PseudomodeParameters)

    base_mlp_values = dict(
        _json_object(
            metadata["base_mlp"],
            "training metadata.base_mlp",
        )
    )
    if format_version == 1:
        base_mlp_values.setdefault("tier_normalized_loss", False)
    if format_version < 3:
        base_mlp_values.setdefault("time_switch", "linear")
        base_mlp_values.setdefault("switch_time_constant", 1.0)
    if format_version < 4:
        base_mlp_values.setdefault("constant_loss_normalization", False)
        base_mlp_values.setdefault("ansatz_scale_normalization", False)
        base_mlp_values.setdefault(
            "normalization_floor",
            MLP.normalization_floor,
        )
    if format_version < 5:
        base_mlp_values.setdefault("lower_tier_loss_cutoff", None)
        base_mlp_values.setdefault("lower_tier_loss_weight", None)
    if format_version < 6:
        base_mlp_values.setdefault("tier_loss_power", None)
        base_mlp_values.setdefault("normalize_hierarchy_coordinates", True)
    if format_version < 8:
        base_mlp_values.setdefault("positive_rdm_ansatz", False)
    _exact_keys(
        base_mlp_values,
        _MLP_FIELDS,
        "training metadata.base_mlp",
    )
    base_mlp = _resolve_parameters(
        MLP,
        base_mlp_values,
        "training metadata.base_mlp",
        array_description="JSON array",
    )
    assert isinstance(base_mlp, MLPParameters)

    raw_sessions = metadata["sessions"]
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError(
            "training metadata.sessions must be a non-empty JSON array"
        )

    sessions = []
    names = set()
    for index, raw_session in enumerate(raw_sessions, start=1):
        location = f"training metadata.sessions[{index}]"
        session_values = _json_object(raw_session, location)
        _exact_keys(session_values, _SESSION_KEYS, location)

        name = session_values["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{location}.name must be a non-empty string")
        if name in names:
            raise ValueError(f"duplicate training session name: {name!r}")
        names.add(name)

        session_mlp_values = dict(
            _json_object(
                session_values["mlp"],
                f"{location}.mlp",
            )
        )
        if format_version == 1:
            session_mlp_values.setdefault(
                "tier_normalized_loss",
                base_mlp.tier_normalized_loss,
            )
        if format_version < 3:
            session_mlp_values.setdefault(
                "time_switch",
                base_mlp.time_switch,
            )
            session_mlp_values.setdefault(
                "switch_time_constant",
                base_mlp.switch_time_constant,
            )
        if format_version < 4:
            session_mlp_values.setdefault(
                "constant_loss_normalization",
                base_mlp.constant_loss_normalization,
            )
            session_mlp_values.setdefault(
                "ansatz_scale_normalization",
                base_mlp.ansatz_scale_normalization,
            )
            session_mlp_values.setdefault(
                "normalization_floor",
                base_mlp.normalization_floor,
            )
        if format_version < 5:
            session_mlp_values.setdefault(
                "lower_tier_loss_cutoff",
                base_mlp.lower_tier_loss_cutoff,
            )
            session_mlp_values.setdefault(
                "lower_tier_loss_weight",
                base_mlp.lower_tier_loss_weight,
            )
        if format_version < 6:
            session_mlp_values.setdefault(
                "tier_loss_power",
                base_mlp.tier_loss_power,
            )
            session_mlp_values.setdefault(
                "normalize_hierarchy_coordinates",
                base_mlp.normalize_hierarchy_coordinates,
            )
        if format_version < 8:
            session_mlp_values.setdefault(
                "positive_rdm_ansatz", base_mlp.positive_rdm_ansatz
            )
        _exact_keys(session_mlp_values, _MLP_FIELDS, f"{location}.mlp")
        session_mlp = _resolve_parameters(
            base_mlp,
            session_mlp_values,
            f"{location}.mlp",
            allowed_fields=_MLP_FIELDS,
            array_description="JSON array",
        )
        assert isinstance(session_mlp, MLPParameters)
        sessions.append(TrainingSession(name=name, mlp=session_mlp))

    return TrainingSequence(
        pseudomode=pseudomode,
        base_mlp=base_mlp,
        sessions=tuple(sessions),
        name=run_name,
        initialize_from=initialize_from,
    )


__all__ = [
    "TrainingSequence",
    "TrainingSession",
    "default_training_sequence",
    "load_training_metadata",
    "load_training_sequence",
    "require_complete_training_checkpoint",
    "resolve_model_path",
    "resolve_pretrained_model_path",
    "save_training_metadata",
    "training_incomplete_marker_path",
    "training_metadata_path",
]
