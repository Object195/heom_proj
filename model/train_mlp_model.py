"""Train, optionally resume, and save the Section-III MLP."""

import argparse
from collections.abc import Mapping
from contextlib import redirect_stdout
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from experiment_parameters import (
    MLP,
    PSEUDOMODE,
    MLPParameters,
    PseudomodeParameters,
)
from heom.heom_rep import heom_state
from heom.heom_solver import prepare_heom_initial_state
from model import (
    EpochRecord,
    HEOMMLP,
    HEOMPINNLoss,
    TrainingConfig,
    compute_heom_dynamical_scales,
    train_mlp,
)
from training_sequence import (
    TrainingSequence,
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


class LiveLossPlot:
    """Interactive logarithmic loss plot updated during training."""

    def __init__(self, update_every: int, final_epoch: int):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.update_every = update_every
        self.final_epoch = final_epoch
        self.epochs = []
        self.losses = []
        plt.ion()
        self.figure, self.axis = plt.subplots(dpi=140)
        (self.line,) = self.axis.plot(
            [],
            [],
            color="#7B2CBF",
            marker="o",
            markersize=4,
        )
        self.axis.set_xlabel("Epoch")
        self.axis.set_ylabel("Dynamical loss")
        self.axis.set_yscale("log")
        self.axis.grid(True, alpha=0.3)
        self.figure.tight_layout()

    def __call__(self, record: EpochRecord):
        if not (
            record.epoch == 1
            or record.epoch % self.update_every == 0
            or record.epoch == self.final_epoch
        ):
            return
        self.epochs.append(record.epoch)
        self.losses.append(max(record.loss, np.finfo(float).tiny))
        self.redraw()

    def redraw(self):
        self.line.set_data(self.epochs, self.losses)
        self.axis.relim()
        self.axis.autoscale_view()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def finish(self, *, show: bool = True):
        self.plt.ioff()
        if show:
            self.plt.show()


def load_saved_model(model: HEOMMLP, path: Path, device: torch.device):
    """Load saved MLP parameters before continuing training."""
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Resumed MLP model: {path}")


def load_pretrained_network(
    model: HEOMMLP,
    path: Path,
    device: torch.device,
) -> None:
    """Initialize only the shared MLP network from a saved checkpoint."""
    require_complete_training_checkpoint(path)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"invalid pretrained model state dictionary: {path}")
    prefix = "network."
    network_state = {
        name.removeprefix(prefix): value
        for name, value in state_dict.items()
        if isinstance(name, str) and name.startswith(prefix)
    }
    if not network_state:
        raise ValueError(
            f"pretrained checkpoint contains no network parameters: {path}"
        )
    try:
        model.network.load_state_dict(network_state)
    except RuntimeError as error:
        raise ValueError(
            f"pretrained network is incompatible with the current MLP: {path}"
        ) from error
    print(f"Initialized MLP network from: {path}")


def save_model(model: HEOMMLP, path: Path) -> None:
    """Atomically replace a checkpoint without risking the previous stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(model.state_dict(), temporary)
        temporary.replace(path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def build_optimizer(
    model: HEOMMLP,
    name: str,
    parameters: MLPParameters = MLP,
):
    if name == "lbfgs":
        return torch.optim.LBFGS(
            model.parameters(),
            lr=parameters.lbfgs_learning_rate,
            max_iter=parameters.lbfgs_max_iter,
            max_eval=parameters.lbfgs_max_eval,
            history_size=parameters.lbfgs_history_size,
            tolerance_grad=parameters.lbfgs_tolerance_grad,
            tolerance_change=parameters.lbfgs_tolerance_change,
            line_search_fn=parameters.lbfgs_line_search,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=parameters.learning_rate,
        weight_decay=parameters.weight_decay,
    )


def build_training_problem(parameters: PseudomodeParameters = PSEUDOMODE):
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.diag([1.0, -1.0])
    h_system = 0.5 * parameters.delta * sigma_z
    h_system += 0.5 * parameters.v * sigma_x
    rho0 = np.diag([1.0, 0.0])
    frequencies = np.array(
        [0.5 * parameters.gamma + 1j * parameters.w0],
        dtype=np.complex128,
    )
    coefficients = np.array([parameters.g**2], dtype=np.complex128)
    hierarchy = heom_state(
        K=0,
        L=parameters.heom_depth,
        H_s=h_system,
        H_c=sigma_z,
        C_list=coefficients,
        gamma_list=frequencies,
    )
    liouvillian = hierarchy.build_Liouvillian(
        markovian_terminator=False,
        normalized=True,
    )
    preparation_start = perf_counter()
    initial_heom_state = prepare_heom_initial_state(
        hierarchy,
        rho0,
        parameters.t_start,
        liouvillian=liouvillian,
        method="BDF",
        rtol=parameters.rtol,
        atol=parameters.atol,
    )
    if parameters.t_start > 0.0:
        print(
            "Sparse HEOM initial-state preparation to "
            f"t={parameters.t_start:g}: "
            f"{perf_counter() - preparation_start:.3f} s"
        )
    return hierarchy, initial_heom_state, liouvillian


class _OptimizerAction(argparse.Action):
    """Record whether ``--optimizer`` was explicitly supplied."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.optimizer_explicit = True


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(optimizer_explicit=False)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="load the resolved checkpoint before training",
    )
    parser.add_argument(
        "--initialize-from",
        nargs="?",
        const="mlp",
        metavar="FOLDER",
        help=(
            "initialize network weights from "
            "saved_models/FOLDER/mlp_state_dict.pt; if FOLDER is omitted, "
            "use saved_models/mlp"
        ),
    )
    parser.add_argument(
        "--plot-loss",
        action="store_true",
        help="display an interactive log-scale loss curve",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "lbfgs"),
        default=MLP.optimizer,
        action=_OptimizerAction,
        help="optimizer used for this training run",
    )
    parser.add_argument(
        "--sequence",
        "--config",
        dest="sequence",
        type=Path,
        help=(
            "TOML file containing sparse defaults and an ordered training "
            "sequence"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help=(
            "explicit checkpoint path; otherwise a sequence name selects "
            "saved_models/<name>/mlp_state_dict.pt, and unnamed sequences "
            "create a timestamped parameter folder for fresh training "
            "(unnamed --resume still uses saved_models/mlp/mlp_state_dict.pt)"
        ),
    )
    return parser


def build_training_config(
    pseudomode: PseudomodeParameters,
    parameters: MLPParameters,
) -> TrainingConfig:
    """Translate one resolved session into the low-level training config."""
    using_lbfgs = parameters.optimizer == "lbfgs"
    return TrainingConfig(
        t_start=pseudomode.t_start,
        t_stop=pseudomode.t_stop,
        epochs=parameters.epochs,
        collocation_points=parameters.collocation_points,
        batch_size=(
            parameters.collocation_points
            if using_lbfgs
            else parameters.batch_size
        ),
        learning_rate=parameters.learning_rate,
        weight_decay=parameters.weight_decay,
        gradient_clip_norm=parameters.gradient_clip_norm,
        resample_each_epoch=(
            False if using_lbfgs else parameters.resample_each_epoch
        ),
        seed=parameters.seed,
        log_every=parameters.log_every,
    )


def _training_dtype(sequence: TrainingSequence) -> torch.dtype:
    return getattr(torch, sequence.base_mlp.dtype)


_CHECKPOINT_PSEUDOMODE_FIELDS = (
    "w0",
    "delta",
    "v",
    "g",
    "gamma",
    "heom_depth",
    "t_start",
    "t_stop",
)
_CHECKPOINT_MLP_FIELDS = (
    "hidden_sizes",
    "activation",
    "dtype",
    "time_switch",
    "switch_time_constant",
    "normalize_hierarchy_coordinates",
    "ansatz_scale_normalization",
    "positive_rdm_ansatz",
)
_PRETRAINED_MLP_FIELDS = (
    "hidden_sizes",
    "activation",
    "dtype",
    "normalize_hierarchy_coordinates",
    "positive_rdm_ansatz",
)


def _checkpoint_config_differences(
    current: TrainingSequence,
    saved: TrainingSequence,
) -> tuple[str, ...]:
    differences = []
    for name in _CHECKPOINT_PSEUDOMODE_FIELDS:
        if getattr(current.pseudomode, name) != getattr(
            saved.pseudomode, name
        ):
            differences.append(f"pseudomode.{name}")
    for name in _CHECKPOINT_MLP_FIELDS:
        if getattr(current.base_mlp, name) != getattr(saved.base_mlp, name):
            differences.append(f"mlp.{name}")
    if (
        current.base_mlp.ansatz_scale_normalization
        and saved.base_mlp.ansatz_scale_normalization
        and current.base_mlp.normalization_floor
        != saved.base_mlp.normalization_floor
    ):
        differences.append("mlp.normalization_floor")
    return tuple(differences)


def _validate_resume_config(
    sequence: TrainingSequence,
    model_path: Path,
) -> None:
    require_complete_training_checkpoint(model_path)
    metadata_path = training_metadata_path(model_path)
    if not metadata_path.is_file():
        if (
            sequence.base_mlp.ansatz_scale_normalization
            or not sequence.base_mlp.normalize_hierarchy_coordinates
            or sequence.base_mlp.positive_rdm_ansatz
        ):
            raise ValueError(
                "cannot resume with the requested model parameterization "
                f"because checkpoint metadata is missing: {metadata_path}"
            )
        return
    saved = load_training_metadata(model_path)
    differences = _checkpoint_config_differences(sequence, saved)
    if differences:
        rendered = ", ".join(differences)
        raise ValueError(
            "checkpoint parameters are incompatible with this training "
            f"sequence: {rendered}"
        )


def _validate_pretrained_config(
    sequence: TrainingSequence,
    pretrained_model_path: Path,
) -> None:
    """Validate metadata fields that determine reusable network semantics."""
    require_complete_training_checkpoint(pretrained_model_path)
    metadata_path = training_metadata_path(pretrained_model_path)
    if not metadata_path.is_file():
        if sequence.base_mlp.positive_rdm_ansatz:
            raise ValueError(
                "cannot verify pretrained root-output semantics for a positive "
                f"RDM ansatz because checkpoint metadata is missing: {metadata_path}"
            )
        return
    saved = load_training_metadata(pretrained_model_path)
    differences = tuple(
        f"mlp.{name}"
        for name in _PRETRAINED_MLP_FIELDS
        if getattr(sequence.base_mlp, name) != getattr(saved.base_mlp, name)
    )
    if differences:
        rendered = ", ".join(differences)
        raise ValueError(
            "pretrained network parameters are incompatible with this "
            f"training sequence: {rendered}"
        )


def _save_training_stage(
    model: HEOMMLP,
    sequence: TrainingSequence,
    model_path: Path,
) -> Path:
    """Replace a checkpoint pair while leaving an interruption marker."""
    marker_path = training_incomplete_marker_path(model_path)
    marker_path.write_text(
        "checkpoint and metadata update in progress\n",
        encoding="utf-8",
    )
    save_model(model, model_path)
    metadata_path = save_training_metadata(sequence, model_path)
    marker_path.unlink()
    return metadata_path


class _TeeStream:
    """Write console output to both the terminal and a persistent log."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(
            getattr(stream, "isatty", lambda: False)()
            for stream in self.streams
        )


def run_training_sequence(
    sequence: TrainingSequence,
    *,
    resume: bool = False,
    plot_loss: bool = False,
    model_path: Path | None = None,
    pretrained_model_path: Path | None = None,
):
    """Run all sessions, checkpoint each stage, and tee output to a log."""
    auto_name = not resume and model_path is None and sequence.name is None
    model_path = resolve_model_path(sequence, model_path)
    if (
        pretrained_model_path is None
        and sequence.initialize_from is not None
    ):
        pretrained_model_path = resolve_pretrained_model_path(
            sequence.initialize_from
        )
    if resume and pretrained_model_path is not None:
        raise ValueError("resume and pretrained initialization are exclusive")
    if not sequence.sessions:
        raise ValueError("a training sequence must contain at least one session")
    if resume:
        _validate_resume_config(sequence, model_path)
    if pretrained_model_path is not None:
        pretrained_model_path = Path(pretrained_model_path)
        if not pretrained_model_path.is_file():
            raise ValueError(
                f"pretrained checkpoint does not exist: {pretrained_model_path}"
            )
        _validate_pretrained_config(sequence, pretrained_model_path)

    if auto_name:
        parameters = sequence.pseudomode
        run_name = (
            f"{datetime.now():%m-%d-%H-%M}_g_{parameters.g}"
            f"_L_{parameters.heom_depth}"
            f"_t_{parameters.t_start}_{parameters.t_stop}"
        )
        saved_models = model_path.parent.parent
        candidate = saved_models / run_name
        suffix = 2
        # Reserve the folder atomically so launches in the same minute do
        # not overwrite each other's checkpoints or training logs.
        while True:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                candidate = saved_models / f"{run_name}_{suffix}"
                suffix += 1
        sequence = replace(sequence, name=candidate.name)
        model_path = candidate / model_path.name

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = model_path.parent / "training.log"
    log_mode = "a" if resume else "w"
    with log_path.open(
        log_mode,
        encoding="utf-8",
        buffering=1,
    ) as log_stream:
        with redirect_stdout(_TeeStream(sys.stdout, log_stream)):
            if resume:
                print()
            print(f"Training log: {log_path}")
            return _run_training_sequence_body(
                sequence,
                resume=resume,
                plot_loss=plot_loss,
                model_path=model_path,
                pretrained_model_path=pretrained_model_path,
            )


def _run_training_sequence_body(
    sequence: TrainingSequence,
    *,
    resume: bool,
    plot_loss: bool,
    model_path: Path,
    pretrained_model_path: Path | None = None,
):
    """Implementation executed while stdout is mirrored into training.log."""
    pseudomode = sequence.pseudomode
    dtype = _training_dtype(sequence)
    try:
        device = torch.device(sequence.base_mlp.device)
    except RuntimeError as error:
        raise ValueError(
            f"invalid MLP device {sequence.base_mlp.device!r}"
        ) from error
    torch.manual_seed(sequence.base_mlp.seed)
    hierarchy, initial_heom_state, liouvillian = build_training_problem(
        pseudomode
    )

    model = HEOMMLP(
        hierarchy,
        hidden_sizes=sequence.base_mlp.hidden_sizes,
        initial_heom_state=initial_heom_state,
        t_start=pseudomode.t_start,
        t_stop=pseudomode.t_stop,
        activation=sequence.base_mlp.activation,
        time_switch=sequence.base_mlp.time_switch,
        switch_time_constant=sequence.base_mlp.switch_time_constant,
        normalize_hierarchy_coordinates=(
            sequence.base_mlp.normalize_hierarchy_coordinates
        ),
        positive_rdm_ansatz=sequence.base_mlp.positive_rdm_ansatz,
        dtype=dtype,
        device=device,
    )
    if resume:
        load_saved_model(model, model_path, device)
    elif pretrained_model_path is not None:
        load_pretrained_network(model, pretrained_model_path, device)

    anchored_state = (
        model.complex_initial_state().detach().cpu().numpy()
    )
    scales = compute_heom_dynamical_scales(
        hierarchy,
        anchored_state,
        liouvillian,
        t_start=pseudomode.t_start,
        t_stop=pseudomode.t_stop,
        tier_normalized=sequence.base_mlp.tier_normalized_loss,
        lower_tier_cutoff=sequence.base_mlp.lower_tier_loss_cutoff,
        lower_tier_weight=sequence.base_mlp.lower_tier_loss_weight,
        tier_loss_power=sequence.base_mlp.tier_loss_power,
        time_switch=sequence.base_mlp.time_switch,
        switch_time_constant=sequence.base_mlp.switch_time_constant,
        normalization_floor=sequence.base_mlp.normalization_floor,
    )
    if sequence.base_mlp.ansatz_scale_normalization:
        model.set_correction_scale(scales.correction_scale)

    objective = HEOMPINNLoss(
        hierarchy,
        liouvillian=liouvillian,
        tier_normalized=sequence.base_mlp.tier_normalized_loss,
        lower_tier_cutoff=sequence.base_mlp.lower_tier_loss_cutoff,
        lower_tier_weight=sequence.base_mlp.lower_tier_loss_weight,
        tier_loss_power=sequence.base_mlp.tier_loss_power,
        normalization_loss=(
            scales.effective_constant_loss
            if sequence.base_mlp.constant_loss_normalization
            else None
        ),
        dtype=dtype,
        device=device,
    )
    print(
        "Anchor dynamical scales: "
        f"L_const={scales.constant_loss:.6e}; "
        f"effective_L_const={scales.effective_constant_loss:.6e}; "
        f"global_RMS(L chi_s)="
        f"{np.sqrt(scales.global_constant_loss):.6e}"
    )
    print(
        "Ansatz correction scale: "
        + (
            f"a_s={scales.correction_scale:.6e} (enabled)"
            if sequence.base_mlp.ansatz_scale_normalization
            else "a_s=1.000000e+00 (disabled)"
        )
    )
    print(
        "Constant-loss normalization: "
        + (
            "enabled (reported loss = raw loss / effective_L_const)"
            if sequence.base_mlp.constant_loss_normalization
            else "disabled"
        )
    )
    print(
        "Tier-0 RDM ansatz: "
        + (
            "normalized A A^dagger; A = sqrt(rho_init) + s(t) * a_s * B"
            if sequence.base_mlp.positive_rdm_ansatz
            else "additive symmetric, trace-preserving correction"
        )
    )
    total_epochs = sum(session.mlp.epochs for session in sequence.sessions)
    loss_plot = (
        LiveLossPlot(1, total_epochs)
        if plot_loss
        else None
    )

    results = []
    epoch_offset = 0
    session_count = len(sequence.sessions)
    for index, session in enumerate(sequence.sessions, start=1):
        parameters = session.mlp
        print(f"Session {index}/{session_count}: {session.name}")
        print(
            "Loss: "
            + (
                "power-law tier-normalized residual "
                f"(p={sequence.base_mlp.tier_loss_power:g})"
                if sequence.base_mlp.tier_loss_power is not None
                else "two-group tier-normalized residual "
                f"(tiers 0-{sequence.base_mlp.lower_tier_loss_cutoff}: "
                f"beta={sequence.base_mlp.lower_tier_loss_weight:g})"
                if sequence.base_mlp.lower_tier_loss_cutoff is not None
                else "equal-weight tier-normalized residual"
                if sequence.base_mlp.tier_normalized_loss
                else "global ADO-normalized residual"
            )
            + (
                " / effective_L_const"
                if sequence.base_mlp.constant_loss_normalization
                else ""
            )
        )
        if parameters.optimizer == "lbfgs":
            print(
                "Optimizer: L-BFGS (float64, fixed full batch, "
                "strong-Wolfe line search, "
                + (
                    "dimensionless normalized loss)"
                    if sequence.base_mlp.constant_loss_normalization
                    else "legacy fixed loss scaling)"
                )
            )
        else:
            print("Optimizer: Adam")
        print(
            f"Epochs: {parameters.epochs}; collocation points: "
            f"{parameters.collocation_points}"
        )

        callback = None
        if loss_plot is not None:
            current_offset = epoch_offset
            current_epochs = parameters.epochs
            current_log_every = parameters.log_every

            def callback(
                record,
                offset=current_offset,
                session_epochs=current_epochs,
                log_every=current_log_every,
            ):
                if (
                    record.epoch == 1
                    or record.epoch % log_every == 0
                    or record.epoch == session_epochs
                ):
                    loss_plot(replace(record, epoch=offset + record.epoch))

        optimizer = build_optimizer(
            model,
            parameters.optimizer,
            parameters,
        )
        result = train_mlp(
            model,
            objective,
            build_training_config(pseudomode, parameters),
            optimizer=optimizer,
            callback=callback,
        )
        results.append(result)
        epoch_offset += parameters.epochs

        metadata_path = _save_training_stage(model, sequence, model_path)
        print(f"Session training time: {result.elapsed_seconds:.3f} s")
        if sequence.base_mlp.constant_loss_normalization:
            raw_final_loss = result.final.raw_loss
            if raw_final_loss is None:
                raw_final_loss = (
                    result.final.loss * scales.effective_constant_loss
                )
            print(
                "Session final normalized loss: "
                f"{result.final.loss:.6e}"
            )
            print(f"Session final raw loss: {raw_final_loss:.6e}")
        else:
            print(f"Session final loss: {result.final.loss:.6e}")
        print(f"Saved MLP model: {model_path}")
        print(f"Saved resolved parameters: {metadata_path}")

    if loss_plot is not None:
        loss_plot.finish()
    return model, tuple(results)


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.sequence is not None and args.optimizer_explicit:
        parser.error(
            "--optimizer cannot be combined with --sequence; set the "
            "optimizer in each TOML session"
        )
    try:
        sequence = (
            load_training_sequence(args.sequence)
            if args.sequence is not None
            else default_training_sequence()
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.sequence is None:
        try:
            base_mlp = replace(sequence.base_mlp, optimizer=args.optimizer)
            session = replace(
                sequence.sessions[0],
                mlp=replace(
                    sequence.sessions[0].mlp,
                    optimizer=args.optimizer,
                ),
            )
            sequence = replace(
                sequence,
                base_mlp=base_mlp,
                sessions=(session,),
            )
        except ValueError as error:
            parser.error(str(error))

    if args.initialize_from is not None:
        try:
            sequence = replace(
                sequence,
                initialize_from=args.initialize_from,
            )
        except ValueError as error:
            parser.error(str(error))
    if args.resume and sequence.initialize_from is not None:
        parser.error("--resume cannot be combined with transfer initialization")

    model_path = resolve_model_path(sequence, args.model_path)
    try:
        pretrained_model_path = (
            resolve_pretrained_model_path(sequence.initialize_from)
            if sequence.initialize_from is not None
            else None
        )
    except ValueError as error:
        parser.error(str(error))

    if args.resume:
        if not model_path.is_file():
            parser.error(f"checkpoint does not exist: {model_path}")
        try:
            _validate_resume_config(sequence, model_path)
        except (OSError, ValueError) as error:
            parser.error(str(error))

    if pretrained_model_path is not None:
        if not pretrained_model_path.is_file():
            parser.error(
                f"pretrained checkpoint does not exist: "
                f"{pretrained_model_path}"
            )
        try:
            _validate_pretrained_config(sequence, pretrained_model_path)
        except (OSError, ValueError) as error:
            parser.error(str(error))

    model, results = run_training_sequence(
        sequence,
        resume=args.resume,
        plot_loss=args.plot_loss,
        model_path=(
            None
            if not args.resume and args.model_path is None and sequence.name is None
            else model_path
        ),
        pretrained_model_path=pretrained_model_path,
    )
    return model, results[-1]


if __name__ == "__main__":
    main()
