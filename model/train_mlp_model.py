"""Train, optionally resume, and save the Section-III MLP."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from experiment_parameters import (
    MLP,
    MLP_MODEL_PATH,
    PSEUDOMODE,
    MLPParameters,
    PseudomodeParameters,
)
from heom.heom_rep import heom_state
from model import (
    EpochRecord,
    HEOMMLP,
    HEOMPINNLoss,
    TrainingConfig,
    train_mlp,
)
from training_sequence import (
    TrainingSequence,
    default_training_sequence,
    load_training_metadata,
    load_training_sequence,
    save_training_metadata,
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
    return hierarchy, rho0, liouvillian


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
        help="load the --model-path checkpoint before training",
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
        default=MLP_MODEL_PATH,
        help="checkpoint path (default: saved_models/mlp/mlp_state_dict.pt)",
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
_CHECKPOINT_MLP_FIELDS = ("hidden_sizes", "activation", "dtype")


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
    return tuple(differences)


def _validate_resume_config(
    sequence: TrainingSequence,
    model_path: Path,
) -> None:
    metadata_path = training_metadata_path(model_path)
    if not metadata_path.is_file():
        return
    saved = load_training_metadata(model_path)
    differences = _checkpoint_config_differences(sequence, saved)
    if differences:
        rendered = ", ".join(differences)
        raise ValueError(
            "checkpoint parameters are incompatible with this training "
            f"sequence: {rendered}"
        )


def run_training_sequence(
    sequence: TrainingSequence,
    *,
    resume: bool = False,
    plot_loss: bool = False,
    model_path: Path = MLP_MODEL_PATH,
):
    """Run all sessions on one in-memory model and checkpoint each stage."""
    model_path = Path(model_path)
    if not sequence.sessions:
        raise ValueError("a training sequence must contain at least one session")

    pseudomode = sequence.pseudomode
    if resume:
        _validate_resume_config(sequence, model_path)
    dtype = _training_dtype(sequence)
    try:
        device = torch.device(sequence.base_mlp.device)
    except RuntimeError as error:
        raise ValueError(
            f"invalid MLP device {sequence.base_mlp.device!r}"
        ) from error
    torch.manual_seed(sequence.base_mlp.seed)
    hierarchy, rho0, liouvillian = build_training_problem(pseudomode)

    model = HEOMMLP(
        hierarchy,
        hidden_sizes=sequence.base_mlp.hidden_sizes,
        rho0=rho0,
        t_start=pseudomode.t_start,
        t_stop=pseudomode.t_stop,
        activation=sequence.base_mlp.activation,
        dtype=dtype,
        device=device,
    )
    if resume:
        load_saved_model(model, model_path, device)

    objective = HEOMPINNLoss(
        hierarchy,
        liouvillian=liouvillian,
        dtype=dtype,
        device=device,
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
        if parameters.optimizer == "lbfgs":
            print(
                "Optimizer: L-BFGS (float64, fixed full batch, "
                "strong-Wolfe line search, residual-sum scaling)"
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

        save_model(model, model_path)
        metadata_path = save_training_metadata(sequence, model_path)
        print(f"Session training time: {result.elapsed_seconds:.3f} s")
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

    if args.resume:
        if not args.model_path.is_file():
            parser.error(f"checkpoint does not exist: {args.model_path}")
        try:
            _validate_resume_config(sequence, args.model_path)
        except (OSError, ValueError) as error:
            parser.error(str(error))

    model, results = run_training_sequence(
        sequence,
        resume=args.resume,
        plot_loss=args.plot_loss,
        model_path=args.model_path,
    )
    return model, results[-1]


if __name__ == "__main__":
    main()
