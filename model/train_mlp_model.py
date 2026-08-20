"""Train, optionally resume, and save the Section-III MLP."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from experiment_parameters import MLP, MLP_MODEL_PATH, PSEUDOMODE
from heom.heom_rep import heom_state
from model import (
    EpochRecord,
    HEOMMLP,
    HEOMPINNLoss,
    TrainingConfig,
    train_mlp,
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


def build_optimizer(model: HEOMMLP, name: str):
    if name == "lbfgs":
        return torch.optim.LBFGS(
            model.parameters(),
            lr=MLP.lbfgs_learning_rate,
            max_iter=MLP.lbfgs_max_iter,
            max_eval=MLP.lbfgs_max_eval,
            history_size=MLP.lbfgs_history_size,
            tolerance_grad=MLP.lbfgs_tolerance_grad,
            tolerance_change=MLP.lbfgs_tolerance_change,
            line_search_fn=MLP.lbfgs_line_search,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=MLP.learning_rate,
        weight_decay=MLP.weight_decay,
    )


def build_training_problem():
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.diag([1.0, -1.0])
    h_system = 0.5 * PSEUDOMODE.delta * sigma_z
    h_system += 0.5 * PSEUDOMODE.v * sigma_x
    rho0 = np.diag([1.0, 0.0])
    frequencies = np.array(
        [0.5 * PSEUDOMODE.gamma + 1j * PSEUDOMODE.w0],
        dtype=np.complex128,
    )
    coefficients = np.array([PSEUDOMODE.g**2], dtype=np.complex128)
    hierarchy = heom_state(
        K=0,
        L=PSEUDOMODE.heom_depth,
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


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="load saved_models/mlp/mlp_state_dict.pt before training",
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
        help="optimizer used for this training run",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    dtype = (
        torch.float64
        if args.optimizer == "lbfgs"
        else getattr(torch, MLP.dtype)
    )
    device = torch.device(MLP.device)
    torch.manual_seed(MLP.seed)
    hierarchy, rho0, liouvillian = build_training_problem()

    model = HEOMMLP(
        hierarchy,
        hidden_sizes=MLP.hidden_sizes,
        rho0=rho0,
        t_start=PSEUDOMODE.t_start,
        t_stop=PSEUDOMODE.t_stop,
        activation=MLP.activation,
        dtype=dtype,
        device=device,
    )
    if args.resume:
        load_saved_model(model, MLP_MODEL_PATH, device)

    objective = HEOMPINNLoss(
        hierarchy,
        liouvillian=liouvillian,
        dtype=dtype,
        device=device,
    )
    config = TrainingConfig(
        t_start=PSEUDOMODE.t_start,
        t_stop=PSEUDOMODE.t_stop,
        epochs=MLP.epochs,
        collocation_points=MLP.collocation_points,
        batch_size=(
            MLP.collocation_points
            if args.optimizer == "lbfgs"
            else MLP.batch_size
        ),
        learning_rate=MLP.learning_rate,
        weight_decay=MLP.weight_decay,
        gradient_clip_norm=MLP.gradient_clip_norm,
        resample_each_epoch=(
            False
            if args.optimizer == "lbfgs"
            else MLP.resample_each_epoch
        ),
        seed=MLP.seed,
        log_every=MLP.log_every,
    )
    loss_plot = (
        LiveLossPlot(MLP.log_every, MLP.epochs)
        if args.plot_loss
        else None
    )
    optimizer = build_optimizer(model, args.optimizer)
    if args.optimizer == "lbfgs":
        print(
            "Optimizer: L-BFGS (float64, fixed full batch, "
            "strong-Wolfe line search)"
        )
    else:
        print("Optimizer: Adam")
    result = train_mlp(
        model,
        objective,
        config,
        optimizer=optimizer,
        callback=loss_plot,
    )

    MLP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MLP_MODEL_PATH)
    print(f"Training time: {result.elapsed_seconds:.3f} s")
    print(f"Final loss: {result.final.loss:.6e}")
    print(f"Saved MLP model: {MLP_MODEL_PATH}")
    if loss_plot is not None:
        loss_plot.finish()
    return model, result


if __name__ == "__main__":
    main()
