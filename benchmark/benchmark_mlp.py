"""Compare Lindblad, sparse normalized HEOM, and the Section-III MLP PINN.

Examples
--------
Run the full benchmark with the defaults used by ``benchmark_lindbladian``::

    python -m benchmark.benchmark_mlp

Use a smaller development run without opening a plot window::

    python -m benchmark.benchmark_mlp --depth 3 --t-stop 5 \
        --n-times 101 --epochs 20 --collocation-points 32 --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

# Support both ``python -m benchmark.benchmark_mlp`` and direct IDE execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from qutip import basis, destroy, mesolve, qeye, sigmax, sigmaz, tensor

from heom.heom_rep import heom_state
from heom.heom_solver import solve_heom
from model import (
    HEOMMLP,
    HEOMPINNLoss,
    LossWeights,
    TrainingConfig,
    solve_mlp,
    train_mlp,
)


# Same damped-pseudomode spin model as benchmark_lindbladian.py.
W0 = 1.0
DELTA = 1.0
V = 0.1
G = 1.0
GAMMA = 0.1
CAVITY_DIMENSION = 20


def run_lindbladian(t_eval, *, cavity_dimension=CAVITY_DIMENSION):
    """Propagate the explicit damped-cavity Lindblad reference."""
    annihilation = tensor(qeye(2), destroy(cavity_dimension))
    sz_full = tensor(sigmaz(), qeye(cavity_dimension))
    sx_full = tensor(sigmax(), qeye(cavity_dimension))
    h_system = 0.5 * DELTA * sz_full + 0.5 * V * sx_full
    h_cavity = W0 * annihilation.dag() * annihilation
    h_interaction = G * sz_full * (annihilation + annihilation.dag())
    h_total = h_system + h_cavity + h_interaction
    collapse_operators = [np.sqrt(GAMMA) * annihilation]
    psi0 = tensor(basis(2, 0), basis(cavity_dimension, 0))

    start = perf_counter()
    result = mesolve(
        h_total,
        psi0,
        t_eval,
        collapse_operators,
        [sz_full],
    )
    elapsed = perf_counter() - start
    print(f"Lindbladian propagation: {elapsed:.3f} s")
    return np.real(result.expect[0])


def build_normalized_hard_heom(depth):
    """Build the shared hierarchy, initial state, and requested Liouvillian."""
    h_system = (
        0.5 * DELTA * sigmaz().full()
        + 0.5 * V * sigmax().full()
    )
    coupling_operator = sigmaz().full()
    rho0 = (basis(2, 0) * basis(2, 0).dag()).full()
    frequencies = np.array(
        [0.5 * GAMMA + 1j * W0],
        dtype=np.complex128,
    )
    coefficients = np.array([G**2], dtype=np.complex128)
    hierarchy = heom_state(
        K=len(frequencies) - 1,
        L=depth,
        H_s=h_system,
        H_c=coupling_operator,
        C_list=coefficients,
        gamma_list=frequencies,
    )

    start = perf_counter()
    liouvillian = hierarchy.build_Liouvillian(
        markovian_terminator=False,
        normalized=True,
    )
    elapsed = perf_counter() - start
    print(
        "Normalized hard-cutoff HEOM construction "
        f"(L={depth}, ADOs={hierarchy.nADO}, shape={liouvillian.shape}): "
        f"{elapsed:.3f} s"
    )
    return hierarchy, rho0, liouvillian


def run_sparse_numerics(hierarchy, rho0, liouvillian, t_eval):
    """Propagate the same normalized hard-cutoff operator used by the PINN."""
    start = perf_counter()
    result = solve_heom(
        hierarchy,
        rho0,
        t_eval,
        liouvillian=liouvillian,
        method="BDF",
        rtol=1e-8,
        atol=1e-10,
    )
    elapsed = perf_counter() - start
    print(
        f"Sparse HEOM propagation: {elapsed:.3f} s "
        f"[nfev={result.nfev}, njev={result.njev}, nlu={result.nlu}]"
    )
    return np.real(result.expectation(sigmaz().full()))


def build_and_train_mlp(
    hierarchy,
    rho0,
    liouvillian,
    args,
):
    """Construct the Section-III model/loss and optionally train or restore it."""
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation,
        dtype=dtype,
        device=device,
    )
    objective = HEOMPINNLoss(
        hierarchy,
        rho0,
        liouvillian=liouvillian,
        weights=LossWeights(
            dynamics=args.dynamics_weight,
            initial_condition=(
                args.collocation_points
                if args.initial_weight is None
                else args.initial_weight
            ),
            trace=(
                1.0 / args.collocation_points
                if args.trace_weight is None
                else args.trace_weight
            ),
        ),
        dtype=dtype,
        device=device,
    )

    if args.checkpoint_in is not None:
        checkpoint = torch.load(
            args.checkpoint_in,
            map_location=device,
            weights_only=True,
        )
        expected_metadata = {
            "format_version": 1,
            "hierarchy_fingerprint": model.hierarchy_fingerprint,
            "hidden_sizes": model.hidden_sizes,
            "activation": model.activation_name,
            "dtype": args.dtype,
        }
        mismatches = []
        for name, expected_value in expected_metadata.items():
            actual_value = checkpoint.get(name)
            if name == "hidden_sizes" and actual_value is not None:
                actual_value = tuple(actual_value)
            if actual_value != expected_value:
                mismatches.append(
                    f"{name}: checkpoint={actual_value!r}, "
                    f"current={expected_value!r}"
                )
        if mismatches:
            raise ValueError(
                "Checkpoint metadata does not match this MLP/HEOM problem: "
                + "; ".join(mismatches)
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded MLP checkpoint: {args.checkpoint_in}")

    if args.epochs:
        training_config = TrainingConfig(
            t_start=0.0,
            t_stop=args.t_stop,
            epochs=args.epochs,
            collocation_points=args.collocation_points,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip_norm,
            resample_each_epoch=not args.fixed_collocation,
            seed=args.seed,
            log_every=args.log_every,
        )
        training_result = train_mlp(model, objective, training_config)
        print(
            f"MLP training: {training_result.elapsed_seconds:.3f} s; "
            f"final loss={training_result.final.total:.6e}"
        )

    if args.checkpoint_out is not None:
        checkpoint_path = Path(args.checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_state_dict": model.state_dict(),
                "hierarchy_fingerprint": model.hierarchy_fingerprint,
                "hidden_sizes": model.hidden_sizes,
                "activation": model.activation_name,
                "dtype": args.dtype,
            },
            checkpoint_path,
        )
        print(f"Saved MLP checkpoint: {checkpoint_path}")
    return model


def run_mlp_solver(model, t_eval):
    """Evaluate a trained MLP and return its root sigma-z trajectory."""
    start = perf_counter()
    result = solve_mlp(model, t_eval)
    elapsed = perf_counter() - start
    print(f"MLP trajectory evaluation: {elapsed:.3f} s")
    return np.real(result.expectation(sigmaz().full()))


def plot_trajectories(
    t_eval,
    sz_lindbladian,
    sz_sparse,
    sz_mlp,
    *,
    depth,
    show=True,
    output=None,
):
    """Plot exactly the three requested trajectories."""
    _, axis = plt.subplots(dpi=200)
    axis.plot(
        t_eval,
        sz_lindbladian,
        "b-",
        linewidth=1.6,
        label="Lindbladian numerics",
    )
    axis.plot(
        t_eval,
        sz_sparse,
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=rf"Sparse normalized HEOM, $L={depth}$",
    )
    axis.plot(
        t_eval,
        sz_mlp,
        color="#7B2CBF",
        linestyle="--",
        linewidth=1.5,
        label="MLP solver",
    )
    axis.set_title(rf"$g={G / W0:g}\,\omega_0$", fontsize=14)
    axis.set_xlabel(r"$t$", fontsize=14)
    axis.set_ylabel(r"$S_z$", fontsize=14)
    axis.legend(fontsize=11)
    axis.grid(True, alpha=0.3)
    axis.tick_params(axis="both", labelsize=13)
    axis.figure.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        axis.figure.savefig(output, dpi=200)
        print(f"Saved trajectory plot: {output}")
    if show:
        plt.show()
    return axis


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--t-stop", type=float, default=100.0)
    parser.add_argument("--n-times", type=int, default=1_000)
    parser.add_argument("--cavity-dimension", type=int, default=CAVITY_DIMENSION)
    parser.add_argument("--epochs", type=int, default=2_000)
    parser.add_argument("--collocation-points", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument(
        "--activation",
        choices=("tanh", "gelu", "silu", "relu"),
        default="tanh",
    )
    parser.add_argument("--dynamics-weight", type=float, default=1.0)
    parser.add_argument(
        "--initial-weight",
        type=float,
        help="defaults to the collocation count, balancing Eq. (12)",
    )
    parser.add_argument(
        "--trace-weight",
        type=float,
        help="defaults to 1/collocation count, balancing Eq. (13)",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--fixed-collocation", action="store_true")
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    return parser


def _validate_arguments(parser, args):
    for name in ("depth", "n_times", "cavity_dimension", "collocation_points", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.epochs < 0:
        parser.error("--epochs must be non-negative")
    if args.epochs == 0 and args.checkpoint_in is None:
        parser.error("--epochs 0 requires --checkpoint-in")
    if not np.isfinite(args.t_stop) or args.t_stop <= 0:
        parser.error("--t-stop must be finite and positive")
    if args.n_times < 2:
        parser.error("--n-times must be at least 2")


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
    if args.no_show:
        # ``--no-show`` must be usable on headless workers even when the
        # process-wide Matplotlib default points at an unavailable GUI backend.
        plt.switch_backend("Agg")
    t_eval = np.linspace(0.0, args.t_stop, args.n_times)

    sz_lindbladian = run_lindbladian(
        t_eval,
        cavity_dimension=args.cavity_dimension,
    )
    hierarchy, rho0, liouvillian = build_normalized_hard_heom(args.depth)
    sz_sparse = run_sparse_numerics(
        hierarchy,
        rho0,
        liouvillian,
        t_eval,
    )
    model = build_and_train_mlp(hierarchy, rho0, liouvillian, args)
    sz_mlp = run_mlp_solver(model, t_eval)

    print(
        "Max |sparse HEOM - Lindbladian|: "
        f"{np.max(np.abs(sz_sparse - sz_lindbladian)):.3e}"
    )
    print(
        "Max |MLP - sparse HEOM|: "
        f"{np.max(np.abs(sz_mlp - sz_sparse)):.3e}"
    )
    plot_trajectories(
        t_eval,
        sz_lindbladian,
        sz_sparse,
        sz_mlp,
        depth=args.depth,
        show=not args.no_show,
        output=args.output,
    )
    return {
        "time": t_eval,
        "lindbladian": sz_lindbladian,
        "sparse_heom": sz_sparse,
        "mlp": sz_mlp,
        "model": model,
    }


if __name__ == "__main__":
    main()
