"""Compare Lindblad, sparse HEOM, and a saved Section-III MLP trajectory.

Train the model first with::

    python -m model.train_mlp_model

Then run this benchmark with::

    python -m benchmark.benchmark_mlp
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from qutip import basis, destroy, mesolve, qeye, sigmax, sigmaz, tensor

from experiment_parameters import MLP, MLP_MODEL_PATH, PSEUDOMODE
from heom.heom_rep import heom_state
from heom.heom_solver import solve_heom
from model import HEOMMLP, solve_mlp
from training_sequence import (
    default_training_sequence,
    load_training_metadata,
    load_training_sequence,
    training_metadata_path,
)


def build_benchmark_time_grids(
    parameters=PSEUDOMODE,
    *,
    t_start=None,
    t_stop=None,
    n_times=None,
):
    """Build MLP output times and reference times from the physical initial time.

    A benchmark may start after the model's initial time.  The numerical
    references must still propagate from that initial time instead of treating
    the first requested output time as a new initial condition.  The returned
    offset selects the requested benchmark samples from the reference result.
    """
    initial_time = float(parameters.t_start)
    benchmark_start = initial_time if t_start is None else float(t_start)
    benchmark_stop = (
        float(parameters.t_stop) if t_stop is None else float(t_stop)
    )
    benchmark_count = parameters.n_times if n_times is None else n_times

    if not np.isfinite(benchmark_start):
        raise ValueError("benchmark t_start must be finite")
    if not np.isfinite(benchmark_stop):
        raise ValueError("benchmark t_stop must be finite")
    if benchmark_start < initial_time:
        raise ValueError(
            "benchmark t_start cannot precede the model initial time "
            f"{initial_time:g}"
        )
    if benchmark_stop <= benchmark_start:
        raise ValueError("benchmark t_stop must be greater than t_start")
    if (
        isinstance(benchmark_count, bool)
        or not isinstance(benchmark_count, (int, np.integer))
        or benchmark_count < 2
    ):
        raise ValueError("benchmark n_times must be an integer of at least 2")

    t_eval = np.linspace(
        benchmark_start,
        benchmark_stop,
        int(benchmark_count),
    )
    if benchmark_start == initial_time:
        return t_eval, t_eval, 0

    reference_t_eval = np.concatenate(([initial_time], t_eval))
    return t_eval, reference_t_eval, 1


def run_lindbladian(t_eval, parameters=PSEUDOMODE):
    """Propagate the explicit damped-cavity Lindblad reference."""
    annihilation = tensor(qeye(2), destroy(parameters.cavity_dimension))
    sz_full = tensor(sigmaz(), qeye(parameters.cavity_dimension))
    sx_full = tensor(sigmax(), qeye(parameters.cavity_dimension))
    h_system = 0.5 * parameters.delta * sz_full
    h_system += 0.5 * parameters.v * sx_full
    h_cavity = parameters.w0 * (annihilation.dag() @ annihilation)
    h_interaction = (
        parameters.g * (sz_full @ (annihilation + annihilation.dag()))
    )
    h_total = h_system + h_cavity + h_interaction
    collapse_operators = [np.sqrt(parameters.gamma) * annihilation]
    psi0 = tensor(
        basis(2, 0),
        basis(parameters.cavity_dimension, 0),
    )

    start = perf_counter()
    result = mesolve(
        h_total,
        psi0,
        t_eval,
        c_ops=collapse_operators,
        e_ops={"sz": sz_full},
    )
    print(f"Lindbladian propagation: {perf_counter() - start:.3f} s")
    return np.asarray(result.e_data["sz"]).real


def build_normalized_hard_heom(parameters=PSEUDOMODE):
    """Build the normalized, hard-truncated free-pole HEOM."""
    h_system = 0.5 * parameters.delta * sigmaz().full()
    h_system += 0.5 * parameters.v * sigmax().full()
    rho0 = basis(2, 0).proj().full()
    frequencies = np.array(
        [0.5 * parameters.gamma + 1j * parameters.w0],
        dtype=np.complex128,
    )
    coefficients = np.array([parameters.g**2], dtype=np.complex128)
    hierarchy = heom_state(
        K=0,
        L=parameters.heom_depth,
        H_s=h_system,
        H_c=sigmaz().full(),
        C_list=coefficients,
        gamma_list=frequencies,
    )

    start = perf_counter()
    liouvillian = hierarchy.build_Liouvillian(
        markovian_terminator=False,
        normalized=True,
    )
    print(
        "Normalized hard-cutoff HEOM construction "
        f"(L={parameters.heom_depth}, ADOs={hierarchy.nADO}, "
        f"shape={liouvillian.shape}): {perf_counter() - start:.3f} s"
    )
    return hierarchy, rho0, liouvillian


def run_sparse_numerics(
    hierarchy,
    rho0,
    liouvillian,
    t_eval,
    parameters=PSEUDOMODE,
):
    start = perf_counter()
    result = solve_heom(
        hierarchy,
        rho0,
        t_eval,
        liouvillian=liouvillian,
        method="BDF",
        rtol=parameters.rtol,
        atol=parameters.atol,
    )
    print(
        f"Sparse HEOM propagation: {perf_counter() - start:.3f} s "
        f"[nfev={result.nfev}, njev={result.njev}, nlu={result.nlu}]"
    )
    return np.real(result.expectation(sigmaz().full()))


def load_mlp(
    hierarchy,
    rho0,
    *,
    mlp_parameters=MLP,
    pseudomode_parameters=PSEUDOMODE,
    model_path=MLP_MODEL_PATH,
):
    """Rebuild the configured architecture and load its trained weights."""
    device = torch.device(mlp_parameters.device)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=mlp_parameters.hidden_sizes,
        rho0=rho0,
        t_start=pseudomode_parameters.t_start,
        t_stop=pseudomode_parameters.t_stop,
        activation=mlp_parameters.activation,
        dtype=getattr(torch, mlp_parameters.dtype),
        device=device,
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    print(f"Loaded MLP model: {model_path}")
    return model


def run_mlp_solver(model, t_eval, parameters=MLP):
    """Evaluate at physical times; ``HEOMMLP`` normalizes them internally."""
    start = perf_counter()
    result = solve_mlp(
        model,
        t_eval,
        batch_size=parameters.inference_batch_size,
    )
    print(f"MLP trajectory evaluation: {perf_counter() - start:.3f} s")
    return np.real(result.expectation(sigmaz().full()))


def plot_trajectories(
    t_eval,
    lindbladian,
    sparse_heom,
    mlp,
    *,
    show,
    output,
    parameters=PSEUDOMODE,
):
    _, axis = plt.subplots(dpi=200)
    axis.plot(t_eval, lindbladian, "b-", label="Lindbladian numerics")
    axis.plot(
        t_eval,
        sparse_heom,
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=rf"Sparse normalized HEOM, $L={parameters.heom_depth}$",
    )
    axis.plot(
        t_eval,
        mlp,
        color="#7B2CBF",
        linestyle="--",
        label="MLP solver",
    )
    if t_eval[0] <= parameters.t_stop < t_eval[-1]:
        axis.axvline(
            parameters.t_stop,
            color="0.45",
            linestyle="-.",
            linewidth=1.0,
            label="MLP training horizon",
        )
    axis.set_title(
        rf"$g={parameters.g / parameters.w0:g}\,\omega_0$",
        fontsize=14,
    )
    axis.set_xlabel(r"$t$", fontsize=14)
    axis.set_ylabel(r"$S_z$", fontsize=14)
    axis.legend(fontsize=11)
    axis.grid(True, alpha=0.3)
    axis.figure.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        axis.figure.savefig(output, dpi=200)
    if show:
        plt.show()
    return axis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--t-start",
        type=float,
        help=(
            "first physical benchmark time (default: training start); "
            "cannot precede the model initial time"
        ),
    )
    parser.add_argument(
        "--t-stop",
        type=float,
        help=(
            "last physical benchmark time (default: training stop); values "
            "past the training stop test forward extrapolation"
        ),
    )
    parser.add_argument(
        "--n-times",
        type=int,
        help="number of benchmark output times (default: configured n_times)",
    )
    parser.add_argument(
        "--sequence",
        "--config",
        dest="sequence",
        type=Path,
        help="training-sequence TOML used to build the saved model",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MLP_MODEL_PATH,
        help="saved MLP state dictionary to evaluate",
    )
    args = parser.parse_args(argv)
    if args.no_show:
        plt.switch_backend("Agg")

    try:
        if args.sequence is not None:
            sequence = load_training_sequence(args.sequence)
        elif training_metadata_path(args.model_path).is_file():
            sequence = load_training_metadata(args.model_path)
        else:
            sequence = default_training_sequence()
        t_eval, reference_t_eval, reference_offset = build_benchmark_time_grids(
            sequence.pseudomode,
            t_start=args.t_start,
            t_stop=args.t_stop,
            n_times=args.n_times,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    pseudomode = sequence.pseudomode
    mlp_parameters = sequence.base_mlp
    print(
        f"MLP training interval: [{pseudomode.t_start:g}, "
        f"{pseudomode.t_stop:g}]"
    )
    print(
        f"Benchmark interval: [{t_eval[0]:g}, {t_eval[-1]:g}] "
        f"({t_eval.size} points)"
    )
    if reference_offset:
        print(
            "Reference solvers warm up from the model initial time "
            f"t={pseudomode.t_start:g}"
        )

    lindbladian = run_lindbladian(reference_t_eval, pseudomode)[
        reference_offset:
    ]
    hierarchy, rho0, liouvillian = build_normalized_hard_heom(pseudomode)
    sparse_heom = run_sparse_numerics(
        hierarchy,
        rho0,
        liouvillian,
        reference_t_eval,
        pseudomode,
    )[reference_offset:]
    mlp = run_mlp_solver(
        load_mlp(
            hierarchy,
            rho0,
            mlp_parameters=mlp_parameters,
            pseudomode_parameters=pseudomode,
            model_path=args.model_path,
        ),
        t_eval,
        mlp_parameters,
    )

    print(
        "Max |sparse HEOM - Lindbladian|: "
        f"{np.max(np.abs(sparse_heom - lindbladian)):.3e}"
    )
    mlp_error = np.abs(mlp - sparse_heom)
    print(f"Max |MLP - sparse HEOM|: {np.max(mlp_error):.3e}")
    trained_mask = t_eval <= pseudomode.t_stop
    extrapolation_mask = t_eval > pseudomode.t_stop
    if np.any(trained_mask) and np.any(extrapolation_mask):
        print(
            "Max |MLP - sparse HEOM| within training horizon: "
            f"{np.max(mlp_error[trained_mask]):.3e}"
        )
    if np.any(extrapolation_mask):
        print(
            "Max |MLP - sparse HEOM| beyond training horizon: "
            f"{np.max(mlp_error[extrapolation_mask]):.3e}"
        )
    plot_trajectories(
        t_eval,
        lindbladian,
        sparse_heom,
        mlp,
        show=not args.no_show,
        output=args.output,
        parameters=pseudomode,
    )


if __name__ == "__main__":
    main()
