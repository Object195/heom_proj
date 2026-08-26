"""Compare Lindblad, sparse HEOM, and a saved Section-III MLP trajectory.

Train the model first with::

    python -m model.train_mlp_model

Then run this benchmark with::

    python -m benchmark.benchmark_mlp
"""

import argparse
import sys
from dataclasses import dataclass
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
from heom.heom_solver import prepare_heom_initial_state, solve_heom
from model import (
    HEOMMLP,
    HEOMPINNLoss,
    solve_mlp,
    state_and_time_derivative,
)
from training_sequence import (
    default_training_sequence,
    load_training_metadata,
    load_training_sequence,
    training_metadata_path,
)


_LINDBLADIAN_MAX_OUTPUT_STEP = 1.0
_LINDBLADIAN_MAX_INTERNAL_STEPS = 100_000


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


@dataclass(frozen=True)
class TierResidualStatistics:
    """Time-aggregated dynamical residual statistics for one HEOM tier."""

    tier: int
    n_ados: int
    mean_squared_residual: float
    max_time_squared_residual: float
    max_component_residual: float


def compute_tier_residual_statistics(
    model,
    hierarchy,
    liouvillian,
    t_eval,
    *,
    batch_size=1_024,
):
    r"""Evaluate ``R = d chi / dt - L chi`` and aggregate it by ADO tier.

    For a two-level system, the reported mean is exactly

    ``E_l = sum_(q,j in l) (||R_U,qj||^2 + ||R_V,qj||^2)/(4 B N_l)``.

    ``max_time_squared_residual`` applies the same ``1/(4 N_l)`` tier
    normalization at each time and takes the maximum over time.
    ``max_component_residual`` is the largest complex-component magnitude
    ``sqrt(R_U^2 + R_V^2)`` in the tier.
    """
    t_eval = np.asarray(t_eval, dtype=np.float64).reshape(-1)
    if t_eval.size == 0:
        raise ValueError("t_eval must contain at least one time")
    if isinstance(batch_size, bool) or not isinstance(
        batch_size, (int, np.integer)
    ) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    tier_to_ados = {}
    for ado_index, node in enumerate(hierarchy.idx_to_node):
        tier = hierarchy._tier(node)
        tier_to_ados.setdefault(tier, []).append(ado_index)
    tier_indices = {
        tier: torch.as_tensor(indices, dtype=torch.long, device=model.device)
        for tier, indices in tier_to_ados.items()
    }
    tier_sums = {tier: 0.0 for tier in tier_to_ados}
    tier_max_times = {tier: 0.0 for tier in tier_to_ados}
    tier_max_components = {tier: 0.0 for tier in tier_to_ados}

    objective = HEOMPINNLoss(
        hierarchy,
        liouvillian=liouvillian,
        dtype=model.dtype,
        device=model.device,
    )
    was_training = model.training
    model.eval()
    start = perf_counter()
    try:
        with torch.enable_grad():
            for first in range(0, t_eval.size, int(batch_size)):
                times = model.prepare_times(t_eval[first : first + batch_size])
                state, time_derivative = state_and_time_derivative(
                    model,
                    times,
                    create_graph=False,
                )
                residual = time_derivative - objective.rhs(state)
                residual_u, residual_v = residual.split(
                    model.state_size,
                    dim=-1,
                )
                component_energy = (
                    residual_u.reshape(
                        -1,
                        model.n_ados,
                        model.system_size,
                    ).square()
                    + residual_v.reshape(
                        -1,
                        model.n_ados,
                        model.system_size,
                    ).square()
                )

                for tier, indices in tier_indices.items():
                    tier_energy = component_energy.index_select(1, indices)
                    time_squared_residual = tier_energy.mean(dim=(1, 2))
                    tier_sums[tier] += time_squared_residual.sum().item()
                    tier_max_times[tier] = max(
                        tier_max_times[tier],
                        time_squared_residual.amax().item(),
                    )
                    tier_max_components[tier] = max(
                        tier_max_components[tier],
                        tier_energy.amax().sqrt().item(),
                    )
    finally:
        model.train(was_training)
    print(f"MLP residual evaluation: {perf_counter() - start:.3f} s")

    return tuple(
        TierResidualStatistics(
            tier=tier,
            n_ados=len(tier_to_ados[tier]),
            mean_squared_residual=tier_sums[tier] / t_eval.size,
            max_time_squared_residual=tier_max_times[tier],
            max_component_residual=tier_max_components[tier],
        )
        for tier in sorted(tier_to_ados)
    )


def _build_lindbladian_solver_grid(
    t_eval,
    *,
    max_output_step=_LINDBLADIAN_MAX_OUTPUT_STEP,
):
    """Add intermediate output times and return requested-sample indices.

    QuTiP's SciPy integrator applies ``nsteps`` between consecutive entries in
    its time list. A later-start calculation with only ``[0, t_start, ...]``
    can therefore fail even though the physical problem is well behaved.
    Splitting every large gap also makes sparse and dense benchmark grids
    equally reliable.
    """
    t_eval = np.asarray(t_eval, dtype=np.float64)
    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("Lindbladian t_eval must be a 1D array of length >= 2")
    if not np.isfinite(t_eval).all():
        raise ValueError("Lindbladian t_eval must contain only finite times")
    if t_eval[0] < 0.0:
        raise ValueError("Lindbladian t_eval cannot start before physical t=0")
    if np.any(np.diff(t_eval) <= 0.0):
        raise ValueError("Lindbladian t_eval must be strictly increasing")
    if not np.isfinite(max_output_step) or max_output_step <= 0.0:
        raise ValueError("max_output_step must be finite and positive")

    solver_times = [0.0]
    requested_indices = []
    for requested_time in t_eval:
        interval_start = solver_times[-1]
        if requested_time > interval_start:
            interval_count = max(
                1,
                int(np.ceil((requested_time - interval_start) / max_output_step)),
            )
            solver_times.extend(
                np.linspace(
                    interval_start,
                    requested_time,
                    interval_count + 1,
                )[1:]
            )
        requested_indices.append(len(solver_times) - 1)
    return (
        np.asarray(solver_times, dtype=np.float64),
        np.asarray(requested_indices, dtype=np.int64),
    )


def run_lindbladian(t_eval, parameters=PSEUDOMODE):
    """Propagate the explicit damped-cavity Lindblad reference."""
    solver_t_eval, requested_indices = _build_lindbladian_solver_grid(t_eval)
    if solver_t_eval.size > requested_indices.size:
        print(
            "Lindbladian integration grid: "
            f"{solver_t_eval.size} points for {requested_indices.size} "
            "requested samples"
        )
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
        solver_t_eval,
        c_ops=collapse_operators,
        e_ops={"sz": sz_full},
        options={
            "method": "bdf",
            "nsteps": _LINDBLADIAN_MAX_INTERNAL_STEPS,
            "rtol": parameters.rtol,
            "atol": parameters.atol,
        },
    )
    print(f"Lindbladian propagation: {perf_counter() - start:.3f} s")
    expectation = np.asarray(result.e_data["sz"]).real
    return expectation[requested_indices]


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
    start = perf_counter()
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
            f"t={parameters.t_start:g}: {perf_counter() - start:.3f} s"
        )
    return hierarchy, initial_heom_state, liouvillian


def run_sparse_numerics(
    hierarchy,
    initial_heom_state,
    liouvillian,
    t_eval,
    parameters=PSEUDOMODE,
):
    start = perf_counter()
    result = solve_heom(
        hierarchy,
        None,
        t_eval,
        initial_state=initial_heom_state,
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
    initial_heom_state,
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
        initial_heom_state=initial_heom_state,
        t_start=pseudomode_parameters.t_start,
        t_stop=pseudomode_parameters.t_stop,
        activation=mlp_parameters.activation,
        time_switch=mlp_parameters.time_switch,
        switch_time_constant=mlp_parameters.switch_time_constant,
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
    hierarchy, initial_heom_state, liouvillian = (
        build_normalized_hard_heom(pseudomode)
    )
    sparse_heom = run_sparse_numerics(
        hierarchy,
        initial_heom_state,
        liouvillian,
        reference_t_eval,
        pseudomode,
    )[reference_offset:]
    model = load_mlp(
        hierarchy,
        initial_heom_state,
        mlp_parameters=mlp_parameters,
        pseudomode_parameters=pseudomode,
        model_path=args.model_path,
    )
    mlp = run_mlp_solver(
        model,
        t_eval,
        mlp_parameters,
    )
    tier_residuals = compute_tier_residual_statistics(
        model,
        hierarchy,
        liouvillian,
        t_eval,
        batch_size=mlp_parameters.inference_batch_size,
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
    print("Tier-resolved MLP dynamical residuals:")
    for statistics in tier_residuals:
        print(
            f"  tier {statistics.tier} (N={statistics.n_ados}): "
            f"E_res_mean={statistics.mean_squared_residual:.6e}  "
            f"E_res_max_t={statistics.max_time_squared_residual:.6e}  "
            f"max|R|={statistics.max_component_residual:.6e}"
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
