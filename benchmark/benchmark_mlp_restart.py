"""Test MLP ADO predictions by restarting numerical HEOM propagation.

Run ``python -m benchmark.benchmark_mlp_restart`` to select a saved model.
Mode A restarts from its full predicted state; mode B replaces tiers
0..lower_tier_cutoff with the numerical reference before restarting.
Only the restart state is evaluated by the MLP, not the later trajectory.
"""

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from benchmark.benchmark_mlp import (
    build_benchmark_time_grids,
    build_normalized_hard_heom,
    load_mlp,
    mean_absolute_error,
    resolve_default_model_path,
    run_mlp_solver,
    run_sparse_numerics,
)
from experiment_parameters import MLP, MLP_MODEL_PATH, PSEUDOMODE
from training_sequence import (
    load_training_metadata,
    load_training_sequence,
    require_complete_training_checkpoint,
    resolve_model_path,
    training_metadata_path,
)


@dataclass(frozen=True)
class RestartTrajectory:
    mode: str
    label: str
    initial_state: np.ndarray
    state: np.ndarray
    sigma_z: np.ndarray
    sigma_z_mae: float
    full_state_mae: float


@dataclass(frozen=True)
class RestartDiagnostic:
    t: np.ndarray
    reference_state: np.ndarray
    reference_sigma_z: np.ndarray
    predicted_state: np.ndarray
    predicted_sigma_z: float
    trajectories: tuple[RestartTrajectory, ...]


def build_restart_time_grids(
    parameters=PSEUDOMODE, *, restart_time=None, t_stop=None, n_times=None,
):
    """Include the physical anchor time in the continuous reference solve."""
    restart = parameters.t_stop if restart_time is None else float(restart_time)
    stop = (
        restart + parameters.t_stop - parameters.t_start
        if t_stop is None else t_stop
    )
    return build_benchmark_time_grids(
        parameters, t_start=restart, t_stop=stop, n_times=n_times,
    )


def _state_vector(hierarchy, state, name):
    state = np.asarray(state, dtype=np.complex128)
    expected_shape = (hierarchy.nADO * hierarchy.system_size,)
    if state.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{name} must contain only finite values")
    return state


def _validate_lower_tier_cutoff(hierarchy, lower_tier_cutoff):
    if (
        isinstance(lower_tier_cutoff, bool)
        or not isinstance(lower_tier_cutoff, (int, np.integer))
        or not 0 <= lower_tier_cutoff <= hierarchy.L
    ):
        raise ValueError(
            f"lower_tier_cutoff must be an integer between 0 and L={hierarchy.L}"
        )


def build_hybrid_state(hierarchy, reference_state, predicted_state, lower_tier_cutoff):
    """Copy complete normalized ADO matrices; the tier cutoff is inclusive.

    Tier selection also keeps conjugate n/m partners together. No projection,
    trace rescaling, or conversion to unnormalized ADOs is performed.
    A cutoff equal to L gives a purely numerical restart control.
    """
    _validate_lower_tier_cutoff(hierarchy, lower_tier_cutoff)
    reference = _state_vector(hierarchy, reference_state, "reference_state")
    predicted = _state_vector(hierarchy, predicted_state, "predicted_state")
    numerical_ados = np.array([
        hierarchy._tier(node) <= lower_tier_cutoff
        for node in hierarchy.idx_to_node
    ])
    numerical_components = np.repeat(numerical_ados, hierarchy.system_size)
    return np.where(numerical_components, reference, predicted)


def run_restart_diagnostic(
    model, hierarchy, liouvillian, parameters=PSEUDOMODE,
    mlp_parameters=MLP, *, restart_time=None, t_stop=None, n_times=None,
    mode="both", lower_tier_cutoff=0, verbose=False,
):
    """Compare numerical continuations against one uninterrupted reference.

    Both solvers use the same normalized hard-cutoff Liouvillian and tolerances.
    The reference starts at the checkpoint's full initial anchor, not a new
    product state at the restart time. Errors average only the continuation
    samples, including the restart point. E_H averages absolute complex
    component differences over the entire hierarchy (including the root).
    """
    mode = mode.lower()
    if mode not in {"a", "b", "both"}:
        raise ValueError("mode must be A, B, or both")
    if mode in {"b", "both"}:
        _validate_lower_tier_cutoff(hierarchy, lower_tier_cutoff)
    times, reference_times, offset = build_restart_time_grids(
        parameters, restart_time=restart_time, t_stop=t_stop, n_times=n_times,
    )
    # Loading restores this persistent buffer, which may differ from a freshly
    # prepared state (e.g. after a checkpoint transfer or PSD roundoff repair).
    initial_state = _state_vector(
        hierarchy, model.complex_initial_state().detach().cpu().numpy(),
        "checkpoint initial state",
    )
    reference_z, reference_solution = run_sparse_numerics(
        hierarchy, initial_state, liouvillian, reference_times, parameters,
        return_solution=True, verbose=verbose,
    )
    reference_state = reference_solution.y[:, offset:]
    reference_z = reference_z[offset:]
    predicted_z, prediction = run_mlp_solver(
        model, times[:1], mlp_parameters, return_solution=True, verbose=verbose,
    )
    predicted_state = _state_vector(hierarchy, prediction.y[:, 0], "MLP prediction")
    starts = []
    if mode in {"a", "both"}:
        starts.append(("A", "A: full MLP restart", predicted_state.copy()))
    if mode in {"b", "both"}:
        hybrid = build_hybrid_state(
            hierarchy, reference_state[:, 0], predicted_state, lower_tier_cutoff,
        )
        label = f"B: numerical tiers 0..{lower_tier_cutoff}"
        starts.append(("B", label, hybrid))

    trajectories = []
    for branch_mode, label, state in starts:
        sigma_z, solution = run_sparse_numerics(
            hierarchy, state, liouvillian, times, parameters,
            return_solution=True, verbose=verbose,
        )
        trajectories.append(RestartTrajectory(
            mode=branch_mode, label=label, initial_state=state,
            state=solution.y, sigma_z=sigma_z,
            sigma_z_mae=mean_absolute_error(reference_z, sigma_z),
            full_state_mae=mean_absolute_error(reference_state, solution.y),
        ))
    return RestartDiagnostic(
        t=times, reference_state=reference_state, reference_sigma_z=reference_z,
        predicted_state=predicted_state, predicted_sigma_z=float(predicted_z[0]),
        trajectories=tuple(trajectories),
    )


def plot_restart_diagnostic(result, parameters=PSEUDOMODE, *, output=None, show=True):
    """Plot one continuation per panel with the benchmark's two-digit MAEs."""
    figure, axes = plt.subplots(
        len(result.trajectories), 1, squeeze=False, sharex=True,
        figsize=(8, 3.5 * len(result.trajectories)), dpi=200,
    )
    figure.suptitle(
        rf"$g={parameters.g:g}$, $L={parameters.heom_depth}$, "
        rf"$t_\ast={result.t[0]:g}$"
    )
    for axis, trajectory in zip(axes[:, 0], result.trajectories):
        axis.plot(
            result.t, result.reference_sigma_z, color="black", linewidth=1.8,
            label=rf"Reference from $t_0={parameters.t_start:g}$",
        )
        axis.plot(
            result.t, trajectory.sigma_z, "--", color="#7B2CBF",
            label="Numerical continuation",
        )
        axis.set_title(
            trajectory.label + "; "
            + rf"$E_z={trajectory.sigma_z_mae:.1e}$, "
            + rf"$E_H={trajectory.full_state_mae:.1e}$",
            fontsize=11,
        )
        axis.set_ylabel(r"$\langle\sigma_z\rangle$")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=10)
    axes[-1, 0].set_xlabel(r"$t$")
    figure.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=200)
    if show:
        plt.show()
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", type=str.lower, choices=("a", "b", "both"), default="both",
        help="A: full MLP restart; B: hybrid restart (default: both)",
    )
    parser.add_argument(
        "--lower-tier-cutoff", type=int, default=0,
        help="B: numerical tiers 0..cutoff, MLP above; default 0 (RDM only); L gives a numerical control",
    )
    parser.add_argument(
        "--restart-time", type=float,
        help="physical time at which to query the MLP (default: saved training end)",
    )
    parser.add_argument(
        "--t-stop", type=float,
        help="continuation end (default: restart time plus one training-interval length)",
    )
    parser.add_argument("--n-times", type=int, help="output sample count (default: saved n_times)")
    parser.add_argument(
        "--model-path", type=Path,
        help="checkpoint file or model folder; skips the folder dialog",
    )
    parser.add_argument(
        "--sequence", "--config", dest="sequence", type=Path,
        help="resolve a named checkpoint or supply legacy config; saved metadata takes precedence",
    )
    folder_group = parser.add_mutually_exclusive_group()
    folder_group.add_argument(
        "--folder-dialog", dest="folder_dialog", action="store_true", default=True,
        help="select the model folder when no explicit path/sequence is given (default)",
    )
    folder_group.add_argument(
        "--no-folder-dialog", dest="folder_dialog", action="store_false",
        help="load saved_models/mlp without a dialog",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), help="override saved inference device")
    parser.add_argument("--no-show", action="store_true", help="save the plot without displaying it")
    parser.add_argument("--verbose", action="store_true", help="also print solver timings")
    parser.add_argument(
        "--output", type=Path,
        help="extra plot copy; restart_trajectory.png is always saved beside the checkpoint",
    )
    args = parser.parse_args(argv)
    if args.no_show:
        plt.switch_backend("Agg")

    try:
        explicit_sequence = (
            load_training_sequence(args.sequence) if args.sequence is not None else None
        )
        if args.model_path is not None:
            model_path = args.model_path
            if model_path.is_dir():
                model_path = model_path / MLP_MODEL_PATH.name
        elif explicit_sequence is not None:
            model_path = resolve_model_path(explicit_sequence)
        else:
            model_path = resolve_default_model_path(folder_dialog=args.folder_dialog)
        if not model_path.is_file():
            raise FileNotFoundError(f"saved MLP checkpoint not found: {model_path}")
        require_complete_training_checkpoint(model_path)
        if training_metadata_path(model_path).is_file():
            sequence = load_training_metadata(model_path)
        elif explicit_sequence is not None:
            sequence = explicit_sequence
        else:
            raise ValueError(
                "checkpoint has no training metadata; supply its original --sequence "
                "to establish the physical parameters and training interval"
            )
        parameters = sequence.pseudomode
        mlp_parameters = sequence.base_mlp
        if args.device is not None:
            mlp_parameters = replace(mlp_parameters, device=args.device)
        times, _, _ = build_restart_time_grids(
            parameters, restart_time=args.restart_time,
            t_stop=args.t_stop, n_times=args.n_times,
        )
        if args.mode in {"b", "both"} and not 0 <= args.lower_tier_cutoff <= parameters.heom_depth:
            raise ValueError(f"--lower-tier-cutoff must be between 0 and L={parameters.heom_depth}")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(f"Model: {model_path}")
    print(f"Training interval: [{parameters.t_start:g}, {parameters.t_stop:g}]; L={parameters.heom_depth}")
    print(f"Numerical continuation: [{times[0]:g}, {times[-1]:g}] ({times.size} samples)")
    if times[0] > parameters.t_stop:
        print("Note: the MLP restart state is extrapolated beyond its training interval.")
    if args.mode in {"b", "both"}:
        if args.lower_tier_cutoff == parameters.heom_depth:
            print("B: all tiers numerical; this measures the numerical restart discrepancy.")
        else:
            print(
                f"B: numerical tiers 0..{args.lower_tier_cutoff}; "
                f"MLP tiers {args.lower_tier_cutoff + 1}..{parameters.heom_depth}"
            )
    hierarchy, initial_state, liouvillian = build_normalized_hard_heom(parameters, verbose=args.verbose)
    model = load_mlp(
        hierarchy, initial_state, liouvillian=liouvillian,
        mlp_parameters=mlp_parameters, pseudomode_parameters=parameters,
        model_path=model_path, verbose=args.verbose,
    )
    result = run_restart_diagnostic(
        model, hierarchy, liouvillian, parameters, mlp_parameters,
        restart_time=args.restart_time, t_stop=args.t_stop, n_times=args.n_times,
        mode=args.mode, lower_tier_cutoff=args.lower_tier_cutoff, verbose=args.verbose,
    )
    initial_z_error = abs(result.predicted_sigma_z - result.reference_sigma_z[0])
    initial_state_error = mean_absolute_error(result.reference_state[:, 0], result.predicted_state)
    print(f"MLP at restart: |sigma_z error|={initial_z_error:.3e}; mean absolute full state-component error={initial_state_error:.3e}")
    for trajectory in result.trajectories:
        print(f"{trajectory.label} vs reference (continuation interval):")
        print(f"  Mean absolute <sigma_z> error (E_z): {trajectory.sigma_z_mae:.3e}")
        print(f"  Mean absolute full normalized-HEOM state-component error (E_H): {trajectory.full_state_mae:.3e}")

    output = model_path.parent / "restart_trajectory.png"
    figure = plot_restart_diagnostic(result, parameters, output=output, show=False)
    print(f"Saved restart trajectory: {output}")
    if args.output is not None and args.output.resolve() != output.resolve():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=200)
        print(f"Saved additional trajectory: {args.output}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)
    return result


if __name__ == "__main__":
    main()
