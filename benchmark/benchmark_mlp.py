"""Compare Lindblad, sparse HEOM, and a saved Section-III MLP trajectory.

Train the model first with::

    python -m model.train_mlp_model

Then run this benchmark with::

    python -m benchmark.benchmark_mlp

Add ``--tier-diagnostics`` to compute relative dynamical residuals, true ADO
activity, direct MLP ADO errors, and downward forcing into lower tiers.
Console reporting is enabled explicitly with ``--verbose``.
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
    compute_heom_dynamical_scales,
    solve_mlp,
    state_and_time_derivative,
)
from training_sequence import (
    default_training_sequence,
    load_training_metadata,
    load_training_sequence,
    require_complete_training_checkpoint,
    resolve_model_path,
    training_metadata_path,
)


_LINDBLADIAN_MAX_OUTPUT_STEP = 1.0
_LINDBLADIAN_MAX_INTERNAL_STEPS = 100_000


def mean_absolute_error(reference, approximation):
    """Return the elementwise mean absolute error for equally shaped arrays."""
    reference = np.asarray(reference)
    approximation = np.asarray(approximation)
    if reference.shape != approximation.shape:
        raise ValueError(
            "reference and approximation must have the same shape; "
            f"got {reference.shape} and {approximation.shape}"
        )
    if reference.size == 0:
        raise ValueError("cannot compute a mean absolute error for empty arrays")
    return float(np.mean(np.abs(approximation - reference)))


def select_model_folder(initial_directory):
    """Open a native folder picker and return the selected model directory."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError(
            "the model-folder dialog requires tkinter; use "
            "--no-folder-dialog or --model-path instead"
        ) from error

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        selected = filedialog.askdirectory(
            parent=root,
            title="Select the folder containing mlp_state_dict.pt",
            initialdir=str(Path(initial_directory).resolve()),
            mustexist=True,
        )
    except tk.TclError as error:
        raise RuntimeError(
            "could not open the model-folder dialog; use "
            "--no-folder-dialog or --model-path instead"
        ) from error
    finally:
        if root is not None:
            root.destroy()

    return Path(selected) if selected else None


def resolve_default_model_path(*, folder_dialog=True, folder_selector=None):
    """Resolve the default checkpoint, optionally using a folder dialog."""
    if not folder_dialog:
        return MLP_MODEL_PATH
    selector = (
        select_model_folder if folder_selector is None else folder_selector
    )
    model_folder = selector(MLP_MODEL_PATH.parent.parent)
    if model_folder is None:
        raise ValueError("no model folder was selected")
    return Path(model_folder) / MLP_MODEL_PATH.name


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
    """Time-mean residual statistics for one HEOM tier."""

    tier: int
    n_ados: int
    mean_squared_residual: float
    mean_squared_dynamical_activity: float | None = None
    mean_relative_rms_residual: float | None = None


@dataclass(frozen=True)
class TierADOStatistics:
    """True ADO activity and direct MLP state error for one HEOM tier."""

    tier: int
    n_ados: int
    mean_squared_activity: float
    mean_squared_error: float
    mean_relative_rms_error: float


@dataclass(frozen=True)
class TierDownwardForcingStatistics:
    """Downward forcing from one source tier into the tier below it."""

    source_tier: int
    target_tier: int
    n_source_ados: int
    n_target_ados: int
    mean_squared_forcing: float
    mean_squared_error: float
    mean_relative_rms_error: float


def _tier_to_ados(hierarchy):
    tier_to_ados = {}
    for ado_index, node in enumerate(hierarchy.idx_to_node):
        tier = hierarchy._tier(node)
        tier_to_ados.setdefault(tier, []).append(ado_index)
    return tier_to_ados


def _coerce_state_trajectory(
    trajectory,
    hierarchy,
    *,
    name,
    n_times=None,
):
    """Validate a complex HEOM trajectory with shape ``(state, time)``."""
    trajectory = np.asarray(trajectory, dtype=np.complex128)
    expected_state_size = hierarchy.nADO * hierarchy.system_size
    expected_shape = (expected_state_size, n_times)
    if trajectory.ndim != 2 or trajectory.shape[0] != expected_state_size:
        suffix = (
            f" and {n_times} time points" if n_times is not None else ""
        )
        raise ValueError(
            f"{name} must have shape ({expected_state_size}, B){suffix}"
        )
    if n_times is not None and trajectory.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{name} must contain only finite values")
    return trajectory


def _mean_per_ado_relative_rms(numerator, denominator):
    """Average per-ADO RMS ratios without cross-ADO scale weighting."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    ratios = np.empty_like(numerator)
    positive = denominator > 0.0
    ratios[positive] = np.sqrt(numerator[positive] / denominator[positive])
    ratios[~positive] = np.where(
        numerator[~positive] == 0.0,
        0.0,
        np.inf,
    )
    return float(ratios.mean())


def _ado_component_indices(ado_indices, system_size):
    ado_indices = np.asarray(ado_indices, dtype=np.int64)
    components = np.arange(system_size, dtype=np.int64)
    return (ado_indices[:, None] * system_size + components).reshape(-1)


def compute_tier_ado_statistics(
    hierarchy,
    reference_state,
    predicted_state,
):
    r"""Aggregate exact ADO activity and direct MLP ADO error by tier.

    Both trajectories use the ``(nADO * system_size, n_times)`` layout of
    :class:`heom.heom_solver.HEOMSolution` and :class:`model.MLPSolution`.
    For every ADO ``q``, the relative RMS error is computed first as

    ``sqrt(mean(|MLP_q-reference_q|^2) / mean(|reference_q|^2))``.

    These per-ADO ratios are then averaged within each tier.  An arbitrary
    nonzero scaling applied separately to each ADO cancels before the tier
    average, so this statistic is invariant to normalized-versus-unnormalized
    HEOM similarity scaling.  An exactly zero reference with nonzero error has
    infinite relative error; zero reference and zero error contributes zero.
    """
    reference_state = _coerce_state_trajectory(
        reference_state,
        hierarchy,
        name="reference_state",
    )
    predicted_state = _coerce_state_trajectory(
        predicted_state,
        hierarchy,
        name="predicted_state",
        n_times=reference_state.shape[1],
    )
    if reference_state.shape[1] == 0:
        raise ValueError("state trajectories must contain at least one time")

    n_times = reference_state.shape[1]
    state_shape = (n_times, hierarchy.nADO, hierarchy.system_size)
    reference = reference_state.T.reshape(state_shape)
    error = predicted_state.T.reshape(state_shape) - reference
    reference_energy = np.abs(reference) ** 2
    error_energy = np.abs(error) ** 2
    tier_to_ados = _tier_to_ados(hierarchy)
    ado_activity = reference_energy.mean(axis=(0, 2))
    ado_error = error_energy.mean(axis=(0, 2))
    return tuple(
        TierADOStatistics(
            tier=tier,
            n_ados=len(indices),
            mean_squared_activity=float(ado_activity[indices].mean()),
            mean_squared_error=float(ado_error[indices].mean()),
            mean_relative_rms_error=_mean_per_ado_relative_rms(
                ado_error[indices],
                ado_activity[indices],
            ),
        )
        for tier, indices in sorted(tier_to_ados.items())
    )


def compute_tier_downward_forcing_statistics(
    hierarchy,
    liouvillian,
    reference_state,
    predicted_state,
):
    r"""Measure how each HEOM tier drives the tier immediately below it.

    For source tier ``l``, the exact net downward forcing is

    ``F_l(t) = L_(l-1,l) @ chi_l(t)``.

    The block multiplication is performed before taking a norm so all source
    ADOs that feed the same target ADO are summed coherently, exactly as in the
    HEOM equation.  The MLP forcing error is

    ``delta F_l = L_(l-1,l) @ (chi_mlp_l - chi_reference_l)``.

    Absolute values are time/component mean-square magnitudes averaged over
    target ADOs.  The relative value is formed separately for every target
    ADO as ``RMS(delta F_p) / RMS(F_p)`` and then averaged over tier ``l-1``.
    Under an arbitrary diagonal ADO similarity scaling, both forces acquire
    the same target-ADO factor, which cancels in this per-target ratio.
    """
    reference_state = _coerce_state_trajectory(
        reference_state,
        hierarchy,
        name="reference_state",
    )
    predicted_state = _coerce_state_trajectory(
        predicted_state,
        hierarchy,
        name="predicted_state",
        n_times=reference_state.shape[1],
    )
    if reference_state.shape[1] == 0:
        raise ValueError("state trajectories must contain at least one time")
    state_size = hierarchy.nADO * hierarchy.system_size
    if getattr(liouvillian, "shape", None) != (state_size, state_size):
        raise ValueError(
            "liouvillian must have shape "
            f"({state_size}, {state_size})"
        )

    tier_to_ados = _tier_to_ados(hierarchy)
    state_error = predicted_state - reference_state
    statistics = []
    for source_tier in sorted(tier_to_ados):
        if source_tier == 0:
            continue
        target_tier = source_tier - 1
        source_ados = tier_to_ados[source_tier]
        target_ados = tier_to_ados.get(target_tier)
        if target_ados is None:
            continue
        source_components = _ado_component_indices(
            source_ados,
            hierarchy.system_size,
        )
        target_components = _ado_component_indices(
            target_ados,
            hierarchy.system_size,
        )
        downward_block = liouvillian[
            np.ix_(target_components, source_components)
        ]
        exact_forcing = np.asarray(
            downward_block @ reference_state[source_components, :]
        )
        forcing_error = np.asarray(
            downward_block @ state_error[source_components, :]
        )
        forcing_shape = (
            len(target_ados),
            hierarchy.system_size,
            reference_state.shape[1],
        )
        exact_energy = np.abs(exact_forcing.reshape(forcing_shape)) ** 2
        error_energy = np.abs(forcing_error.reshape(forcing_shape)) ** 2
        target_forcing_mean = exact_energy.mean(axis=(1, 2))
        target_error_mean = error_energy.mean(axis=(1, 2))
        statistics.append(
            TierDownwardForcingStatistics(
                source_tier=source_tier,
                target_tier=target_tier,
                n_source_ados=len(source_ados),
                n_target_ados=len(target_ados),
                mean_squared_forcing=float(target_forcing_mean.mean()),
                mean_squared_error=float(target_error_mean.mean()),
                mean_relative_rms_error=_mean_per_ado_relative_rms(
                    target_error_mean,
                    target_forcing_mean,
                ),
            )
        )
    return tuple(statistics)


def compute_tier_residual_statistics(
    model,
    hierarchy,
    liouvillian,
    t_eval,
    *,
    reference_state=None,
    batch_size=1_024,
    verbose=False,
):
    r"""Evaluate ``R = d chi / dt - L chi`` and aggregate it by ADO tier.

    For a two-level system, the reported mean is exactly

    ``E_l = sum_(q,j in l) (||R_U,qj||^2 + ||R_V,qj||^2)/(4 B N_l)``.

    When the exact ``reference_state`` is supplied, its exact HEOM derivative
    ``L @ reference_state`` supplies the dynamical scale.  The relative RMS is
    first calculated independently for every ADO,

    ``epsilon_q = sqrt(mean(|R_q|^2) / mean(|L chi_reference|_q^2))``,

    and then averaged within each tier.  This prevents differently scaled ADOs
    from reweighting the tier ratio.  Only time-mean statistics are retained;
    worst-time and peak-component statistics are intentionally omitted because
    they are unstable near zero crossings and repeated the same tier trend.
    """
    t_eval = np.asarray(t_eval, dtype=np.float64).reshape(-1)
    if t_eval.size == 0:
        raise ValueError("t_eval must contain at least one time")
    if isinstance(batch_size, bool) or not isinstance(
        batch_size, (int, np.integer)
    ) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    tier_to_ados = _tier_to_ados(hierarchy)
    ado_residual_sums = np.zeros(hierarchy.nADO, dtype=np.float64)
    ado_dynamical_mean = None

    if reference_state is not None:
        reference_state = _coerce_state_trajectory(
            reference_state,
            hierarchy,
            name="reference_state",
            n_times=t_eval.size,
        )
        reference_derivative = liouvillian @ reference_state
        reference_derivative = np.asarray(reference_derivative)
        derivative_energy = np.abs(
            reference_derivative.T.reshape(
                t_eval.size,
                hierarchy.nADO,
                hierarchy.system_size,
            )
        ) ** 2
        ado_dynamical_mean = derivative_energy.mean(axis=(0, 2))

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

                ado_residual_sums += (
                    component_energy.sum(dim=(0, 2)).detach().cpu().numpy()
                )
    finally:
        model.train(was_training)
    if verbose:
        print(f"MLP residual evaluation: {perf_counter() - start:.3f} s")

    ado_residual_mean = ado_residual_sums / (
        t_eval.size * hierarchy.system_size
    )
    return tuple(
        TierResidualStatistics(
            tier=tier,
            n_ados=len(indices),
            mean_squared_residual=float(ado_residual_mean[indices].mean()),
            mean_squared_dynamical_activity=(
                None
                if ado_dynamical_mean is None
                else float(ado_dynamical_mean[indices].mean())
            ),
            mean_relative_rms_residual=(
                None
                if ado_dynamical_mean is None
                else _mean_per_ado_relative_rms(
                    ado_residual_mean[indices],
                    ado_dynamical_mean[indices],
                )
            ),
        )
        for tier, indices in sorted(tier_to_ados.items())
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


def run_lindbladian(t_eval, parameters=PSEUDOMODE, *, verbose=False):
    """Propagate the explicit damped-cavity Lindblad reference."""
    solver_t_eval, requested_indices = _build_lindbladian_solver_grid(t_eval)
    if verbose and solver_t_eval.size > requested_indices.size:
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
    if verbose:
        print(f"Lindbladian propagation: {perf_counter() - start:.3f} s")
    expectation = np.asarray(result.e_data["sz"]).real
    return expectation[requested_indices]


def build_normalized_hard_heom(parameters=PSEUDOMODE, *, verbose=False):
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
    if verbose:
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
    if verbose and parameters.t_start > 0.0:
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
    *,
    return_solution=False,
    verbose=False,
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
    if verbose:
        print(
            f"Sparse HEOM propagation: {perf_counter() - start:.3f} s "
            f"[nfev={result.nfev}, njev={result.njev}, nlu={result.nlu}]"
        )
    expectation = np.real(result.expectation(sigmaz().full()))
    if return_solution:
        return expectation, result
    return expectation


def load_mlp(
    hierarchy,
    initial_heom_state,
    *,
    liouvillian=None,
    mlp_parameters=MLP,
    pseudomode_parameters=PSEUDOMODE,
    model_path=MLP_MODEL_PATH,
    verbose=False,
):
    """Rebuild the configured architecture and load its trained weights."""
    require_complete_training_checkpoint(model_path)
    metadata_path = training_metadata_path(model_path)
    saved_positive_rdm = (
        load_training_metadata(model_path).base_mlp.positive_rdm_ansatz
        if metadata_path.is_file()
        else False
    )
    if saved_positive_rdm != mlp_parameters.positive_rdm_ansatz:
        raise ValueError(
            "checkpoint positive_rdm_ansatz does not match the requested "
            "benchmark model; use the checkpoint's saved parameterization"
        )
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
        normalize_hierarchy_coordinates=(
            mlp_parameters.normalize_hierarchy_coordinates
        ),
        positive_rdm_ansatz=mlp_parameters.positive_rdm_ansatz,
        dtype=getattr(torch, mlp_parameters.dtype),
        device=device,
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    if mlp_parameters.ansatz_scale_normalization:
        if liouvillian is None:
            liouvillian = hierarchy.build_Liouvillian(
                markovian_terminator=False,
                normalized=True,
            )
        scales = compute_heom_dynamical_scales(
            hierarchy,
            model.complex_initial_state().detach().cpu().numpy(),
            liouvillian,
            t_start=pseudomode_parameters.t_start,
            t_stop=pseudomode_parameters.t_stop,
            tier_normalized=mlp_parameters.tier_normalized_loss,
            lower_tier_cutoff=mlp_parameters.lower_tier_loss_cutoff,
            lower_tier_weight=mlp_parameters.lower_tier_loss_weight,
            tier_loss_power=mlp_parameters.tier_loss_power,
            time_switch=mlp_parameters.time_switch,
            switch_time_constant=mlp_parameters.switch_time_constant,
            normalization_floor=mlp_parameters.normalization_floor,
        )
        model.set_correction_scale(scales.correction_scale)
    if verbose:
        print(f"Loaded MLP model: {model_path}")
    return model


def run_mlp_solver(
    model,
    t_eval,
    parameters=MLP,
    *,
    return_solution=False,
    verbose=False,
):
    """Evaluate at physical times; ``HEOMMLP`` normalizes them internally."""
    start = perf_counter()
    result = solve_mlp(
        model,
        t_eval,
        batch_size=parameters.inference_batch_size,
    )
    if verbose:
        print(f"MLP trajectory evaluation: {perf_counter() - start:.3f} s")
    expectation = np.real(result.expectation(sigmaz().full()))
    if return_solution:
        return expectation, result
    return expectation


def plot_trajectories(
    t_eval,
    lindbladian,
    sparse_heom,
    mlp,
    *,
    show,
    output,
    parameters=PSEUDOMODE,
    sigma_z_mae=None,
    full_state_mae=None,
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
    title = (
        rf"$g={parameters.g / parameters.w0:g}\,\omega_0$, "
        rf"$L={parameters.heom_depth}$"
    )
    errors = []
    if sigma_z_mae is not None:
        errors.append(rf"$E_z={sigma_z_mae:.1e}$")
    if full_state_mae is not None:
        errors.append(rf"$E_H={full_state_mae:.1e}$")
    if errors:
        title += ", " + ", ".join(errors)
    axis.set_title(title, fontsize=12)
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


def _plot_positive_tier_series(axis, tiers, values, *, label, **style):
    """Plot finite positive values, as required by a logarithmic axis."""
    values = np.asarray(values, dtype=np.float64)
    visible = np.isfinite(values) & (values > 0.0)
    if np.any(visible):
        axis.plot(tiers[visible], values[visible], label=label, **style)


def plot_tier_diagnostics(
    residual_statistics,
    ado_statistics,
    downward_forcing_statistics=None,
    *,
    show=True,
    output=None,
):
    """Plot the mean tier diagnostics in one compact two-panel figure.

    Point plots are used instead of grouped bars because several quantities
    share every tier and span many orders of magnitude.  Squared statistics
    are square-rooted so every displayed series is a magnitude.  Worst-time
    and peak-component statistics are intentionally not computed or plotted.
    """
    residual_by_tier = {
        item.tier: item for item in residual_statistics
    }
    ado_by_tier = {item.tier: item for item in ado_statistics}
    if not residual_by_tier or residual_by_tier.keys() != ado_by_tier.keys():
        raise ValueError(
            "residual and ADO statistics must contain the same nonempty tiers"
        )
    if any(
        item.mean_relative_rms_residual is None
        for item in residual_by_tier.values()
    ):
        raise ValueError(
            "residual statistics require an exact reference trajectory"
        )

    tiers = np.asarray(sorted(residual_by_tier), dtype=np.int64)
    residuals = [residual_by_tier[tier] for tier in tiers]
    ados = [ado_by_tier[tier] for tier in tiers]
    downward_forcing = (
        tuple(downward_forcing_statistics)
        if downward_forcing_statistics is not None
        else ()
    )
    forcing_tiers = np.asarray(
        [item.source_tier for item in downward_forcing],
        dtype=np.int64,
    )
    if downward_forcing and set(forcing_tiers) != set(tiers[tiers > 0]):
        raise ValueError(
            "downward forcing statistics must contain every source tier "
            "above tier zero"
        )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
        sharex=True,
        dpi=200,
    )
    relative_axis, mean_axis = axes

    relative_series = (
        (
            [item.mean_relative_rms_residual for item in residuals],
            r"Residual: mean per-ADO relative RMS",
            {"color": "#C1121F", "marker": "o", "linestyle": "-"},
        ),
        (
            [item.mean_relative_rms_error for item in ados],
            r"State error: mean per-ADO relative RMS",
            {"color": "#F77F00", "marker": "s", "linestyle": "--"},
        ),
    )
    for values, label, style in relative_series:
        _plot_positive_tier_series(
            relative_axis,
            tiers,
            values,
            label=label,
            linewidth=1.4,
            markersize=4.5,
            **style,
        )
    if downward_forcing:
        _plot_positive_tier_series(
            relative_axis,
            forcing_tiers,
            [item.mean_relative_rms_error for item in downward_forcing],
            label="Downward forcing error: mean per-target relative RMS",
            color="#7B2CBF",
            marker="^",
            linestyle=":",
            linewidth=1.4,
            markersize=4.5,
        )

    absolute_styles = (
        (
            "True ADO",
            "#003049",
            "o",
            "-",
        ),
        (
            "MLP ADO error",
            "#F77F00",
            "s",
            "--",
        ),
        (
            r"Exact $|d\chi/dt|$",
            "#2A9D8F",
            "^",
            "-",
        ),
        (
            "MLP residual",
            "#C1121F",
            "D",
            "--",
        ),
    )
    mean_values = (
        np.sqrt([item.mean_squared_activity for item in ados]),
        np.sqrt([item.mean_squared_error for item in ados]),
        np.sqrt(
            [item.mean_squared_dynamical_activity for item in residuals]
        ),
        np.sqrt([item.mean_squared_residual for item in residuals]),
    )
    for values, (label, color, marker, linestyle) in zip(
        mean_values,
        absolute_styles,
    ):
        _plot_positive_tier_series(
            mean_axis,
            tiers,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.4,
            markersize=4.5,
        )
    if downward_forcing:
        _plot_positive_tier_series(
            mean_axis,
            forcing_tiers,
            np.sqrt(
                [item.mean_squared_forcing for item in downward_forcing]
            ),
            label="Exact downward forcing",
            color="#6A4C93",
            marker="v",
            linestyle="-",
            linewidth=1.4,
            markersize=4.5,
        )
        _plot_positive_tier_series(
            mean_axis,
            forcing_tiers,
            np.sqrt([item.mean_squared_error for item in downward_forcing]),
            label="MLP downward-forcing error",
            color="#C77DFF",
            marker="P",
            linestyle="--",
            linewidth=1.4,
            markersize=4.5,
        )

    relative_axis.set_title("Mean relative accuracy")
    mean_axis.set_title("Time-mean RMS magnitude")
    for axis in axes:
        axis.set_yscale("log")
        axis.set_ylabel("Magnitude")
        axis.set_xlabel("HEOM tier")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_xticks(tiers)
    figure.suptitle("Tier-resolved MLP diagnostics")
    figure.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=200)
    if show:
        plt.show()
    return axes


def _default_tier_output(trajectory_output):
    if trajectory_output is None:
        return None
    trajectory_output = Path(trajectory_output)
    suffix = trajectory_output.suffix or ".png"
    return trajectory_output.with_name(
        f"{trajectory_output.stem}_tier_diagnostics{suffix}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        help="optional extra plot copy; trajectory.png is always saved beside the checkpoint",
    )
    parser.add_argument(
        "--tier-output",
        type=Path,
        help=(
            "tier-diagnostic figure path (default: derived from --output)"
        ),
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print timing, horizon-specific errors, and tier statistics",
    )
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
        help=(
            "saved MLP state dictionary to evaluate (default: the named "
            "sequence folder, selected folder, or "
            "saved_models/mlp/mlp_state_dict.pt)"
        ),
    )
    folder_dialog_group = parser.add_mutually_exclusive_group()
    folder_dialog_group.add_argument(
        "--folder-dialog",
        dest="folder_dialog",
        action="store_true",
        default=True,
        help=(
            "select the model folder with a dialog when neither --sequence "
            "nor --model-path is supplied (default)"
        ),
    )
    folder_dialog_group.add_argument(
        "--no-folder-dialog",
        dest="folder_dialog",
        action="store_false",
        help=(
            "do not open the folder dialog; use "
            "saved_models/mlp/mlp_state_dict.pt by default"
        ),
    )
    parser.add_argument(
        "--tier-diagnostics",
        action="store_true",
        help=(
            "compute exact-dynamics-relative tier residuals, true ADO "
            "activity, direct MLP ADO errors, and downward forcing"
        ),
    )
    args = parser.parse_args(argv)
    if args.no_show:
        plt.switch_backend("Agg")

    try:
        if args.sequence is not None:
            sequence = load_training_sequence(args.sequence)
            model_path = resolve_model_path(sequence, args.model_path)
        else:
            model_path = (
                args.model_path
                if args.model_path is not None
                else resolve_default_model_path(
                    folder_dialog=args.folder_dialog,
                )
            )
            if training_metadata_path(model_path).is_file():
                sequence = load_training_metadata(model_path)
            else:
                sequence = default_training_sequence()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"saved MLP checkpoint not found: {model_path}"
            )
        require_complete_training_checkpoint(model_path)
        t_eval, reference_t_eval, reference_offset = build_benchmark_time_grids(
            sequence.pseudomode,
            t_start=args.t_start,
            t_stop=args.t_stop,
            n_times=args.n_times,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    pseudomode = sequence.pseudomode
    mlp_parameters = sequence.base_mlp
    if args.verbose:
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

    lindbladian = run_lindbladian(
        reference_t_eval,
        pseudomode,
        verbose=args.verbose,
    )[
        reference_offset:
    ]
    hierarchy, initial_heom_state, liouvillian = (
        build_normalized_hard_heom(pseudomode, verbose=args.verbose)
    )
    sparse_output = run_sparse_numerics(
        hierarchy,
        initial_heom_state,
        liouvillian,
        reference_t_eval,
        pseudomode,
        return_solution=True,
        verbose=args.verbose,
    )
    sparse_heom, sparse_solution = sparse_output
    reference_state = sparse_solution.y[:, reference_offset:]
    sparse_heom = sparse_heom[reference_offset:]
    model = load_mlp(
        hierarchy,
        initial_heom_state,
        liouvillian=liouvillian,
        mlp_parameters=mlp_parameters,
        pseudomode_parameters=pseudomode,
        model_path=model_path,
        verbose=args.verbose,
    )
    mlp_output = run_mlp_solver(
        model,
        t_eval,
        mlp_parameters,
        return_solution=True,
        verbose=args.verbose,
    )
    mlp, mlp_solution = mlp_output
    if args.tier_diagnostics:
        tier_residuals = compute_tier_residual_statistics(
            model,
            hierarchy,
            liouvillian,
            t_eval,
            reference_state=reference_state,
            batch_size=mlp_parameters.inference_batch_size,
            verbose=args.verbose,
        )
        tier_ado_statistics = compute_tier_ado_statistics(
            hierarchy,
            reference_state,
            mlp_solution.y,
        )
        tier_downward_forcing = compute_tier_downward_forcing_statistics(
            hierarchy,
            liouvillian,
            reference_state,
            mlp_solution.y,
        )
    else:
        tier_residuals = None
        tier_ado_statistics = None
        tier_downward_forcing = None
    sparse_lindbladian_sigma_z_mae = mean_absolute_error(
        lindbladian,
        sparse_heom,
    )
    mlp_sparse_sigma_z_mae = mean_absolute_error(sparse_heom, mlp)
    mlp_sparse_full_state_mae = mean_absolute_error(
        reference_state,
        mlp_solution.y,
    )
    print(
        "Mean absolute <sigma_z> error "
        "(sparse HEOM vs Lindbladian): "
        f"{sparse_lindbladian_sigma_z_mae:.3e}"
    )
    print(
        "Mean absolute <sigma_z> error (MLP vs sparse HEOM): "
        f"{mlp_sparse_sigma_z_mae:.3e}"
    )
    print(
        "Mean absolute full normalized-HEOM state-component error "
        "(MLP vs sparse HEOM): "
        f"{mlp_sparse_full_state_mae:.3e}"
    )
    if args.verbose:

        trained_mask = t_eval <= pseudomode.t_stop
        extrapolation_mask = t_eval > pseudomode.t_stop
        if np.any(trained_mask) and np.any(extrapolation_mask):
            trained_sigma_z_mae = mean_absolute_error(
                sparse_heom[trained_mask],
                mlp[trained_mask],
            )
            print(
                "Mean absolute <sigma_z> error (MLP vs sparse HEOM) "
                "within training horizon: "
                f"{trained_sigma_z_mae:.3e}"
            )
        if np.any(extrapolation_mask):
            extrapolated_sigma_z_mae = mean_absolute_error(
                sparse_heom[extrapolation_mask],
                mlp[extrapolation_mask],
            )
            print(
                "Mean absolute <sigma_z> error (MLP vs sparse HEOM) "
                "beyond training horizon: "
                f"{extrapolated_sigma_z_mae:.3e}"
            )
        if tier_residuals is not None:
            print("Tier-resolved MLP dynamical residuals:")
            for statistics in tier_residuals:
                print(
                    f"  tier {statistics.tier} (N={statistics.n_ados}): "
                    "mean_per_ADO_rel_R_rms="
                    f"{statistics.mean_relative_rms_residual:.6e}  "
                    f"E_res_mean={statistics.mean_squared_residual:.6e}  "
                    "E_dyn_mean="
                    f"{statistics.mean_squared_dynamical_activity:.6e}"
                )

            print("Tier-resolved true ADO activity and direct MLP ADO errors:")
            for statistics in tier_ado_statistics:
                print(
                    f"  tier {statistics.tier} (N={statistics.n_ados}): "
                    f"A_mean={statistics.mean_squared_activity:.6e}  "
                    "mean_per_ADO_rel_state_rms="
                    f"{statistics.mean_relative_rms_error:.6e}  "
                    f"E_state_mean={statistics.mean_squared_error:.6e}  "
                )
            print("Tier-resolved downward forcing into lower tiers:")
            for statistics in tier_downward_forcing:
                print(
                    f"  tier {statistics.source_tier} -> "
                    f"{statistics.target_tier} "
                    f"(N_source={statistics.n_source_ados}, "
                    f"N_target={statistics.n_target_ados}): "
                    "mean_per_target_rel_force_rms="
                    f"{statistics.mean_relative_rms_error:.6e}  "
                    "E_force_mean="
                    f"{statistics.mean_squared_forcing:.6e}  "
                    "E_force_error_mean="
                    f"{statistics.mean_squared_error:.6e}"
                )
    trajectory_output = model_path.parent / "trajectory.png"
    trajectory_axis = plot_trajectories(
        t_eval,
        lindbladian,
        sparse_heom,
        mlp,
        show=False,
        output=trajectory_output,
        parameters=pseudomode,
        sigma_z_mae=mlp_sparse_sigma_z_mae,
        full_state_mae=mlp_sparse_full_state_mae,
    )
    print(f"Saved trajectory: {trajectory_output}")
    if args.output is not None and args.output.resolve() != trajectory_output.resolve():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        trajectory_axis.figure.savefig(args.output, dpi=200)
    if tier_residuals is not None:
        tier_output = (
            args.tier_output
            if args.tier_output is not None
            else _default_tier_output(args.output)
        )
        plot_tier_diagnostics(
            tier_residuals,
            tier_ado_statistics,
            tier_downward_forcing,
            show=False,
            output=tier_output,
        )
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
