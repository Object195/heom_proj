"""Compare a damped pseudomode with QuTiP and the local HEOM builder.

The normalized sparse run also plots time-resolved ADO activity and RMS time
derivatives for every tier. Heatmaps retain all tiers; line plots show at most
seven representative tiers by default. For example::

    python -m benchmark.benchmark_lindbladian --depths 5 10 20 \
        --tier-lines 0 1 2 5 10 20 --skip-spectrum

Depth-dependent observable differences complement the activity plots: ADO
amplitudes and derivatives depend on the representation and are not error bounds.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

# Allow both ``python -m benchmark.benchmark_lindbladian`` and direct file
# execution from an IDE.  Direct execution otherwise places only the
# ``benchmark`` directory on Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from qutip import (
    basis,
    destroy,
    mesolve,
    qeye,
    sigmax,
    sigmaz,
    tensor,
)
from qutip.solver.heom import BosonicBath, HEOMSolver

from experiment_parameters import PSEUDOMODE
from heom.heom_rep import heom_state
from heom.heom_solver import diagnose_heom_spectrum, solve_heom


w0 = PSEUDOMODE.w0
Delta = PSEUDOMODE.delta
V = PSEUDOMODE.v
g = PSEUDOMODE.g
gamma = PSEUDOMODE.gamma
cavity_dimension = PSEUDOMODE.cavity_dimension
depth_list = list(PSEUDOMODE.qutip_depths)
tlist = np.linspace(
    PSEUDOMODE.t_start,
    PSEUDOMODE.t_stop,
    PSEUDOMODE.n_times,
)


@dataclass(frozen=True)
class TierTimeDiagnostics:
    """Per-time RMS over matrix elements and ADOs, without a time average.

    ``ado_rms`` and ``derivative_rms`` both have shape ``(n_tiers, n_times)``,
    including tier zero. Derivatives include the full HEOM right-hand side.
    Both quantities use the supplied hierarchy's normalization convention.
    """

    tiers: np.ndarray
    n_ados: np.ndarray
    ado_rms: np.ndarray
    derivative_rms: np.ndarray


def compute_tier_time_diagnostics(hierarchy, liouvillian, state):
    r"""Compute ``sqrt(mean(|chi_l(t)|^2))`` and RMS time derivatives.

    ``state`` uses the solver layout ``(nADO * system_size, n_times)``.
    The mean includes every matrix element and ADO within a tier, matching
    the component RMS convention in ``benchmark_mlp``. The exact derivative
    is ``liouvillian @ state``, including same-tier dynamics and couplings
    from other tiers, summed before taking complex magnitudes. The time mean
    of ``derivative_rms**2`` matches that benchmark's
    ``mean_squared_dynamical_activity`` for the same state and Liouvillian.
    """
    state = np.asarray(state)
    state_size = hierarchy.nADO * hierarchy.system_size
    if state.ndim != 2 or state.shape[0] != state_size or not state.shape[1]:
        raise ValueError("state must have shape (nADO * system_size, n_times)")
    if not np.isfinite(state).all():
        raise ValueError("state must contain only finite values")
    if liouvillian.shape != (state_size, state_size):
        raise ValueError("liouvillian shape must match the full HEOM state")

    ado_tiers = np.array([
        hierarchy._tier(node) for node in hierarchy.idx_to_node
    ])
    tiers, counts = np.unique(ado_tiers, return_counts=True)
    component_tiers = np.repeat(ado_tiers, hierarchy.system_size)
    components = {
        tier: np.flatnonzero(component_tiers == tier) for tier in tiers
    }
    ado_rms = np.stack([
        np.sqrt(np.mean(np.abs(state[components[tier]]) ** 2, axis=0))
        for tier in tiers
    ])
    derivative = np.asarray(liouvillian @ state)
    derivative_rms = np.stack([
        np.sqrt(np.mean(np.abs(derivative[components[tier]]) ** 2, axis=0))
        for tier in tiers
    ])
    return TierTimeDiagnostics(tiers, counts, ado_rms, derivative_rms)


def select_plot_tiers(tiers, selected_tiers=None):
    """Keep low tiers and the boundary, sampling the interior if necessary."""
    tiers = np.asarray(tiers, dtype=int)
    if selected_tiers is not None:
        selected = np.unique(selected_tiers)
        if not selected.size or not np.isin(selected, tiers).all():
            raise ValueError(f"selected tiers must be drawn from {tiers.tolist()}")
        return selected
    if len(tiers) <= 7:
        return tiers
    interior = np.linspace(2, len(tiers) - 1, 5).round().astype(int)
    indices = np.unique(np.r_[0, 1, interior])
    return tiers[indices]


def plot_tier_time_diagnostics(
    times,
    diagnostics,
    *,
    calculation_name,
    selected_tiers=None,
    dynamic_range_decades=10,
    output=None,
):
    """Plot every tier on log-color heatmaps, plus selected semilog traces.

    Each heatmap has one color scale shared across all times and tiers;
    individual tiers are never rescaled. The display spans at most
    ``dynamic_range_decades`` below each diagnostic's global peak. Smaller
    positive values use the heatmap's under-range color and fall below the
    line axes; the computed values are unchanged. Exact zeros are gray in
    heatmaps and gaps in log traces. Tier selection affects only line panels.
    The returned axes support zooming into any time interval without reruns.
    """
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or times.size < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("times must contain at least two increasing finite times")
    if not np.isfinite(dynamic_range_decades) or dynamic_range_decades <= 0:
        raise ValueError("dynamic_range_decades must be finite and positive")
    selected = select_plot_tiers(diagnostics.tiers, selected_tiers)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150, sharex=True,
                             layout="constrained")
    colors = dict(zip(selected, plt.get_cmap("viridis")(
        np.linspace(0.05, 0.9, len(selected))
    )))
    rows = (
        (diagnostics.tiers, diagnostics.ado_rms, "ADO component RMS", "HEOM tier"),
        (diagnostics.tiers, diagnostics.derivative_rms,
         "ADO time-derivative component RMS", "HEOM tier"),
    )
    for row, (tiers, values, title, tier_label) in enumerate(rows):
        heat_axis, line_axis = axes[row]
        if values.shape != (len(tiers), len(times)):
            raise ValueError("diagnostic arrays must match tiers and times")
        positive = values[values > 0]
        vmin, vmax = (
            (positive.min(), positive.max()) if positive.size else (1e-10, 1.0)
        )
        vmin = max(vmin, vmax * 10.0 ** (-dynamic_range_decades))
        if vmin == vmax:
            vmin = vmax / 10
        cmap = plt.get_cmap("viridis").with_extremes(bad="0.85", under="#171126")
        mesh = heat_axis.pcolormesh(
            times, tiers, np.ma.masked_less_equal(values, 0),
            shading="nearest", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
            rasterized=True,
        )
        heat_axis.set_ylim(tiers[0] - 0.5, tiers[-1] + 0.5)
        heat_axis.set_yticks(select_plot_tiers(tiers))
        heat_axis.set_ylabel(tier_label)
        heat_axis.set_title(f"{title}: all tiers (gray = zero)")
        fig.colorbar(mesh, ax=heat_axis,
                     extend="min" if np.any(positive < vmin) else "neither",
                     label="RMS" if row == 0 else "RMS / time")
        for tier, series in zip(tiers, values):
            if tier not in selected:
                continue
            count = diagnostics.n_ados[np.flatnonzero(diagnostics.tiers == tier)[0]]
            label = f"Tier {tier} (N={count})"
            line_axis.plot(times, np.where(series > 0, series, np.nan),
                           color=colors[tier], label=label)
        line_axis.set_yscale("log")
        line_axis.set_ylim(vmin / 2, vmax * 2)
        line_axis.set_title(f"{title}: selected tiers")
        line_axis.set_ylabel("RMS" if row == 0 else "RMS / time")
        line_axis.grid(True, alpha=0.3)
        if line_axis.lines:
            line_axis.legend(fontsize=8, ncol=2)
    for axis in axes.flat:
        axis.set_xlabel("Time")
        axis.set_xlim(times[0], times[-1])
    fig.suptitle(
        f"Sparse HEOM tier dynamics ({calculation_name}, L={diagnostics.tiers[-1]})\n"
        f"Display: up to {dynamic_range_decades:g} decades below each diagnostic's peak; "
        "magnitudes depend on ADO scaling."
    )
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    return axes


def run_pseudomode_model(t_eval=None):
    """Simulate the explicitly damped cavity pseudomode."""
    if t_eval is None:
        t_eval = tlist

    a = tensor(qeye(2), destroy(cavity_dimension))
    sz_full = tensor(sigmaz(), qeye(cavity_dimension))
    sx_full = tensor(sigmax(), qeye(cavity_dimension))

    h_system = 0.5 * Delta * sz_full + 0.5 * V * sx_full
    h_cavity = w0 * (a.dag() @ a)
    h_interaction = g * (sz_full @ (a + a.dag()))
    h_total = h_system + h_cavity + h_interaction

    collapse_operators = [np.sqrt(gamma) * a]
    psi0 = tensor(basis(2, 0), basis(cavity_dimension, 0))

    start = perf_counter()
    result = mesolve(
        h_total,
        psi0,
        t_eval,
        c_ops=collapse_operators,
        e_ops={"sz": sz_full},
    )
    elapsed = perf_counter() - start
    print(f"Pseudomode Lindblad propagation: {elapsed:.3f} s")
    return np.asarray(result.e_data["sz"]).real


def pseudomode_bath_expansion():
    """Return the real/imaginary exponential expansion used by QuTiP."""
    nu_plus = 0.5 * gamma + 1j * w0
    nu_minus = 0.5 * gamma - 1j * w0

    frequencies = np.array([nu_plus, nu_minus], dtype=np.complex128)
    coefficients_real = np.array(
        [0.5 * g**2, 0.5 * g**2], dtype=np.complex128
    )
    coefficients_imag = np.array(
        [-0.5j * g**2, 0.5j * g**2], dtype=np.complex128
    )
    return frequencies, coefficients_real, coefficients_imag


def free_pole_bath_expansion():
    """Return the single physical pole used by the free-pole HEOM."""
    frequencies = np.array([0.5 * gamma + 1j * w0], dtype=np.complex128)
    coefficients = np.array([g**2], dtype=np.complex128)
    return frequencies, coefficients


def run_qutip_heom(t_eval=None, depths=None):
    """Simulate the reduced spin with QuTiP's HEOM implementation."""
    if t_eval is None:
        t_eval = tlist
    if depths is None:
        depths = depth_list

    h_system = 0.5 * Delta * sigmaz() + 0.5 * V * sigmax()
    rho0 = basis(2, 0).proj()
    frequencies, coefficients_real, coefficients_imag = (
        pseudomode_bath_expansion()
    )

    bath = BosonicBath(
        sigmaz(),
        coefficients_real,
        frequencies,
        coefficients_imag,
        frequencies,
    )
    options = {
        "method": "bdf",
        "nsteps": 1_000_000,
        "rtol": PSEUDOMODE.rtol,
        "atol": PSEUDOMODE.atol,
        "progress_bar": "",
    }

    trajectories = {}
    for max_depth in depths:
        solver = HEOMSolver(
            h_system,
            bath,
            max_depth=max_depth,
            options=options,
        )
        start = perf_counter()
        result = solver.run(rho0, t_eval, e_ops={"sz": sigmaz()})
        elapsed = perf_counter() - start
        print(f"QuTiP HEOM propagation (L={max_depth}): {elapsed:.3f} s")
        trajectories[max_depth] = np.asarray(result.e_data["sz"]).real
    return trajectories


def diagnose_sparse_liouvillian(
    model,
    rho0,
    liouvillian,
    truncation_name,
    max_depth,
    *,
    relevant_only=True,
    stability_tol=1e-10,
):
    """Plot the spectrum and warn about exponentially growing HEOM modes."""
    _, ax = plt.subplots(dpi=200)
    start = perf_counter()
    spectrum = diagnose_heom_spectrum(
        model,
        rho0,
        liouvillian=liouvillian,
        relevant_only=relevant_only,
        ax=ax,
        show=False,
    )
    elapsed = perf_counter() - start

    mode_description = "initial-state-relevant" if relevant_only else "all"
    ax.set_title(
        f"Sparse HEOM spectrum ({truncation_name}, "
        rf"$L={max_depth}$; {mode_description} modes)"
    )
    ax.figure.tight_layout()

    largest_selected = spectrum.largest_real_part
    largest_overall = float(np.max(spectrum.eigenvalues.real))
    print(
        f"Sparse HEOM ({truncation_name}) spectral diagnostic "
        f"(L={max_depth}): {elapsed:.3f} s"
    )
    if largest_selected > stability_tol:
        print(
            "DIVERGENCE WARNING: "
            f"largest {mode_description} Re(lambda)={largest_selected:.6e} "
            f"> tolerance {stability_tol:.1e}."
        )
    elif relevant_only and largest_overall > stability_tol:
        print(
            "SPECTRAL WARNING: the Liouvillian has a growing mode "
            f"(largest overall Re(lambda)={largest_overall:.6e}), but it does "
            "not pass the initial-state relevance threshold."
        )
    else:
        print(
            f"No growing {mode_description} modes detected above "
            f"{stability_tol:.1e}."
        )
    return spectrum


def run_sparse_heom(
    max_depth,
    *,
    markovian_terminator=False,
    normalized=False,
    diagnose_spectrum=True,
    relevant_spectrum_only=True,
    spectral_stability_tol=1e-10,
    tier_diagnostics=False,
    tier_lines=None,
    tier_decades=10,
    tier_output=None,
):
    """Build, diagnose, and propagate a sparse HEOM.

    The spectral diagnostic is enabled by default.  Set
    ``relevant_spectrum_only=False`` to plot every eigenvalue rather than only
    modes excited by ``rho0``, or ``diagnose_spectrum=False`` to skip the dense
    eigendecomposition for large hierarchies.  Set ``normalized=True`` to use
    square-root-scaled ADO raising and lowering blocks.
    ``tier_diagnostics=True`` plots all tiers using the same propagated state;
    ``tier_lines`` optionally selects which tiers also receive line traces.
    """
    h_system = 0.5 * Delta * sigmaz().full() + 0.5 * V * sigmax().full()
    coupling_operator = sigmaz().full()
    rho0 = basis(2, 0).proj().full()
    frequencies, coefficients = free_pole_bath_expansion()

    model = heom_state(
        K=len(frequencies) - 1,
        L=max_depth,
        H_s=h_system,
        H_c=coupling_operator,
        C_list=coefficients,
        gamma_list=frequencies,
    )

    truncation_name = (
        "Markovian terminator" if markovian_terminator else "hard cutoff"
    )
    representation_name = "normalized" if normalized else "unnormalized"
    calculation_name = f"{representation_name}, {truncation_name}"
    start = perf_counter()
    liouvillian = model.build_Liouvillian(
        markovian_terminator=markovian_terminator,
        normalized=normalized,
    )
    build_elapsed = perf_counter() - start
    print(
        f"Sparse HEOM ({calculation_name}) construction "
        f"(L={max_depth}, ADOs={model.nADO}, shape={liouvillian.shape}): "
        f"{build_elapsed:.3f} s"
    )

    if diagnose_spectrum:
        diagnose_sparse_liouvillian(
            model,
            rho0,
            liouvillian,
            calculation_name,
            max_depth,
            relevant_only=relevant_spectrum_only,
            stability_tol=spectral_stability_tol,
        )

    start = perf_counter()
    result = solve_heom(
        model,
        rho0,
        tlist,
        liouvillian=liouvillian,
        method="BDF",
        rtol=PSEUDOMODE.rtol,
        atol=PSEUDOMODE.atol,
    )
    solve_elapsed = perf_counter() - start
    print(
        f"Sparse HEOM ({calculation_name}) BDF propagation "
        f"(L={max_depth}): {solve_elapsed:.3f} s "
        f"[nfev={result.nfev}, njev={result.njev}, nlu={result.nlu}]"
    )
    if tier_diagnostics:
        diagnostics = compute_tier_time_diagnostics(model, liouvillian, result.y)
        plot_tier_time_diagnostics(
            result.t,
            diagnostics,
            calculation_name=calculation_name,
            selected_tiers=tier_lines,
            dynamic_range_decades=tier_decades,
            output=tier_output,
        )
    return np.real(result.expectation(sigmaz().full()))


def plot_trajectories(
    sz_pseudomode,
    qutip_heom,
    sz_sparse_hard,
    sz_sparse_normalized,
    sz_sparse_markovian,
    sparse_depth,
):
    """Plot the reference, normalized, and truncated HEOM calculations."""
    colors = ["#F08080", "#CD5C5C", "#B22222"]
    plt.figure(dpi=200)
    plt.plot(tlist, sz_pseudomode, "b-", label="Pseudomode Lindblad")

    for i, max_depth in enumerate(sorted(qutip_heom)):
        plt.plot(
            tlist,
            qutip_heom[max_depth],
            "--",
            label=rf"QuTiP HEOM, $L={max_depth}$",
            color=colors[i % len(colors)],
        )

    plt.plot(
        tlist,
        sz_sparse_hard,
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=rf"Sparse HEOM, hard cutoff, $L={sparse_depth}$",
    )
    plt.plot(
        tlist,
        sz_sparse_markovian,
        color="#008B8B",
        linestyle="-.",
        linewidth=1.8,
        label=rf"Sparse HEOM, Markovian terminator, $L={sparse_depth}$",
    )
    plt.plot(
        tlist,
        sz_sparse_normalized,
        color="#7B2CBF",
        linestyle="--",
        linewidth=1.4,
        label=rf"Sparse HEOM, normalized hard cutoff, $L={sparse_depth}$",
    )
    plt.title(rf"$g={g / w0:g}\,\omega_0$", fontsize=14)
    plt.xlabel(r"$t$", fontsize=14)
    plt.ylabel(r"$S_z$", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.tight_layout()


def plot_depth_errors(sz_pseudomode, qutip_heom):
    """Show time-local reference error and changes on raising the cutoff.

    Increasing the cutoff changes all omitted tiers together, so adjacent
    differences do not isolate a single tier's causal effect. The finite
    cavity pseudomode reference also needs its own dimension convergence.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150,
                             sharex=True, layout="constrained")
    depths = sorted(qutip_heom)
    for depth in depths:
        error = np.abs(qutip_heom[depth] - sz_pseudomode)
        axes[0].plot(tlist, np.where(error > 0, error, np.nan), label=f"L={depth}")
    for lower, upper in zip(depths[:-1], depths[1:]):
        difference = np.abs(qutip_heom[upper] - qutip_heom[lower])
        axes[1].plot(tlist, np.where(difference > 0, difference, np.nan),
                     label=f"L={upper} vs {lower}")
    axes[0].set_title("QuTiP HEOM vs finite-cavity pseudomode")
    axes[1].set_title("Change on increasing hierarchy depth")
    if len(depths) < 2:
        axes[1].text(0.5, 0.5, "Use --depths 5 10 20 to compare cutoffs",
                     ha="center", transform=axes[1].transAxes)
    for axis in axes:
        if not any(np.isfinite(line.get_ydata()).any() for line in axis.lines):
            axis.set_ylim(1e-16, 1)
        axis.set_yscale("log")
        axis.set_xlabel("Time")
        axis.set_ylabel(r"Absolute difference in $S_z$")
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend(fontsize=9)
    return axes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", nargs="+", type=int, default=depth_list,
                        help="QuTiP cutoffs; sparse runs use the largest")
    parser.add_argument("--tier-lines", nargs="+", type=int,
                        help="tiers for line traces; heatmaps always show every tier")
    parser.add_argument("--tier-output", type=Path,
                        help="save the tier figure to this path")
    parser.add_argument("--tier-decades", type=float, default=10,
                        help="display range below each diagnostic's peak (default: 10 decades)")
    parser.add_argument("--skip-spectrum", action="store_true",
                        help="skip dense eigendecompositions for large hierarchies")
    parser.add_argument("--no-show", action="store_true",
                        help="compute plots without opening plot windows")
    args = parser.parse_args(argv)
    depths = sorted(set(args.depths))
    if depths[0] < 0:
        parser.error("depths must be nonnegative")
    if not np.isfinite(args.tier_decades) or args.tier_decades <= 0:
        parser.error("tier-decades must be finite and positive")
    sparse_depth = depths[-1]
    try:
        select_plot_tiers(np.arange(sparse_depth + 1), args.tier_lines)
    except ValueError as error:
        parser.error(str(error))
    sz_pseudomode = run_pseudomode_model()
    qutip_heom = run_qutip_heom(depths=depths)
    sz_sparse_hard = run_sparse_heom(
        sparse_depth, diagnose_spectrum=not args.skip_spectrum,
    )
    sz_sparse_normalized = run_sparse_heom(
        sparse_depth,
        normalized=True,
        diagnose_spectrum=not args.skip_spectrum,
        tier_diagnostics=True,
        tier_lines=args.tier_lines,
        tier_decades=args.tier_decades,
        tier_output=args.tier_output,
    )
    sz_sparse_markovian = run_sparse_heom(
        sparse_depth,
        markovian_terminator=True,
        diagnose_spectrum=not args.skip_spectrum,
    )

    hard_qutip_difference = np.max(
        np.abs(sz_sparse_hard - qutip_heom[sparse_depth])
    )
    markovian_qutip_difference = np.max(
        np.abs(sz_sparse_markovian - qutip_heom[sparse_depth])
    )
    hard_pseudomode_difference = np.max(
        np.abs(sz_sparse_hard - sz_pseudomode)
    )
    markovian_pseudomode_difference = np.max(
        np.abs(sz_sparse_markovian - sz_pseudomode)
    )
    hard_markovian_difference = np.max(
        np.abs(sz_sparse_hard - sz_sparse_markovian)
    )
    normalized_hard_difference = np.max(
        np.abs(sz_sparse_normalized - sz_sparse_hard)
    )
    print(
        "Max |sparse HEOM (hard cutoff) - QuTiP HEOM|: "
        f"{hard_qutip_difference:.3e}"
    )
    print(
        "Max |sparse HEOM (Markovian terminator) - QuTiP HEOM|: "
        f"{markovian_qutip_difference:.3e}"
    )
    print(
        "Max |sparse HEOM (hard cutoff) - pseudomode|: "
        f"{hard_pseudomode_difference:.3e}"
    )
    print(
        "Max |sparse HEOM (Markovian terminator) - pseudomode|: "
        f"{markovian_pseudomode_difference:.3e}"
    )
    print(
        "Max |sparse HEOM (hard cutoff) - sparse HEOM "
        f"(Markovian terminator)|: {hard_markovian_difference:.3e}"
    )
    print(
        "Max |sparse HEOM (normalized) - sparse HEOM (unnormalized)|: "
        f"{normalized_hard_difference:.3e}"
    )

    plot_trajectories(
        sz_pseudomode,
        qutip_heom,
        sz_sparse_hard,
        sz_sparse_normalized,
        sz_sparse_markovian,
        sparse_depth,
    )
    plot_depth_errors(sz_pseudomode, qutip_heom)
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
