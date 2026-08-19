"""Compare a damped pseudomode with QuTiP and the local HEOM builder."""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from qutip import (
    Options,
    basis,
    destroy,
    mesolve,
    qeye,
    sigmax,
    sigmaz,
    tensor,
)
from qutip.nonmarkov.heom import BosonicBath, HEOMSolver

from heom_rep import heom_state
from heom_solver import diagnose_heom_spectrum, solve_heom


# Model parameters
w0 = 1.0
Delta = 1
V = 0.1
# This value is strong enough to show non-Markovian dynamics while remaining
# converged for the hierarchy depths used below. At g=2, substantially more
# careful hierarchy termination is needed and a shallow HEOM is unstable.
g = 1
gamma = 0.1

cavity_dimension = 20
depth_list = [20]
tlist = np.linspace(0.0, 100.0, 1000)


def run_pseudomode_model():
    """Simulate the explicitly damped cavity pseudomode."""
    a = tensor(qeye(2), destroy(cavity_dimension))
    sz_full = tensor(sigmaz(), qeye(cavity_dimension))
    sx_full = tensor(sigmax(), qeye(cavity_dimension))

    h_system = 0.5 * Delta * sz_full + 0.5 * V * sx_full
    h_cavity = w0 * a.dag() * a
    h_interaction = g * sz_full * (a + a.dag())
    h_total = h_system + h_cavity + h_interaction

    collapse_operators = [np.sqrt(gamma) * a]
    psi0 = tensor(basis(2, 0), basis(cavity_dimension, 0))

    start = perf_counter()
    result = mesolve(
        h_total,
        psi0,
        tlist,
        collapse_operators,
        [sz_full],
    )
    elapsed = perf_counter() - start
    print(f"Pseudomode Lindblad propagation: {elapsed:.3f} s")
    return np.real(result.expect[0])


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


def run_qutip_heom():
    """Simulate the reduced spin with QuTiP's HEOM implementation."""
    h_system = 0.5 * Delta * sigmaz() + 0.5 * V * sigmax()
    rho0 = basis(2, 0) * basis(2, 0).dag()
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
    options = Options(
        method="bdf",
        nsteps=1_000_000,
        rtol=1e-8,
        atol=1e-10,
    )

    trajectories = {}
    for max_depth in depth_list:
        solver = HEOMSolver(
            h_system,
            bath,
            max_depth=max_depth,
            options=options,
        )
        start = perf_counter()
        result = solver.run(rho0, tlist, e_ops=[sigmaz()])
        elapsed = perf_counter() - start
        print(f"QuTiP HEOM propagation (L={max_depth}): {elapsed:.3f} s")
        trajectories[max_depth] = np.real(result.expect[0])
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
):
    """Build, diagnose, and propagate a sparse HEOM.

    The spectral diagnostic is enabled by default.  Set
    ``relevant_spectrum_only=False`` to plot every eigenvalue rather than only
    modes excited by ``rho0``, or ``diagnose_spectrum=False`` to skip the dense
    eigendecomposition for large hierarchies.  Set ``normalized=True`` to use
    square-root-scaled ADO raising and lowering blocks.
    """
    h_system = 0.5 * Delta * sigmaz().full() + 0.5 * V * sigmax().full()
    coupling_operator = sigmaz().full()
    rho0 = (basis(2, 0) * basis(2, 0).dag()).full()
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
        rtol=1e-8,
        atol=1e-10,
    )
    solve_elapsed = perf_counter() - start
    print(
        f"Sparse HEOM ({calculation_name}) BDF propagation "
        f"(L={max_depth}): {solve_elapsed:.3f} s "
        f"[nfev={result.nfev}, njev={result.njev}, nlu={result.nlu}]"
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

    for i, max_depth in enumerate(depth_list):
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
    plt.show()


def main():
    sz_pseudomode = run_pseudomode_model()
    qutip_heom = run_qutip_heom()
    sparse_depth = depth_list[-1]
    sz_sparse_hard = run_sparse_heom(sparse_depth)
    sz_sparse_normalized = run_sparse_heom(
        sparse_depth,
        normalized=True,
    )
    sz_sparse_markovian = run_sparse_heom(
        sparse_depth,
        markovian_terminator=True,
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
    np.testing.assert_allclose(
        sz_sparse_normalized,
        sz_sparse_hard,
        rtol=2e-7,
        atol=2e-9,
        err_msg=(
            "Normalized and unnormalized HEOM reduced observables disagree"
        ),
    )

    plot_trajectories(
        sz_pseudomode,
        qutip_heom,
        sz_sparse_hard,
        sz_sparse_normalized,
        sz_sparse_markovian,
        sparse_depth,
    )


if __name__ == "__main__":
    main()
