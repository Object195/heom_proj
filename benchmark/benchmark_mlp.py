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


def run_lindbladian(t_eval):
    """Propagate the explicit damped-cavity Lindblad reference."""
    annihilation = tensor(qeye(2), destroy(PSEUDOMODE.cavity_dimension))
    sz_full = tensor(sigmaz(), qeye(PSEUDOMODE.cavity_dimension))
    sx_full = tensor(sigmax(), qeye(PSEUDOMODE.cavity_dimension))
    h_system = 0.5 * PSEUDOMODE.delta * sz_full
    h_system += 0.5 * PSEUDOMODE.v * sx_full
    h_cavity = PSEUDOMODE.w0 * (annihilation.dag() @ annihilation)
    h_interaction = (
        PSEUDOMODE.g * (sz_full @ (annihilation + annihilation.dag()))
    )
    h_total = h_system + h_cavity + h_interaction
    collapse_operators = [np.sqrt(PSEUDOMODE.gamma) * annihilation]
    psi0 = tensor(
        basis(2, 0),
        basis(PSEUDOMODE.cavity_dimension, 0),
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


def build_normalized_hard_heom():
    """Build the normalized, hard-truncated free-pole HEOM."""
    h_system = 0.5 * PSEUDOMODE.delta * sigmaz().full()
    h_system += 0.5 * PSEUDOMODE.v * sigmax().full()
    rho0 = basis(2, 0).proj().full()
    frequencies = np.array(
        [0.5 * PSEUDOMODE.gamma + 1j * PSEUDOMODE.w0],
        dtype=np.complex128,
    )
    coefficients = np.array([PSEUDOMODE.g**2], dtype=np.complex128)
    hierarchy = heom_state(
        K=0,
        L=PSEUDOMODE.heom_depth,
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
        f"(L={PSEUDOMODE.heom_depth}, ADOs={hierarchy.nADO}, "
        f"shape={liouvillian.shape}): {perf_counter() - start:.3f} s"
    )
    return hierarchy, rho0, liouvillian


def run_sparse_numerics(hierarchy, rho0, liouvillian, t_eval):
    start = perf_counter()
    result = solve_heom(
        hierarchy,
        rho0,
        t_eval,
        liouvillian=liouvillian,
        method="BDF",
        rtol=PSEUDOMODE.rtol,
        atol=PSEUDOMODE.atol,
    )
    print(
        f"Sparse HEOM propagation: {perf_counter() - start:.3f} s "
        f"[nfev={result.nfev}, njev={result.njev}, nlu={result.nlu}]"
    )
    return np.real(result.expectation(sigmaz().full()))


def load_mlp(hierarchy, rho0):
    """Rebuild the configured architecture and load its trained weights."""
    device = torch.device(MLP.device)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=MLP.hidden_sizes,
        rho0=rho0,
        t_start=PSEUDOMODE.t_start,
        t_stop=PSEUDOMODE.t_stop,
        activation=MLP.activation,
        dtype=getattr(torch, MLP.dtype),
        device=device,
    )
    model.load_state_dict(
        torch.load(MLP_MODEL_PATH, map_location=device, weights_only=True)
    )
    print(f"Loaded MLP model: {MLP_MODEL_PATH}")
    return model


def run_mlp_solver(model, t_eval):
    """Evaluate at physical times; ``HEOMMLP`` normalizes them internally."""
    start = perf_counter()
    result = solve_mlp(
        model,
        t_eval,
        batch_size=MLP.inference_batch_size,
    )
    print(f"MLP trajectory evaluation: {perf_counter() - start:.3f} s")
    return np.real(result.expectation(sigmaz().full()))


def plot_trajectories(t_eval, lindbladian, sparse_heom, mlp, *, show, output):
    _, axis = plt.subplots(dpi=200)
    axis.plot(t_eval, lindbladian, "b-", label="Lindbladian numerics")
    axis.plot(
        t_eval,
        sparse_heom,
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=rf"Sparse normalized HEOM, $L={PSEUDOMODE.heom_depth}$",
    )
    axis.plot(
        t_eval,
        mlp,
        color="#7B2CBF",
        linestyle="--",
        label="MLP solver",
    )
    axis.set_title(
        rf"$g={PSEUDOMODE.g / PSEUDOMODE.w0:g}\,\omega_0$",
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
    args = parser.parse_args(argv)
    if args.no_show:
        plt.switch_backend("Agg")

    t_eval = np.linspace(
        PSEUDOMODE.t_start,
        PSEUDOMODE.t_stop,
        PSEUDOMODE.n_times,
    )
    lindbladian = run_lindbladian(t_eval)
    hierarchy, rho0, liouvillian = build_normalized_hard_heom()
    sparse_heom = run_sparse_numerics(
        hierarchy,
        rho0,
        liouvillian,
        t_eval,
    )
    mlp = run_mlp_solver(load_mlp(hierarchy, rho0), t_eval)

    print(
        "Max |sparse HEOM - Lindbladian|: "
        f"{np.max(np.abs(sparse_heom - lindbladian)):.3e}"
    )
    print(
        "Max |MLP - sparse HEOM|: "
        f"{np.max(np.abs(mlp - sparse_heom)):.3e}"
    )
    plot_trajectories(
        t_eval,
        lindbladian,
        sparse_heom,
        mlp,
        show=not args.no_show,
        output=args.output,
    )


if __name__ == "__main__":
    main()
