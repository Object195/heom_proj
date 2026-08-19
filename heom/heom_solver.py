"""Time-integration and spectral-diagnostic helpers for sparse HEOM."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eig
from scipy.integrate import solve_ivp

from .heom_rep import heom_state


@dataclass(frozen=True)
class HEOMSolution:
    """Result of integrating a :class:`heom_state` hierarchy."""

    t: np.ndarray
    y: np.ndarray
    system_dimension: int
    success: bool
    message: str
    nfev: int
    njev: int
    nlu: int

    @property
    def primary_ados(self):
        """Return the physical density matrix at every output time."""
        system_size = self.system_dimension**2
        root_vectors = self.y[:system_size, :]
        return np.stack([
            root_vectors[:, i].reshape(
                (self.system_dimension, self.system_dimension), order="F"
            )
            for i in range(root_vectors.shape[1])
        ])

    def expectation(self, operator):
        """Evaluate ``Tr(operator @ rho_0(t))`` along the trajectory."""
        operator = np.asarray(operator, dtype=np.complex128)
        expected_shape = (self.system_dimension, self.system_dimension)
        if operator.shape != expected_shape:
            raise ValueError(
                f"operator has shape {operator.shape}, expected {expected_shape}"
            )
        values = np.einsum("tij,ji->t", self.primary_ados, operator)
        return np.real_if_close(values)


@dataclass(frozen=True)
class HEOMSpectrum:
    """Eigenvalue diagnostic returned by :func:`diagnose_heom_spectrum`.

    ``eigenvalues`` and ``right_eigenvectors`` contain the complete spectrum.
    ``selected`` identifies the modes included in the plot and in
    ``largest_real_part``.  When an initial state is supplied,
    ``mode_amplitudes`` contains its coefficients in the right-eigenvector
    basis; otherwise it is ``None``.
    """

    eigenvalues: np.ndarray
    right_eigenvectors: np.ndarray
    selected: np.ndarray
    mode_amplitudes: np.ndarray | None
    largest_real_part: float

    @property
    def selected_eigenvalues(self):
        """Return the eigenvalues included in the diagnostic plot."""
        return self.eigenvalues[self.selected]


def _coerce_liouvillian(heom, liouvillian):
    """Return a complex CSR Liouvillian with dimensions matching ``heom``."""
    if not sp.issparse(liouvillian):
        liouvillian = sp.csr_array(liouvillian, dtype=np.complex128)
    else:
        if liouvillian.format != "csr":
            liouvillian = liouvillian.tocsr()
        if liouvillian.dtype != np.dtype(np.complex128):
            liouvillian = liouvillian.astype(np.complex128)

    expected_size = heom.nADO * heom.system_size
    if liouvillian.shape != (expected_size, expected_size):
        raise ValueError(
            f"liouvillian has shape {liouvillian.shape}, expected "
            f"{(expected_size, expected_size)}"
        )
    return liouvillian


def build_heom_ode(heom):
    """Build the constant sparse Jacobian and right-hand side ``L @ y``."""
    if not isinstance(heom, heom_state):
        raise TypeError("heom must be an instance of heom_state")

    liouvillian = heom.build_Liouvillian()

    def rhs(_time, state_vector):
        return liouvillian @ state_vector

    return liouvillian, rhs


def diagnose_heom_spectrum(
    heom,
    rho0=None,
    *,
    liouvillian=None,
    relevant_only=False,
    relevance_rtol=1e-9,
    relevance_atol=1e-12,
    ax=None,
    show=True,
):
    """Diagonalize and plot the complete HEOM Liouvillian spectrum.

    Parameters
    ----------
    heom : heom_state
        Hierarchy whose Liouvillian is to be diagnosed.
    rho0 : array_like, optional
        Initial physical density matrix.  It is embedded into the full HEOM
        state using :meth:`heom_state.build_initial_state`.  It is required
        when ``relevant_only=True``.
    liouvillian : array_like or sparse matrix, optional
        A previously assembled Liouvillian.  Passing it avoids rebuilding the
        operator (and preserves choices such as a Markovian terminator).
    relevant_only : bool, default=False
        Plot and report only modes with a significant coefficient in the
        expansion of the supplied initial HEOM state in right eigenvectors.
    relevance_rtol, relevance_atol : float
        A mode with coefficient ``c`` is retained when
        ``abs(c) > relevance_atol + relevance_rtol * max(abs(c))``.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw.  New axes are created when omitted.
    show : bool, default=True
        Call :func:`matplotlib.pyplot.show` after drawing.

    Returns
    -------
    HEOMSpectrum
        The full eigensystem, modal selection, amplitudes, and the largest
        real part among the plotted eigenvalues.

    Notes
    -----
    Computing every eigenvalue requires converting the sparse Liouvillian to
    a dense array and scales cubically with its dimension.  Modal relevance is
    based on the expansion ``initial_state = eigenvectors @ amplitudes``.  A
    least-squares expansion is used so nearly defective, non-normal operators
    can still be inspected.
    """
    if not isinstance(heom, heom_state):
        raise TypeError("heom must be an instance of heom_state")
    if not isinstance(relevant_only, (bool, np.bool_)):
        raise TypeError("relevant_only must be a boolean")
    for name, tolerance in (
        ("relevance_rtol", relevance_rtol),
        ("relevance_atol", relevance_atol),
    ):
        if (
            not np.isscalar(tolerance)
            or not np.isrealobj(tolerance)
            or not np.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError(f"{name} must be a finite, non-negative scalar")
    if relevant_only and rho0 is None:
        raise ValueError("rho0 is required when relevant_only=True")

    if liouvillian is None:
        liouvillian = heom.build_Liouvillian()
    liouvillian = _coerce_liouvillian(heom, liouvillian)

    eigenvalues, right_eigenvectors = eig(liouvillian.toarray())
    mode_amplitudes = None
    relevant = np.ones(eigenvalues.size, dtype=bool)
    if rho0 is not None:
        initial_state = heom.build_initial_state(rho0, as_sparse=False)
        mode_amplitudes = np.linalg.lstsq(
            right_eigenvectors,
            initial_state,
            rcond=None,
        )[0]
        amplitude_scale = np.max(np.abs(mode_amplitudes), initial=0.0)
        threshold = relevance_atol + relevance_rtol * amplitude_scale
        relevant = np.abs(mode_amplitudes) > threshold

    selected = relevant if relevant_only else np.ones(eigenvalues.size, dtype=bool)
    selected_eigenvalues = eigenvalues[selected]
    if selected_eigenvalues.size == 0:
        raise ValueError(
            "No eigenmodes pass the relevance threshold; reduce "
            "relevance_rtol or relevance_atol."
        )

    # Keep matplotlib optional for users who only propagate HEOM trajectories.
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(selected_eigenvalues.real, selected_eigenvalues.imag)
    ax.axvline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$\operatorname{Re}(\lambda)$")
    ax.set_ylabel(r"$\operatorname{Im}(\lambda)$")
    qualifier = "Relevant " if relevant_only else ""
    ax.set_title(f"{qualifier}HEOM Liouvillian eigenvalues")
    ax.grid(True, alpha=0.3)

    largest_real_part = float(np.max(selected_eigenvalues.real))
    print(
        f"Largest real part ({qualifier.lower()}eigenvalues): "
        f"{largest_real_part:.12g}"
    )
    if relevant_only:
        print(f"Relevant eigenvalues: {selected_eigenvalues.size}/{eigenvalues.size}")
    if show:
        plt.show()

    return HEOMSpectrum(
        eigenvalues=eigenvalues,
        right_eigenvectors=right_eigenvectors,
        selected=selected,
        mode_amplitudes=mode_amplitudes,
        largest_real_part=largest_real_part,
    )


def solve_heom(
    heom,
    rho0,
    t_eval,
    *,
    liouvillian=None,
    method="BDF",
    rtol=1e-8,
    atol=1e-10,
    **solver_options,
):
    """Integrate a time-independent HEOM with SciPy's stiff BDF solver.

    The HEOM Liouvillian is supplied as the exact constant sparse Jacobian,
    avoiding a finite-difference Jacobian calculation. A prebuilt
    ``liouvillian`` may be passed when construction and propagation are timed
    separately.
    """
    if not isinstance(heom, heom_state):
        raise TypeError("heom must be an instance of heom_state")

    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval must be a one-dimensional array with 2+ times")
    if not np.all(np.isfinite(t_eval)) or np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval must contain finite, strictly increasing times")

    if liouvillian is None:
        liouvillian, rhs = build_heom_ode(heom)
    else:
        liouvillian = _coerce_liouvillian(heom, liouvillian)

        def rhs(_time, state_vector):
            return liouvillian @ state_vector

    initial_state = heom.build_initial_state(rho0, as_sparse=False)

    method_name = method.upper() if isinstance(method, str) else method
    if method_name in {"BDF", "RADAU"}:
        solver_options.setdefault("jac", liouvillian)

    ode_result = solve_ivp(
        rhs,
        (t_eval[0], t_eval[-1]),
        initial_state,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        **solver_options,
    )
    if not ode_result.success:
        raise RuntimeError(f"HEOM integration failed: {ode_result.message}")

    return HEOMSolution(
        t=ode_result.t,
        y=ode_result.y,
        system_dimension=heom.H_s.shape[0],
        success=ode_result.success,
        message=ode_result.message,
        nfev=ode_result.nfev,
        njev=ode_result.njev,
        nlu=ode_result.nlu,
    )


__all__ = [
    "HEOMSolution",
    "HEOMSpectrum",
    "build_heom_ode",
    "diagnose_heom_spectrum",
    "solve_heom",
]
