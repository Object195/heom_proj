"""Section-III coordinate MLP for normalized free-pole HEOM states."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch
from torch import nn

from heom import q_func
from heom.heom_rep import heom_state


def hierarchy_coordinates(
    hierarchy: heom_state,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Return ``[n/L, m/L]`` or raw ``[n, m]`` rows in BFS order."""
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")
    coordinates = np.asarray(
        [
            np.asarray(n + m, dtype=np.float64)
            for n, m in hierarchy.idx_to_node
        ],
        dtype=np.float64,
    )
    return coordinates / hierarchy.L if normalize else coordinates


def conjugate_ado_permutation(hierarchy: heom_state) -> np.ndarray:
    """Return the BFS index of ``(m, n)`` for every ``(n, m)`` ADO."""
    return np.asarray(
        [hierarchy.node_to_idx[(m, n)] for n, m in hierarchy.idx_to_node],
        dtype=np.int64,
    )


def column_vector_to_matrix(vector: torch.Tensor, dimension: int) -> torch.Tensor:
    """Unvectorize the final axis using the HEOM column-major convention."""
    return vector.reshape(*vector.shape[:-1], dimension, dimension).transpose(
        -2, -1
    )


def matrix_to_column_vector(matrix: torch.Tensor) -> torch.Tensor:
    """Column-vectorize matrices stored on the final two axes."""
    dimension = matrix.shape[-1]
    return matrix.transpose(-2, -1).reshape(*matrix.shape[:-2], dimension**2)


def _activation(name: str) -> nn.Module:
    return {
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }[name.lower()]()


class HEOMMLP(nn.Module):
    """Shared MLP evaluated at every BFS-ordered ADO coordinate.

    ``forward(times)`` returns the constrained physical state
    ``initial_state + s * correction_scale * correction`` with shape
    ``(batch, 2*N)``. The real and imaginary halves use ADO-major,
    column-major ordering, matching the sparse Liouvillian from
    ``heom_state.build_Liouvillian``. With ``positive_rdm_ansatz=True``,
    only the root is replaced by ``A A^dagger / Tr(A A^dagger)``, where
    ``A = sqrt(rho_initial) + s * correction_scale * B`` and ``B`` is the
    unconstrained root network output.
    """

    def __init__(
        self,
        hierarchy: heom_state,
        hidden_sizes: Sequence[int] = (64, 64, 64),
        *,
        rho0=None,
        initial_heom_state=None,
        t_start: float,
        t_stop: float,
        activation: str = "tanh",
        time_switch: str = "linear",
        switch_time_constant: float = 1.0,
        normalize_hierarchy_coordinates: bool = True,
        positive_rdm_ansatz: bool = False,
        correction_scale: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.hierarchy = hierarchy
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation_name = activation.lower()
        self.n_ados = hierarchy.nADO
        self.system_dimension = hierarchy.H_s.shape[0]
        self.system_size = self.system_dimension**2
        self.state_size = self.n_ados * self.system_size
        self.input_size = 2 * (hierarchy.K + 1) + 1
        if not isinstance(normalize_hierarchy_coordinates, bool):
            raise TypeError(
                "normalize_hierarchy_coordinates must be a boolean"
            )
        self.normalize_hierarchy_coordinates = (
            normalize_hierarchy_coordinates
        )
        if not isinstance(positive_rdm_ansatz, bool):
            raise TypeError("positive_rdm_ansatz must be a boolean")
        self.positive_rdm_ansatz = positive_rdm_ansatz
        self.t_start = float(t_start)
        self.t_stop = float(t_stop)
        self.time_span = self.t_stop - self.t_start
        if not math.isfinite(self.time_span) or self.time_span <= 0.0:
            raise ValueError("t_stop must be finite and greater than t_start")
        if not isinstance(time_switch, str):
            raise ValueError("time_switch must be 'linear' or 'exponential'")
        self.time_switch = time_switch.lower()
        if self.time_switch not in {"linear", "exponential"}:
            raise ValueError("time_switch must be 'linear' or 'exponential'")
        self.switch_time_constant = float(switch_time_constant)
        if (
            not math.isfinite(self.switch_time_constant)
            or self.switch_time_constant <= 0.0
        ):
            raise ValueError("switch_time_constant must be finite and positive")
        correction_scale = float(correction_scale)
        if not math.isfinite(correction_scale) or correction_scale <= 0.0:
            raise ValueError("correction_scale must be finite and positive")
        self.normalized_switch_time_constant = (
            self.switch_time_constant / self.time_span
        )
        self._exponential_switch_denominator = -math.expm1(
            -1.0 / self.normalized_switch_time_constant
        )

        self.register_buffer(
            "ado_coordinates",
            torch.as_tensor(
                hierarchy_coordinates(
                    hierarchy,
                    normalize=normalize_hierarchy_coordinates,
                ),
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "conjugate_indices",
            torch.as_tensor(
                conjugate_ado_permutation(hierarchy),
                dtype=torch.long,
                device=device,
            ),
            persistent=False,
        )
        if (rho0 is None) == (initial_heom_state is None):
            raise ValueError(
                "pass exactly one of rho0 or initial_heom_state"
            )
        if initial_heom_state is None:
            initial_complex = hierarchy.build_initial_state(
                rho0,
                as_sparse=False,
            )
        else:
            initial_complex = np.asarray(
                initial_heom_state,
                dtype=np.complex128,
            )
            if initial_complex.shape != (self.state_size,):
                raise ValueError(
                    "initial_heom_state must be a flat full HEOM vector "
                    f"with shape ({self.state_size},)"
                )
            if not np.isfinite(initial_complex).all():
                raise ValueError(
                    "initial_heom_state must contain only finite values"
                )
        self.register_buffer(
            "initial_state",
            torch.as_tensor(
                q_func.state_to_real(initial_complex),
                dtype=dtype,
                device=device,
            ),
        )
        self.register_buffer(
            "correction_scale",
            torch.as_tensor(correction_scale, dtype=dtype, device=device),
            persistent=False,
        )
        if self.positive_rdm_ansatz:
            for name in ("root_sqrt_real", "root_sqrt_imag"):
                self.register_buffer(
                    name,
                    torch.zeros(
                        self.system_dimension, self.system_dimension,
                        dtype=dtype, device=device,
                    ),
                    persistent=False,
                )
            self._refresh_root_factor()
            self.register_load_state_dict_post_hook(
                self._refresh_root_factor_after_load
            )
        root_diagonal_indices = np.arange(self.system_dimension) * (
            self.system_dimension + 1
        )
        self.register_buffer(
            "root_diagonal_indices",
            torch.as_tensor(
                root_diagonal_indices,
                dtype=torch.long,
                device=device,
            ),
            persistent=False,
        )
        root_identity = np.zeros(self.state_size)
        root_identity[root_diagonal_indices] = 1.0
        self.register_buffer(
            "root_identity",
            torch.as_tensor(root_identity, dtype=dtype, device=device),
            persistent=False,
        )

        widths = (self.input_size,) + self.hidden_sizes
        layers: list[nn.Module] = []
        for input_width, output_width in zip(widths[:-1], widths[1:]):
            layers.extend(
                (
                    nn.Linear(input_width, output_width, dtype=dtype, device=device),
                    _activation(self.activation_name),
                )
            )
        layers.append(
            nn.Linear(
                widths[-1],
                2 * self.system_size,
                dtype=dtype,
                device=device,
            )
        )
        self.network = nn.Sequential(*layers)

    def _refresh_root_factor(self) -> None:
        """Validate the anchor and cache its principal PSD square root.

        Only floating-point roundoff is repaired. A materially nonphysical
        root (e.g. from unconverged truncated HEOM) must not be silently
        projected onto a different initial condition.
        """
        initial = self.complex_initial_state().detach().cpu().numpy()
        root = initial[:self.system_size].reshape(
            self.system_dimension, self.system_dimension, order="F"
        )
        tolerance = 100 * torch.finfo(self.dtype).eps * self.system_dimension
        if not np.isfinite(root).all():
            raise ValueError("positive RDM ansatz requires a finite initial RDM")
        if np.max(np.abs(root - root.conj().T)) > tolerance:
            raise ValueError("positive RDM ansatz requires a Hermitian initial RDM")
        if abs(np.trace(root) - 1.0) > tolerance:
            raise ValueError("positive RDM ansatz requires a unit-trace initial RDM")
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (root + root.conj().T))
        if eigenvalues.min() < -tolerance:
            raise ValueError(
                "positive RDM ansatz requires a positive-semidefinite initial "
                "RDM; check the initial state or HEOM truncation/solver accuracy"
            )
        eigenvalues = np.maximum(eigenvalues, 0.0)
        eigenvalues /= eigenvalues.sum()
        root = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
        root_sqrt = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.conj().T
        # The same roundoff-cleaned root is used by the anchor, loss scales,
        # saved state, and square-root factor.
        flat = root.reshape(-1, order="F")
        with torch.no_grad():
            self.initial_state[:self.system_size].copy_(
                torch.as_tensor(flat.real.copy(), dtype=self.dtype, device=self.device)
            )
            self.initial_state[self.state_size:self.state_size + self.system_size].copy_(
                torch.as_tensor(flat.imag.copy(), dtype=self.dtype, device=self.device)
            )
            self.root_sqrt_real.copy_(
                torch.as_tensor(root_sqrt.real.copy(), dtype=self.dtype, device=self.device)
            )
            self.root_sqrt_imag.copy_(
                torch.as_tensor(root_sqrt.imag.copy(), dtype=self.dtype, device=self.device)
            )

    def _refresh_root_factor_after_load(self, module, incompatible_keys) -> None:
        # initial_state is persistent and can change on load_state_dict;
        # derived buffers must follow it, including direct API checkpoint loads.
        self._refresh_root_factor()

    @property
    def dtype(self) -> torch.dtype:
        return self.ado_coordinates.dtype

    @property
    def device(self) -> torch.device:
        return self.ado_coordinates.device

    def prepare_times(self, times) -> torch.Tensor:
        """Convert physical times to a flat tensor on the model device."""
        return torch.as_tensor(
            times,
            dtype=self.dtype,
            device=self.device,
        ).reshape(-1)

    def normalize_times(self, times) -> torch.Tensor:
        """Map physical time from ``[t_start, t_stop]`` to ``[-1, 1]``."""
        times = self.prepare_times(times)
        return 2.0 * (times - self.t_start) / self.time_span - 1.0

    def switching_function(self, times) -> torch.Tensor:
        """Return the initial-condition switch at physical ``times``.

        The exponential switch is evaluated with normalized elapsed time
        ``u = (t - t_start) / (t_stop - t_start)`` and
        ``tau_c = switch_time_constant / (t_stop - t_start)``. Thus the
        configured time constant remains in physical time units while the
        numerical calculation uses the same time scale as the network input.
        ``expm1`` preserves accuracy when the time constant is large.
        """
        elapsed_fraction = 0.5 * (self.normalize_times(times) + 1.0)
        if self.time_switch == "linear":
            return elapsed_fraction
        numerator = -torch.expm1(
            -elapsed_fraction / self.normalized_switch_time_constant
        )
        return numerator / self._exponential_switch_denominator

    def coordinate_inputs(self, times) -> torch.Tensor:
        """Build the ``(batch, nADO, 2*K+3)`` MLP input tensor."""
        normalized_times = self.normalize_times(times)
        coordinates = self.ado_coordinates.expand(
            normalized_times.numel(), -1, -1
        )
        time_column = normalized_times[:, None, None].expand(
            -1, self.n_ados, 1
        )
        return torch.cat((coordinates, time_column), dim=-1)

    def raw_output(self, times) -> torch.Tensor:
        return self.network(self.coordinate_inputs(times))

    def symmetrize_raw(
        self,
        raw_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply ``rho(n,m) = rho(m,n)^dagger`` from Section III."""
        raw_u, raw_v = raw_output.split(self.system_size, dim=-1)
        matrix_u = column_vector_to_matrix(raw_u, self.system_dimension)
        matrix_v = column_vector_to_matrix(raw_v, self.system_dimension)
        partner_u = matrix_u[:, self.conjugate_indices].transpose(-2, -1)
        partner_v = matrix_v[:, self.conjugate_indices].transpose(-2, -1)
        symmetric_u = 0.5 * (matrix_u + partner_u)
        symmetric_v = 0.5 * (matrix_v - partner_v)
        flat_u = matrix_to_column_vector(symmetric_u).reshape(-1, self.state_size)
        flat_v = matrix_to_column_vector(symmetric_v).reshape(-1, self.state_size)
        return flat_u, flat_v

    def state_correction(self, times) -> torch.Tensor:
        """Return the legacy additive correction (used by the default ansatz)."""
        real_state, imaginary_state = self.symmetrize_raw(self.raw_output(times))
        real_trace = real_state.index_select(
            1, self.root_diagonal_indices
        ).sum(dim=1, keepdim=True)
        imaginary_trace = imaginary_state.index_select(
            1, self.root_diagonal_indices
        ).sum(dim=1, keepdim=True)
        real_state = real_state - (
            real_trace / self.system_dimension
        ) * self.root_identity
        imaginary_state = imaginary_state - (
            imaginary_trace / self.system_dimension
        ) * self.root_identity
        return torch.cat((real_state, imaginary_state), dim=-1)

    def forward(self, times) -> torch.Tensor:
        times = self.prepare_times(times)
        switch = self.switching_function(times)
        if self.positive_rdm_ansatz:
            raw = self.raw_output(times)
            root_u, root_v = self._positive_root(raw[:, 0], switch)
            symmetric_u, symmetric_v = self.symmetrize_raw(raw)
            initial_u, initial_v = self.initial_state.split(self.state_size)
            scale = switch[:, None] * self.correction_scale
            # Discard the symmetrized root: B is neither symmetrized nor
            # made traceless. All non-root ADOs keep their previous ansatz.
            return torch.cat(
                (
                    root_u,
                    initial_u[self.system_size:]
                    + scale * symmetric_u[:, self.system_size:],
                    root_v,
                    initial_v[self.system_size:]
                    + scale * symmetric_v[:, self.system_size:],
                ),
                dim=-1,
            )
        return (
            self.initial_state
            + switch[:, None]
            * self.correction_scale
            * self.state_correction(times)
        )

    def _positive_root(self, raw_root, switch):
        """Build the normalized Gram matrix using real arithmetic for JVPs.

        At A=0 the requested quotient has no continuous extension; choose
        the initial RDM there. For every nonzero A the formula is unchanged.
        Rescaling before forming the Gram matrix avoids overflow/underflow.
        """
        raw_u, raw_v = raw_root.split(self.system_size, dim=-1)
        scale = switch[:, None, None] * self.correction_scale
        factor_u = self.root_sqrt_real + scale * column_vector_to_matrix(
            raw_u, self.system_dimension
        )
        factor_v = self.root_sqrt_imag + scale * column_vector_to_matrix(
            raw_v, self.system_dimension
        )
        magnitude = torch.maximum(
            factor_u.detach().abs().amax(dim=(-2, -1), keepdim=True),
            factor_v.detach().abs().amax(dim=(-2, -1), keepdim=True),
        )
        zero = magnitude == 0
        factor_u = torch.where(zero, self.root_sqrt_real, factor_u)
        factor_v = torch.where(zero, self.root_sqrt_imag, factor_v)
        divisor = torch.where(zero, torch.ones_like(magnitude), magnitude)
        factor_u = factor_u / divisor
        factor_v = factor_v / divisor
        transpose_u = factor_u.transpose(-2, -1)
        transpose_v = factor_v.transpose(-2, -1)
        gram_u = factor_u @ transpose_u + factor_v @ transpose_v
        gram_v = factor_v @ transpose_u - factor_u @ transpose_v
        trace = (factor_u.square() + factor_v.square()).sum(
            dim=(-2, -1), keepdim=True
        )
        return (
            matrix_to_column_vector(gram_u / trace),
            matrix_to_column_vector(gram_v / trace),
        )

    def set_correction_scale(self, correction_scale: float) -> None:
        """Update the nonpersistent ansatz scale after loading a checkpoint."""
        correction_scale = float(correction_scale)
        if not math.isfinite(correction_scale) or correction_scale <= 0.0:
            raise ValueError("correction_scale must be finite and positive")
        with torch.no_grad():
            self.correction_scale.fill_(correction_scale)

    def complex_initial_state(self) -> torch.Tensor:
        """Return the constrained full HEOM initial state as a complex vector."""
        real_state, imaginary_state = self.initial_state.split(self.state_size)
        return torch.complex(real_state, imaginary_state)

    def complex_states(self, times) -> torch.Tensor:
        real_state, imaginary_state = self(times).split(self.state_size, dim=-1)
        return torch.complex(real_state, imaginary_state)

    def root_density_matrices(self, times) -> torch.Tensor:
        root = self.complex_states(times)[:, : self.system_size]
        return column_vector_to_matrix(root, self.system_dimension)

    def expectation(self, times, operator) -> torch.Tensor:
        density_matrices = self.root_density_matrices(times)
        operator = torch.as_tensor(
            operator,
            dtype=density_matrices.dtype,
            device=density_matrices.device,
        )
        return torch.einsum("bij,ji->b", density_matrices, operator)


def state_and_time_derivative(
    model: HEOMMLP,
    times,
    *,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the state and its derivative with respect to physical time.

    ``HEOMMLP.forward`` normalizes time and applies its configured switch
    internally. Differentiating the composed model here includes both
    corresponding chain-rule factors.
    """
    times = model.prepare_times(times)
    return torch.autograd.functional.jvp(
        model,
        times,
        torch.ones_like(times),
        create_graph=create_graph,
    )


__all__ = [
    "HEOMMLP",
    "column_vector_to_matrix",
    "conjugate_ado_permutation",
    "hierarchy_coordinates",
    "matrix_to_column_vector",
    "state_and_time_derivative",
]
