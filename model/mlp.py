"""Coordinate-based MLP for normalized, hard-truncated HEOM states.

The network follows Section III of ``HEOM_DL.pdf``: every ADO is queried by
its normalized hierarchy coordinate and a common time, and the rows are kept
in the BFS order defined by :class:`heom.heom_rep.heom_state`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from heom.heom_rep import heom_state


def _validate_hierarchy(hierarchy: heom_state) -> None:
    if not isinstance(hierarchy, heom_state):
        raise TypeError("hierarchy must be an instance of heom_state")


def _split_node(hierarchy: heom_state, node):
    """Return full-length n and m tuples without changing hierarchy order."""
    if hierarchy.has_complex_poles:
        n_vector, m_vector = node
    else:
        n_vector = node
        m_vector = (0,) * (hierarchy.K + 1)
    return tuple(n_vector), tuple(m_vector)


def hierarchy_fingerprint(hierarchy: heom_state) -> str:
    """Return a deterministic signature of hierarchy ordering and physics."""
    _validate_hierarchy(hierarchy)
    digest = hashlib.sha256()
    digest.update(np.asarray([hierarchy.K, hierarchy.L], dtype="<i8").tobytes())
    digest.update(repr(tuple(hierarchy.idx_to_node)).encode("utf-8"))
    digest.update(repr(tuple(hierarchy.hierarchy_modes)).encode("utf-8"))
    for name, value in (
        ("H_s", hierarchy.H_s),
        ("H_c", hierarchy.H_c),
        ("C_list", hierarchy.C_list),
        ("gamma_list", hierarchy.gamma_list),
    ):
        digest.update(name.encode("ascii"))
        if value is None:
            digest.update(b"<none>")
            continue
        array = np.ascontiguousarray(value, dtype=np.complex128)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.astype("<c16", copy=False).tobytes())
    return digest.hexdigest()


def hierarchy_coordinates(hierarchy: heom_state) -> np.ndarray:
    """Return the Section-III ADO coordinates in the hierarchy's BFS order.

    The result has shape ``(nADO, 2 * (K + 1))`` and contains ``[n/L, m/L]``.
    Inactive ``m`` coordinates for real poles are retained as zeros so the
    documented input width ``2*K + 3`` (after adding time) is preserved.
    For the root-only ``L=0`` hierarchy, all normalized coordinates are zero.
    """
    _validate_hierarchy(hierarchy)
    scale = float(hierarchy.L) if hierarchy.L else 1.0
    coordinates = np.empty(
        (hierarchy.nADO, 2 * (hierarchy.K + 1)),
        dtype=np.float64,
    )
    for ado_index, node in enumerate(hierarchy.idx_to_node):
        n_vector, m_vector = _split_node(hierarchy, node)
        coordinates[ado_index] = np.asarray(
            n_vector + m_vector,
            dtype=np.float64,
        ) / scale
    return coordinates


def conjugate_ado_permutation(hierarchy: heom_state) -> np.ndarray:
    """Map every BFS ADO index to the index of its adjoint partner.

    Complex-pole ``n_k`` and ``m_k`` occupations are swapped.  A real pole has
    only one hierarchy coordinate, so its occupation is unchanged.  The
    returned permutation is an involution and can therefore enforce the
    HEOM adjoint symmetry with one vectorized gather.
    """
    _validate_hierarchy(hierarchy)
    complex_modes = set(hierarchy.complex_modes)
    permutation = np.empty(hierarchy.nADO, dtype=np.int64)

    for ado_index, node in enumerate(hierarchy.idx_to_node):
        n_vector, m_vector = _split_node(hierarchy, node)
        partner_n = list(n_vector)
        partner_m = list(m_vector)
        for mode in complex_modes:
            partner_n[mode], partner_m[mode] = (
                partner_m[mode],
                partner_n[mode],
            )

        if hierarchy.has_complex_poles:
            partner = tuple(partner_n), tuple(partner_m)
        else:
            partner = tuple(partner_n)
        try:
            permutation[ado_index] = hierarchy.node_to_idx[partner]
        except KeyError as error:  # pragma: no cover - hierarchy invariant
            raise ValueError(
                f"Adjoint partner {partner!r} is absent from the hierarchy"
            ) from error

    indices = np.arange(hierarchy.nADO, dtype=np.int64)
    if (
        np.unique(permutation).size != hierarchy.nADO
        or not np.array_equal(permutation[permutation], indices)
    ):
        raise ValueError("The hierarchy adjoint mapping is not an involution")
    return permutation


def normalized_adjoint_factors(hierarchy: heom_state) -> np.ndarray:
    r"""Return phase factors in the normalized ADO adjoint relation.

    If ``s_i`` is the stepwise normalization used by ``heom_rep.py`` and
    ``p(i)`` is the conjugate ADO index, normalized states obey

    ``rho_tilde_i = conj(s_p(i)) / s_i * rho_tilde_p(i)^dagger``.

    Magnitudes of partner scales agree, so only their phases are needed.  For
    the all-complex free-pole hierarchy assumed in Section III every factor is
    one.  Nontrivial factors make the same projection valid for collapsed real
    poles with complex or negative coefficients and for mixed hierarchies.
    """
    _validate_hierarchy(hierarchy)
    if hierarchy.C_list is None:
        raise ValueError("C_list is required for normalized ADO symmetry")
    coefficients = np.asarray(hierarchy.C_list, dtype=np.complex128)
    if np.any(coefficients == 0):
        raise ValueError("Normalized ADO symmetry requires non-zero C_list")

    n_steps = np.sqrt(coefficients)
    m_steps = np.sqrt(np.conj(coefficients))
    n_phases = n_steps / np.abs(n_steps)
    m_phases = m_steps / np.abs(m_steps)
    scale_phases = np.ones(hierarchy.nADO, dtype=np.complex128)
    for ado_index, node in enumerate(hierarchy.idx_to_node):
        n_vector, m_vector = _split_node(hierarchy, node)
        phase = 1.0 + 0.0j
        for mode, occupation in enumerate(n_vector):
            phase *= n_phases[mode] ** occupation
        for mode, occupation in enumerate(m_vector):
            phase *= m_phases[mode] ** occupation
        scale_phases[ado_index] = phase

    permutation = conjugate_ado_permutation(hierarchy)
    factors = np.conj(scale_phases[permutation]) / scale_phases
    factors /= np.abs(factors)
    factors[np.isclose(factors, 1.0, rtol=1e-13, atol=1e-15)] = 1.0
    if not np.allclose(
        factors * np.conj(factors[permutation]),
        1.0,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Normalized ADO adjoint factors are inconsistent")
    return factors


def column_vector_to_matrix(vector: torch.Tensor, dimension: int) -> torch.Tensor:
    """Unvectorize the final axis using NumPy/HEOM column-major ordering."""
    if not isinstance(vector, torch.Tensor):
        raise TypeError("vector must be a torch.Tensor")
    if vector.shape[-1] != dimension**2:
        raise ValueError(
            f"vector has final size {vector.shape[-1]}, expected {dimension**2}"
        )
    return vector.reshape(*vector.shape[:-1], dimension, dimension).transpose(
        -2, -1
    )


def matrix_to_column_vector(matrix: torch.Tensor) -> torch.Tensor:
    """Vectorize square matrices on the final axes in column-major order."""
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("matrix must be a torch.Tensor")
    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError("matrix must have square final two axes")
    dimension = matrix.shape[-1]
    return matrix.transpose(-2, -1).reshape(*matrix.shape[:-2], dimension**2)


def _activation(name: str) -> nn.Module:
    activations = {
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }
    try:
        return activations[name.lower()]()
    except (AttributeError, KeyError) as error:
        choices = ", ".join(sorted(activations))
        raise ValueError(f"activation must be one of: {choices}") from error


class HEOMMLP(nn.Module):
    """Shared MLP that predicts every ADO at one or more times.

    Parameters
    ----------
    hierarchy
        The hierarchy that fixes the BFS row order and ADO coordinates.
    hidden_sizes
        Width of each hidden layer.  An empty sequence gives one linear map.
    activation
        Smooth ``"tanh"`` is the default because the loss differentiates the
        prediction with respect to time.
    dtype, device
        PyTorch parameter and buffer placement.  ``float64`` matches the
        complex128 HEOM Liouvillian constructed by the repository.

    Notes
    -----
    ``forward(times)`` returns ``[U, V]`` with shape ``(batch, 2*N)``, where
    ``N = nADO * d**2``.  Each half is ADO-major and each ADO is column-major,
    exactly matching ``heom_state.build_Liouvillian``.
    """

    def __init__(
        self,
        hierarchy: heom_state,
        hidden_sizes: Sequence[int] = (64, 64, 64),
        *,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        _validate_hierarchy(hierarchy)
        hidden_sizes = tuple(hidden_sizes)
        if any(not isinstance(width, int) or width <= 0 for width in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64")
        requested_device = torch.device("cpu" if device is None else device)
        if requested_device.type not in ("cpu", "cuda"):
            raise ValueError("HEOMMLP currently supports CPU and CUDA devices")
        for name, operator in (
            ("H_s", hierarchy.H_s),
            ("H_c", hierarchy.H_c),
        ):
            if not np.allclose(
                operator,
                np.asarray(operator).conj().T,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{name} must be Hermitian for the symmetry-constrained MLP"
                )

        self.hierarchy = hierarchy
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation.lower()
        self.n_ados = hierarchy.nADO
        self.system_dimension = hierarchy.H_s.shape[0]
        self.system_size = self.system_dimension**2
        self.state_size = self.n_ados * self.system_size
        self.input_size = 2 * (hierarchy.K + 1) + 1
        self.output_size_per_ado = 2 * self.system_size
        self.hierarchy_fingerprint = hierarchy_fingerprint(hierarchy)

        factory_kwargs = {"dtype": dtype, "device": requested_device}
        self.register_buffer(
            "ado_coordinates",
            torch.as_tensor(
                hierarchy_coordinates(hierarchy),
                **factory_kwargs,
            ),
            persistent=False,
        )
        self.register_buffer(
            "conjugate_indices",
            torch.as_tensor(
                conjugate_ado_permutation(hierarchy),
                dtype=torch.long,
                device=requested_device,
            ),
            persistent=False,
        )
        adjoint_factors = normalized_adjoint_factors(hierarchy)
        self.register_buffer(
            "adjoint_factor_real",
            torch.as_tensor(
                adjoint_factors.real,
                dtype=dtype,
                device=requested_device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "adjoint_factor_imag",
            torch.as_tensor(
                adjoint_factors.imag,
                dtype=dtype,
                device=requested_device,
            ),
            persistent=False,
        )

        widths = (self.input_size,) + hidden_sizes
        layers: list[nn.Module] = []
        for input_width, output_width in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(input_width, output_width, **factory_kwargs))
            layers.append(_activation(self.activation_name))
        layers.append(
            nn.Linear(widths[-1], self.output_size_per_ado, **factory_kwargs)
        )
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    @property
    def dtype(self) -> torch.dtype:
        return self.ado_coordinates.dtype

    @property
    def device(self) -> torch.device:
        return self.ado_coordinates.device

    def reset_parameters(self) -> None:
        """Apply a deterministic-family Xavier initialization to linear layers."""
        gain_name = "tanh" if self.activation_name == "tanh" else "linear"
        hidden_gain = nn.init.calculate_gain(gain_name)
        linear_layers = [
            layer for layer in self.network if isinstance(layer, nn.Linear)
        ]
        for layer in linear_layers[:-1]:
            nn.init.xavier_uniform_(layer.weight, gain=hidden_gain)
            nn.init.zeros_(layer.bias)
        output_layer = linear_layers[-1]
        nn.init.xavier_uniform_(output_layer.weight, gain=1.0)
        nn.init.zeros_(output_layer.bias)

    def prepare_times(self, times) -> torch.Tensor:
        """Coerce scalar or one-dimensional times to the model placement."""
        if self.dtype not in (torch.float32, torch.float64):
            raise RuntimeError(
                "HEOMMLP parameters must remain torch.float32 or torch.float64"
            )
        if self.device.type not in ("cpu", "cuda"):
            raise RuntimeError("HEOMMLP supports CPU and CUDA devices")
        times = torch.as_tensor(times, dtype=self.dtype, device=self.device)
        if times.ndim == 0:
            times = times.reshape(1)
        elif times.ndim != 1:
            raise ValueError("times must be a scalar or one-dimensional tensor")
        return times

    def coordinate_inputs(self, times) -> torch.Tensor:
        """Build ``(batch, nADO, 2*K+3)`` Section-III inputs."""
        times = self.prepare_times(times)
        batch_size = times.shape[0]
        coordinates = self.ado_coordinates.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        time_column = times[:, None, None].expand(-1, self.n_ados, 1)
        return torch.cat((coordinates, time_column), dim=-1)

    def raw_output(self, times) -> torch.Tensor:
        """Return unsymmetrized real/imaginary ADO channels."""
        return self.network(self.coordinate_inputs(times))

    def symmetrize_raw(self, raw_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Enforce normalized HEOM adjoint symmetry and return flat ``U, V``.

        The normalization factor is one for Section III's all-complex case;
        its phase is retained for compatible real/mixed-pole hierarchies.
        """
        expected_shape = (self.n_ados, self.output_size_per_ado)
        if raw_output.ndim != 3 or tuple(raw_output.shape[1:]) != expected_shape:
            raise ValueError(
                "raw_output must have shape "
                f"(batch, {expected_shape[0]}, {expected_shape[1]})"
            )

        raw_u, raw_v = raw_output.split(self.system_size, dim=-1)
        matrix_u = column_vector_to_matrix(raw_u, self.system_dimension)
        matrix_v = column_vector_to_matrix(raw_v, self.system_dimension)

        partner_u_transpose = matrix_u.index_select(
            1, self.conjugate_indices
        ).transpose(-2, -1)
        partner_v_transpose = matrix_v.index_select(
            1, self.conjugate_indices
        ).transpose(-2, -1)
        factor_real = self.adjoint_factor_real[None, :, None, None]
        factor_imag = self.adjoint_factor_imag[None, :, None, None]
        # (a + ib)(U_partner^T - i V_partner^T) implements the
        # normalization-aware factor multiplying the partner's adjoint.
        projected_partner_u = (
            factor_real * partner_u_transpose
            + factor_imag * partner_v_transpose
        )
        projected_partner_v = (
            factor_imag * partner_u_transpose
            - factor_real * partner_v_transpose
        )
        symmetric_u = 0.5 * (matrix_u + projected_partner_u)
        symmetric_v = 0.5 * (matrix_v + projected_partner_v)

        flat_u = matrix_to_column_vector(symmetric_u).reshape(
            raw_output.shape[0], self.state_size
        )
        flat_v = matrix_to_column_vector(symmetric_v).reshape(
            raw_output.shape[0], self.state_size
        )
        return flat_u, flat_v

    def forward(self, times) -> torch.Tensor:
        """Predict globally blocked ``[Re(chi), Im(chi)]`` HEOM states."""
        real_state, imaginary_state = self.symmetrize_raw(self.raw_output(times))
        return torch.cat((real_state, imaginary_state), dim=-1)

    def complex_states(self, times) -> torch.Tensor:
        """Predict complex stacked HEOM states with shape ``(batch, N)``."""
        real_state, imaginary_state = self(times).split(self.state_size, dim=-1)
        return torch.complex(real_state, imaginary_state)

    def root_density_matrices(self, times) -> torch.Tensor:
        """Return the physical root ADO as complex ``(batch, d, d)`` matrices."""
        root_vectors = self.complex_states(times)[:, : self.system_size]
        return column_vector_to_matrix(root_vectors, self.system_dimension)

    def expectation(self, times, operator) -> torch.Tensor:
        """Evaluate ``Tr(operator @ rho_root(t))`` without leaving PyTorch."""
        density_matrices = self.root_density_matrices(times)
        operator = torch.as_tensor(
            operator,
            dtype=density_matrices.dtype,
            device=density_matrices.device,
        )
        expected_shape = (self.system_dimension, self.system_dimension)
        if tuple(operator.shape) != expected_shape:
            raise ValueError(
                f"operator has shape {tuple(operator.shape)}, "
                f"expected {expected_shape}"
            )
        return torch.einsum("bij,ji->b", density_matrices, operator)


def state_and_time_derivative(
    model: HEOMMLP,
    times,
    *,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return states and ``d state / dt`` using one batched JVP.

    Each output batch row depends only on the corresponding scalar time.  A
    vector of unit tangents therefore yields every row's full time derivative
    without constructing a dense output-by-time Jacobian or looping over state
    components.  Symmetry enforcement stays inside the differentiated call.
    """
    if not isinstance(model, HEOMMLP):
        raise TypeError("model must be an instance of HEOMMLP")
    times = model.prepare_times(times)
    return torch.autograd.functional.jvp(
        model,
        times,
        torch.ones_like(times),
        create_graph=create_graph,
        strict=False,
    )


__all__ = [
    "HEOMMLP",
    "column_vector_to_matrix",
    "conjugate_ado_permutation",
    "hierarchy_fingerprint",
    "hierarchy_coordinates",
    "matrix_to_column_vector",
    "normalized_adjoint_factors",
    "state_and_time_derivative",
]
