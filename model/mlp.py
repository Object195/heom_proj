"""Section-III coordinate MLP for normalized free-pole HEOM states."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from heom.heom_rep import heom_state


def hierarchy_coordinates(hierarchy: heom_state) -> np.ndarray:
    """Return ``[n/L, m/L]`` rows in the hierarchy's BFS order."""
    return np.asarray(
        [
            np.asarray(n + m, dtype=np.float64) / hierarchy.L
            for n, m in hierarchy.idx_to_node
        ],
        dtype=np.float64,
    )


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

    ``forward(times)`` returns ``[U, V]`` with shape ``(batch, 2*N)``.
    Both halves use ADO-major, column-major ordering, matching the sparse
    Liouvillian from ``heom_state.build_Liouvillian``.
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
        self.hierarchy = hierarchy
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation_name = activation.lower()
        self.n_ados = hierarchy.nADO
        self.system_dimension = hierarchy.H_s.shape[0]
        self.system_size = self.system_dimension**2
        self.state_size = self.n_ados * self.system_size
        self.input_size = 2 * (hierarchy.K + 1) + 1

        self.register_buffer(
            "ado_coordinates",
            torch.as_tensor(
                hierarchy_coordinates(hierarchy),
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

    @property
    def dtype(self) -> torch.dtype:
        return self.ado_coordinates.dtype

    @property
    def device(self) -> torch.device:
        return self.ado_coordinates.device

    def prepare_times(self, times) -> torch.Tensor:
        return torch.as_tensor(
            times,
            dtype=self.dtype,
            device=self.device,
        ).reshape(-1)

    def coordinate_inputs(self, times) -> torch.Tensor:
        """Build the ``(batch, nADO, 2*K+3)`` MLP input tensor."""
        times = self.prepare_times(times)
        coordinates = self.ado_coordinates.expand(times.numel(), -1, -1)
        time_column = times[:, None, None].expand(-1, self.n_ados, 1)
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

    def forward(self, times) -> torch.Tensor:
        real_state, imaginary_state = self.symmetrize_raw(self.raw_output(times))
        return torch.cat((real_state, imaginary_state), dim=-1)

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
    """Evaluate the state and its time derivative with one batched JVP."""
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
