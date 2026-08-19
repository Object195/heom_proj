"""Physics-informed loss, optimization loop, and inference for ``HEOMMLP``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from heom import q_func
from heom.heom_rep import heom_state

from .mlp import HEOMMLP, hierarchy_fingerprint, state_and_time_derivative


@dataclass(frozen=True)
class LossWeights:
    """Weights in ``omega_d L_d + omega_I L_IC + omega_t L_t``."""

    dynamics: float = 1.0
    initial_condition: float = 1.0
    trace: float = 1.0

    @classmethod
    def balanced(cls, q_x: int) -> "LossWeights":
        """Balance the explicitly summed/divided Section-III conventions.

        This makes the weighted IC term an unnormalized squared error and the
        weighted trace term a mean over ``q_x`` while leaving dynamics as Eq.
        (10).  It is a practical starting point; the physical weights remain
        user-tunable.
        """
        if not isinstance(q_x, int) or isinstance(q_x, bool) or q_x <= 0:
            raise ValueError("q_x must be a positive integer")
        return cls(initial_condition=float(q_x), trace=1.0 / q_x)

    def __post_init__(self) -> None:
        for name, value in (
            ("dynamics", self.dynamics),
            ("initial_condition", self.initial_condition),
            ("trace", self.trace),
        ):
            if not np.isscalar(value) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} weight must be finite and non-negative")


class LossTerms(NamedTuple):
    """Differentiable terms returned by :class:`HEOMPINNLoss`."""

    total: torch.Tensor
    dynamics: torch.Tensor
    initial_condition: torch.Tensor
    trace: torch.Tensor


def _coerce_liouvillian(
    hierarchy: heom_state,
    liouvillian,
) -> sp.csr_array:
    expected_size = hierarchy.nADO * hierarchy.system_size
    if sp.issparse(liouvillian):
        liouvillian = sp.csr_array(liouvillian, dtype=np.complex128)
    else:
        liouvillian = sp.csr_array(
            np.asarray(liouvillian, dtype=np.complex128)
        )
    if liouvillian.shape != (expected_size, expected_size):
        raise ValueError(
            f"liouvillian has shape {liouvillian.shape}, expected "
            f"{(expected_size, expected_size)}"
        )
    return liouvillian


def scipy_sparse_to_torch(
    matrix,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a coalesced PyTorch COO tensor."""
    if not sp.issparse(matrix):
        raise TypeError("matrix must be a SciPy sparse matrix")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")
    coordinate_matrix = sp.coo_array(matrix)
    coordinate_matrix.eliminate_zeros()
    indices = torch.as_tensor(
        np.vstack((coordinate_matrix.row, coordinate_matrix.col)),
        dtype=torch.long,
        device=device,
    )
    values = torch.as_tensor(
        coordinate_matrix.data,
        dtype=dtype,
        device=device,
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=coordinate_matrix.shape,
        dtype=dtype,
        device=device,
        check_invariants=True,
    ).coalesce()


class HEOMPINNLoss(nn.Module):
    """Section-III residual, initial-condition, and trace objective.

    If no Liouvillian is supplied, it is built *only* through
    ``heom_state.build_Liouvillian(markovian_terminator=False,
    normalized=True)``, i.e. the requested normalized hard truncation.  A
    prebuilt operator can be supplied so numerical and MLP solvers share the
    exact same sparse matrix.
    """

    def __init__(
        self,
        hierarchy: heom_state,
        rho0,
        *,
        liouvillian=None,
        weights: LossWeights = LossWeights(),
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(hierarchy, heom_state):
            raise TypeError("hierarchy must be an instance of heom_state")
        if not isinstance(weights, LossWeights):
            raise TypeError("weights must be a LossWeights instance")
        if liouvillian is None:
            liouvillian = hierarchy.build_Liouvillian(
                markovian_terminator=False,
                normalized=True,
            )
        else:
            expected_options = {
                "markovian_terminator": False,
                "normalized": True,
            }
            if (
                getattr(hierarchy, "liouvillian", None) is not liouvillian
                or getattr(hierarchy, "liouvillian_options", None)
                != expected_options
            ):
                raise ValueError(
                    "A supplied liouvillian must be the normalized hard-cutoff "
                    "matrix returned by the most recent call to "
                    "hierarchy.build_Liouvillian(markovian_terminator=False, "
                    "normalized=True)"
                )
        liouvillian = _coerce_liouvillian(hierarchy, liouvillian)

        self.hierarchy = hierarchy
        self.hierarchy_fingerprint = hierarchy_fingerprint(hierarchy)
        self.weights = weights
        self.n_ados = hierarchy.nADO
        self.system_dimension = hierarchy.H_s.shape[0]
        self.system_size = hierarchy.system_size
        self.state_size = self.n_ados * self.system_size

        # q_func performs the fixed SciPy preprocessing; all training-time
        # operations below stay in PyTorch and retain gradients.
        real_liouvillian = q_func.sup_op_to_real(liouvillian).tocsr()
        self.register_buffer(
            "real_liouvillian",
            scipy_sparse_to_torch(
                real_liouvillian,
                dtype=dtype,
                device=device,
            ),
        )

        rho0 = np.asarray(rho0, dtype=np.complex128)
        expected_rho_shape = (
            self.system_dimension,
            self.system_dimension,
        )
        if rho0.shape != expected_rho_shape:
            raise ValueError(
                f"rho0 has shape {rho0.shape}, expected {expected_rho_shape}"
            )
        if not np.allclose(rho0, rho0.conj().T, rtol=1e-10, atol=1e-12):
            raise ValueError("rho0 must be Hermitian for the symmetry-constrained MLP")
        if not np.allclose(np.trace(rho0), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("rho0 must have unit trace")
        initial_complex = hierarchy.build_initial_state(rho0, as_sparse=False)
        initial_real = q_func.state_to_real(initial_complex)
        self.register_buffer(
            "initial_state",
            torch.as_tensor(initial_real, dtype=dtype, device=device),
        )
        root_diagonal = np.arange(self.system_dimension, dtype=np.int64)
        root_diagonal *= self.system_dimension + 1
        self.register_buffer(
            "root_diagonal_indices",
            torch.as_tensor(root_diagonal, dtype=torch.long, device=device),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.initial_state.dtype

    @property
    def device(self) -> torch.device:
        return self.initial_state.device

    def _validate_model(self, model: HEOMMLP) -> None:
        if not isinstance(model, HEOMMLP):
            raise TypeError("model must be an instance of HEOMMLP")
        if model.hierarchy is not self.hierarchy:
            raise ValueError(
                "model and HEOMPINNLoss must share the exact hierarchy "
                "instance so their BFS ADO ordering cannot diverge"
            )
        if model.hierarchy_fingerprint != self.hierarchy_fingerprint:
            raise ValueError(
                "model and HEOMPINNLoss were constructed from different "
                "hierarchy physics"
            )
        if model.state_size != self.state_size:
            raise ValueError(
                f"model state size is {model.state_size}, expected {self.state_size}"
            )
        if model.dtype != self.dtype or model.device != self.device:
            raise ValueError(
                "model and HEOMPINNLoss must use the same dtype and device"
            )

    def rhs(self, real_state: torch.Tensor) -> torch.Tensor:
        """Apply the sparse real HEOM Liouvillian to batched row states."""
        if real_state.ndim != 2 or real_state.shape[-1] != 2 * self.state_size:
            raise ValueError(
                f"real_state must have shape (batch, {2 * self.state_size})"
            )
        return torch.sparse.mm(
            self.real_liouvillian,
            real_state.transpose(0, 1),
        ).transpose(0, 1)

    def dynamics_loss(
        self,
        state: torch.Tensor,
        time_derivative: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate Eq. (10), including its ``N * q_X`` denominator."""
        if state.shape != time_derivative.shape:
            raise ValueError("state and time_derivative must have identical shapes")
        residual = time_derivative - self.rhs(state)
        batch_size = state.shape[0]
        return residual.square().sum() / (self.state_size * batch_size)

    def initial_condition_loss(
        self,
        model: HEOMMLP,
        *,
        q_x: int,
    ) -> torch.Tensor:
        """Evaluate Eq. (12), retaining its documented ``1 / q_X`` factor."""
        zero_time = self.initial_state.new_zeros(1)
        difference = model(zero_time)[0] - self.initial_state
        return difference.square().sum() / q_x

    def trace_loss(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate the real root-ADO trace penalty in Eq. (13)."""
        real_state = state[:, : self.state_size]
        root_trace = real_state.index_select(
            1, self.root_diagonal_indices
        ).sum(dim=1)
        return (root_trace - 1.0).square().sum()

    def forward(
        self,
        model: HEOMMLP,
        times,
        *,
        q_x: int | None = None,
    ) -> LossTerms:
        """Evaluate the complete differentiable Section-III objective.

        ``q_x`` is the total collocation-set size in Eqs. (10)--(13).  When a
        minibatch is supplied, the trace sum is scaled by ``q_x / batch`` to
        give an unbiased estimate of the full sum.  Omitting it treats the
        provided batch as the complete collection.
        """
        self._validate_model(model)
        times = model.prepare_times(times)
        if times.numel() == 0:
            raise ValueError("times must contain at least one collocation point")
        batch_size = times.numel()
        if q_x is None:
            q_x = batch_size
        if (
            not isinstance(q_x, int)
            or isinstance(q_x, bool)
            or q_x < batch_size
        ):
            raise ValueError("q_x must be an integer at least as large as the batch")
        state, time_derivative = state_and_time_derivative(
            model,
            times,
            create_graph=True,
        )
        dynamics = self.dynamics_loss(state, time_derivative)
        initial_condition = self.initial_condition_loss(
            model,
            q_x=q_x,
        )
        trace = (q_x / batch_size) * self.trace_loss(state)
        total = (
            self.weights.dynamics * dynamics
            + self.weights.initial_condition * initial_condition
            + self.weights.trace * trace
        )
        return LossTerms(total, dynamics, initial_condition, trace)


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer and collocation configuration for :func:`train_mlp`."""

    t_start: float
    t_stop: float
    epochs: int = 1_000
    collocation_points: int = 512
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None
    resample_each_epoch: bool = True
    seed: int = 0
    log_every: int = 100

    def __post_init__(self) -> None:
        if not np.isfinite(self.t_start) or not np.isfinite(self.t_stop):
            raise ValueError("t_start and t_stop must be finite")
        if self.t_stop <= self.t_start:
            raise ValueError("t_stop must be greater than t_start")
        for name, value in (
            ("epochs", self.epochs),
            ("collocation_points", self.collocation_points),
            ("batch_size", self.batch_size),
            ("log_every", self.log_every),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if self.gradient_clip_norm is not None and (
            not np.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive")
        if not isinstance(self.resample_each_epoch, bool):
            raise TypeError("resample_each_epoch must be a boolean")


@dataclass(frozen=True)
class EpochRecord:
    """Collocation-weighted loss estimates after one training epoch."""

    epoch: int
    total: float
    dynamics: float
    initial_condition: float
    trace: float


@dataclass(frozen=True)
class TrainingResult:
    """History and wall-clock duration returned by :func:`train_mlp`."""

    history: tuple[EpochRecord, ...]
    elapsed_seconds: float

    @property
    def final(self) -> EpochRecord:
        return self.history[-1]


def _collocation_times(
    config: TrainingConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if not config.resample_each_epoch:
        return torch.linspace(
            config.t_start,
            config.t_stop,
            config.collocation_points,
            dtype=dtype,
            device=device,
        )

    # One randomly jittered point per interval gives uniform coverage while
    # changing the physics collocation set from epoch to epoch.
    offsets = torch.rand(
        config.collocation_points,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    strata = torch.arange(
        config.collocation_points,
        dtype=dtype,
        device=device,
    )
    unit_times = (strata + offsets) / config.collocation_points
    return config.t_start + (config.t_stop - config.t_start) * unit_times


def train_mlp(
    model: HEOMMLP,
    objective: HEOMPINNLoss,
    config: TrainingConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    callback: Callable[[EpochRecord], None] | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train an HEOM MLP on vectorized time-collocation minibatches."""
    if not isinstance(model, HEOMMLP):
        raise TypeError("model must be an instance of HEOMMLP")
    if not isinstance(objective, HEOMPINNLoss):
        raise TypeError("objective must be an instance of HEOMPINNLoss")
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig instance")
    objective._validate_model(model)
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    generator = torch.Generator(device=model.device)
    generator.manual_seed(config.seed)
    fixed_times = None
    if not config.resample_each_epoch:
        fixed_times = _collocation_times(
            config,
            dtype=model.dtype,
            device=model.device,
            generator=generator,
        )

    model.train()
    history: list[EpochRecord] = []
    start_time = perf_counter()
    for epoch in range(1, config.epochs + 1):
        if fixed_times is None:
            collocation_times = _collocation_times(
                config,
                dtype=model.dtype,
                device=model.device,
                generator=generator,
            )
        else:
            collocation_times = fixed_times
        order = torch.randperm(
            config.collocation_points,
            device=model.device,
            generator=generator,
        )

        totals = np.zeros(4, dtype=np.float64)
        points_seen = 0
        for first in range(0, config.collocation_points, config.batch_size):
            batch_indices = order[first : first + config.batch_size]
            batch_times = collocation_times.index_select(0, batch_indices)
            current_batch_size = batch_times.numel()
            optimizer.zero_grad(set_to_none=True)
            terms = objective(
                model,
                batch_times,
                q_x=config.collocation_points,
            )
            if not torch.isfinite(terms.total):
                raise FloatingPointError(
                    f"Non-finite training loss encountered at epoch {epoch}"
                )
            terms.total.backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                )
            optimizer.step()
            totals += current_batch_size * np.asarray(
                [
                    terms.total.detach().item(),
                    terms.dynamics.detach().item(),
                    terms.initial_condition.detach().item(),
                    terms.trace.detach().item(),
                ]
            )
            points_seen += current_batch_size

        averages = totals / points_seen
        record = EpochRecord(epoch, *averages.tolist())
        history.append(record)
        if callback is not None:
            callback(record)
        if verbose and (
            epoch == 1
            or epoch % config.log_every == 0
            or epoch == config.epochs
        ):
            print(
                f"Epoch {epoch:6d}/{config.epochs}: "
                f"loss={record.total:.6e}, dynamics={record.dynamics:.6e}, "
                f"IC={record.initial_condition:.6e}, trace={record.trace:.6e}"
            )

    return TrainingResult(
        history=tuple(history),
        elapsed_seconds=perf_counter() - start_time,
    )


@dataclass(frozen=True)
class MLPSolution:
    """MLP trajectory in the same complex stacked-state convention as HEOM."""

    t: np.ndarray
    y: np.ndarray
    system_dimension: int

    @property
    def primary_ados(self) -> np.ndarray:
        """Return physical root density matrices at every output time."""
        system_size = self.system_dimension**2
        root_vectors = self.y[:system_size, :].T
        return root_vectors.reshape(
            -1,
            self.system_dimension,
            self.system_dimension,
        ).transpose(0, 2, 1)

    def expectation(self, operator) -> np.ndarray:
        """Evaluate ``Tr(operator @ rho_root(t))`` along the trajectory."""
        operator = np.asarray(operator, dtype=np.complex128)
        expected_shape = (self.system_dimension, self.system_dimension)
        if operator.shape != expected_shape:
            raise ValueError(
                f"operator has shape {operator.shape}, expected {expected_shape}"
            )
        values = np.einsum("tij,ji->t", self.primary_ados, operator)
        return np.real_if_close(values)


def solve_mlp(
    model: HEOMMLP,
    t_eval,
    *,
    batch_size: int = 1_024,
) -> MLPSolution:
    """Evaluate a trained model and return a complex HEOM trajectory."""
    if not isinstance(model, HEOMMLP):
        raise TypeError("model must be an instance of HEOMMLP")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    t_eval = np.asarray(t_eval, dtype=np.float64)
    if t_eval.ndim != 1 or t_eval.size == 0:
        raise ValueError("t_eval must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(t_eval)) or np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval must contain finite, strictly increasing times")

    was_training = model.training
    model.eval()
    predictions = []
    with torch.no_grad():
        for first in range(0, t_eval.size, batch_size):
            times = torch.as_tensor(
                t_eval[first : first + batch_size],
                dtype=model.dtype,
                device=model.device,
            )
            predictions.append(model(times).cpu())
    if was_training:
        model.train()

    real_prediction = torch.cat(predictions, dim=0).numpy()
    complex_prediction = (
        real_prediction[:, : model.state_size]
        + 1j * real_prediction[:, model.state_size :]
    )
    return MLPSolution(
        t=t_eval.copy(),
        y=complex_prediction.T,
        system_dimension=model.system_dimension,
    )


__all__ = [
    "EpochRecord",
    "HEOMPINNLoss",
    "LossTerms",
    "LossWeights",
    "MLPSolution",
    "TrainingConfig",
    "TrainingResult",
    "scipy_sparse_to_torch",
    "solve_mlp",
    "train_mlp",
]
