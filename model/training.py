"""Physics-informed loss, training loop, and inference for ``HEOMMLP``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from heom import q_func
from heom.heom_rep import heom_state

from .mlp import HEOMMLP, state_and_time_derivative


def scipy_sparse_to_torch(
    matrix,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a PyTorch COO tensor."""
    matrix = sp.coo_array(matrix)
    matrix.eliminate_zeros()
    indices = torch.as_tensor(
        np.vstack((matrix.row, matrix.col)),
        dtype=torch.long,
        device=device,
    )
    values = torch.as_tensor(matrix.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=matrix.shape,
        dtype=dtype,
        device=device,
        check_invariants=False,
    ).coalesce()


class HEOMPINNLoss(nn.Module):
    """HEOM dynamical residual for a hard-constrained ``HEOMMLP``."""

    def __init__(
        self,
        hierarchy: heom_state,
        *,
        liouvillian=None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if liouvillian is None:
            liouvillian = hierarchy.build_Liouvillian(
                markovian_terminator=False,
                normalized=True,
            )

        self.state_size = hierarchy.nADO * hierarchy.system_size

        real_liouvillian = q_func.sup_op_to_real(liouvillian).tocsr()
        self.register_buffer(
            "real_liouvillian",
            scipy_sparse_to_torch(
                real_liouvillian,
                dtype=dtype,
                device=device,
            ),
        )

    def rhs(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(
            self.real_liouvillian,
            state.transpose(0, 1),
        ).transpose(0, 1)

    def dynamics_loss(
        self,
        state: torch.Tensor,
        time_derivative: torch.Tensor,
    ) -> torch.Tensor:
        residual = time_derivative - self.rhs(state)
        return residual.square().sum() / (
            self.state_size * state.shape[0]
        )

    def forward(
        self,
        model: HEOMMLP,
        times,
    ) -> torch.Tensor:
        times = model.prepare_times(times)
        state, time_derivative = state_and_time_derivative(model, times)
        return self.dynamics_loss(state, time_derivative)


@dataclass(frozen=True)
class TrainingConfig:
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


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    loss: float
    gradient_inf: float | None = None
    parameter_change_inf: float | None = None
    lbfgs_iterations: int | None = None
    lbfgs_evaluations: int | None = None
    lbfgs_curvature_pairs: int | None = None


@dataclass(frozen=True)
class TrainingResult:
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
    """Train with Adam minibatches or a fixed full-batch L-BFGS closure."""
    optimizer = optimizer or torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=model.device)
    generator.manual_seed(config.seed)
    fixed_times = None
    using_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
    lbfgs_loss_scale = None
    if using_lbfgs:
        fixed_times = torch.linspace(
            config.t_start,
            config.t_stop,
            config.collocation_points,
            dtype=model.dtype,
            device=model.device,
        )
        # HEOMPINNLoss reports a mean residual so results remain comparable
        # across hierarchy and collocation-grid sizes.  PyTorch L-BFGS uses an
        # absolute y^T s > 1e-10 safeguard for accepting curvature pairs; the
        # tiny mean-loss scale can therefore leave its history permanently
        # empty.  Optimize the equivalent residual sum while retaining the
        # normalized mean for reporting and convergence diagnostics.
        lbfgs_loss_scale = objective.state_size * fixed_times.numel()
    elif not config.resample_each_epoch:
        fixed_times = _collocation_times(
            config,
            dtype=model.dtype,
            device=model.device,
            generator=generator,
        )

    model.train()
    history = []
    start_time = perf_counter()
    for epoch in range(1, config.epochs + 1):
        log_epoch = (
            epoch == 1
            or epoch % config.log_every == 0
            or epoch == config.epochs
        )
        report_lbfgs_diagnostics = using_lbfgs and verbose and log_epoch
        parameters_before = (
            tuple(
                parameter.detach().clone()
                for parameter in model.parameters()
            )
            if report_lbfgs_diagnostics
            else None
        )
        collocation_times = fixed_times
        if collocation_times is None:
            collocation_times = _collocation_times(
                config,
                dtype=model.dtype,
                device=model.device,
                generator=generator,
            )
        total = 0.0
        if using_lbfgs:
            first_parameter = optimizer.param_groups[0]["params"][0]
            optimizer_state = optimizer.state[first_parameter]
            evaluations_before = int(optimizer_state.get("func_evals", 0))
            iterations_before = int(optimizer_state.get("n_iter", 0))

            def closure():
                optimizer.zero_grad(set_to_none=True)
                mean_loss = objective(model, collocation_times)
                optimizer_loss = lbfgs_loss_scale * mean_loss
                optimizer_loss.backward()
                return optimizer_loss

            optimizer.step(closure)
            optimizer_state = optimizer.state[first_parameter]
            lbfgs_evaluations = (
                int(optimizer_state.get("func_evals", evaluations_before))
                - evaluations_before
            )
            lbfgs_iterations = (
                int(optimizer_state.get("n_iter", iterations_before))
                - iterations_before
            )
            lbfgs_curvature_pairs = len(
                optimizer_state.get("old_dirs", ())
            )

            # Evaluate the normalized objective at the final parameters.
            # Gradients from the scaled closure are deliberately discarded so
            # g_inf remains comparable with Adam runs and earlier checkpoints.
            optimizer.zero_grad(set_to_none=True)
            latest_loss = objective(model, collocation_times)
            gradient_inf = None
            parameter_change_inf = None
            if report_lbfgs_diagnostics:
                latest_loss.backward()
                gradient_inf = torch.stack(
                    [
                        parameter.grad.detach().abs().amax()
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                ).amax().item()
                parameter_change_inf = torch.stack(
                    [
                        (parameter.detach() - previous).abs().amax()
                        for parameter, previous in zip(
                            model.parameters(), parameters_before
                        )
                    ]
                ).amax().item()
            total = config.collocation_points * latest_loss.detach().item()
        else:
            gradient_inf = None
            parameter_change_inf = None
            lbfgs_iterations = None
            lbfgs_evaluations = None
            lbfgs_curvature_pairs = None
            order = torch.randperm(
                config.collocation_points,
                device=model.device,
                generator=generator,
            )
            for first in range(
                0, config.collocation_points, config.batch_size
            ):
                indices = order[first : first + config.batch_size]
                batch_times = collocation_times[indices]
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model, batch_times)
                loss.backward()
                if config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.gradient_clip_norm,
                    )
                optimizer.step()
                total += batch_times.numel() * loss.detach().item()

        record = EpochRecord(
            epoch,
            total / config.collocation_points,
            gradient_inf,
            parameter_change_inf,
            lbfgs_iterations,
            lbfgs_evaluations,
            lbfgs_curvature_pairs,
        )
        history.append(record)
        if callback is not None:
            callback(record)
        if verbose and log_epoch:
            message = (
                f"Epoch {epoch:6d}/{config.epochs}: "
                f"loss={record.loss:.6e}"
            )
            if using_lbfgs:
                message += (
                    f"  g_inf={record.gradient_inf:.6e}"
                    "  "
                    f"delta_theta_inf={record.parameter_change_inf:.6e}"
                    "  "
                    f"lbfgs_iter={record.lbfgs_iterations}"
                    "  "
                    f"evals={record.lbfgs_evaluations}"
                    "  "
                    f"history={record.lbfgs_curvature_pairs}"
                )
            print(message)

    return TrainingResult(
        history=tuple(history),
        elapsed_seconds=perf_counter() - start_time,
    )


@dataclass(frozen=True)
class MLPSolution:
    t: np.ndarray
    y: np.ndarray
    system_dimension: int

    @property
    def primary_ados(self) -> np.ndarray:
        dimension = self.system_dimension
        root = self.y[: dimension**2].T
        return root.reshape(-1, dimension, dimension).transpose(0, 2, 1)

    def expectation(self, operator) -> np.ndarray:
        values = np.einsum("tij,ji->t", self.primary_ados, operator)
        return np.real_if_close(values)


def solve_mlp(
    model: HEOMMLP,
    t_eval,
    *,
    batch_size: int = 1_024,
) -> MLPSolution:
    """Evaluate on a physical-time grid; the model normalizes internally."""
    t_eval = np.asarray(t_eval, dtype=np.float64)
    was_training = model.training
    model.eval()
    predictions = []
    with torch.no_grad():
        for first in range(0, t_eval.size, batch_size):
            predictions.append(model(t_eval[first : first + batch_size]).cpu())
    model.train(was_training)

    prediction = torch.cat(predictions).numpy()
    state = (
        prediction[:, : model.state_size]
        + 1j * prediction[:, model.state_size :]
    )
    return MLPSolution(t_eval, state.T, model.system_dimension)


__all__ = [
    "EpochRecord",
    "HEOMPINNLoss",
    "MLPSolution",
    "TrainingConfig",
    "TrainingResult",
    "solve_mlp",
    "train_mlp",
]
