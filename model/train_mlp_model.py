"""Train the Section-III MLP and save its state dictionary."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from experiment_parameters import MLP, MLP_MODEL_PATH, PSEUDOMODE
from heom.heom_rep import heom_state
from model import (
    HEOMMLP,
    HEOMPINNLoss,
    LossWeights,
    TrainingConfig,
    train_mlp,
)


def build_training_problem():
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.diag([1.0, -1.0])
    h_system = 0.5 * PSEUDOMODE.delta * sigma_z
    h_system += 0.5 * PSEUDOMODE.v * sigma_x
    rho0 = np.diag([1.0, 0.0])
    frequencies = np.array(
        [0.5 * PSEUDOMODE.gamma + 1j * PSEUDOMODE.w0],
        dtype=np.complex128,
    )
    coefficients = np.array([PSEUDOMODE.g**2], dtype=np.complex128)
    hierarchy = heom_state(
        K=0,
        L=PSEUDOMODE.heom_depth,
        H_s=h_system,
        H_c=sigma_z,
        C_list=coefficients,
        gamma_list=frequencies,
    )
    liouvillian = hierarchy.build_Liouvillian(
        markovian_terminator=False,
        normalized=True,
    )
    return hierarchy, rho0, liouvillian


def main():
    dtype = getattr(torch, MLP.dtype)
    device = torch.device(MLP.device)
    torch.manual_seed(MLP.seed)
    hierarchy, rho0, liouvillian = build_training_problem()

    model = HEOMMLP(
        hierarchy,
        hidden_sizes=MLP.hidden_sizes,
        t_start=PSEUDOMODE.t_start,
        t_stop=PSEUDOMODE.t_stop,
        activation=MLP.activation,
        dtype=dtype,
        device=device,
    )
    objective = HEOMPINNLoss(
        hierarchy,
        rho0,
        liouvillian=liouvillian,
        weights=LossWeights(
            dynamics=MLP.dynamics_weight,
            initial_condition=MLP.initial_condition_weight,
            trace=MLP.trace_weight,
        ),
        dtype=dtype,
        device=device,
    )
    config = TrainingConfig(
        t_start=PSEUDOMODE.t_start,
        t_stop=PSEUDOMODE.t_stop,
        epochs=MLP.epochs,
        collocation_points=MLP.collocation_points,
        batch_size=MLP.batch_size,
        learning_rate=MLP.learning_rate,
        weight_decay=MLP.weight_decay,
        gradient_clip_norm=MLP.gradient_clip_norm,
        resample_each_epoch=MLP.resample_each_epoch,
        seed=MLP.seed,
        log_every=MLP.log_every,
    )
    result = train_mlp(model, objective, config)

    MLP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MLP_MODEL_PATH)
    print(f"Training time: {result.elapsed_seconds:.3f} s")
    print(f"Final loss: {result.final.total:.6e}")
    print(f"Saved MLP model: {MLP_MODEL_PATH}")
    return model, result


if __name__ == "__main__":
    main()
