"""Parameters shared by MLP training and trajectory benchmarks."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PseudomodeParameters:
    w0: float = 1.0
    delta: float = 1.0
    v: float = 0.1
    g: float = 0.1
    gamma: float = 1
    cavity_dimension: int = 20
    heom_depth: int = 5
    qutip_depths: tuple[int, ...] = (5,)
    t_start: float = 0
    t_stop: float = 10
    n_times: int = 1_000
    rtol: float = 1e-8
    atol: float = 1e-10


@dataclass(frozen=True)
class MLPParameters:
    hidden_sizes: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"
    dtype: str = "float64"
    device: str = "cuda"
    tier_normalized_loss: bool = True
    #time_switch: str = "exponential"
    time_switch: str = "linear"
    switch_time_constant: float = 1.0
    epochs: int = 300
    collocation_points: int = 1024
    batch_size: int = 64
    #optimizer: str = "adam"
    optimizer: str = "lbfgs"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lbfgs_learning_rate: float = 1.0
    lbfgs_max_iter: int = 50
    lbfgs_max_eval: int = 100
    lbfgs_history_size: int = 100
    lbfgs_tolerance_grad: float = 1e-12
    lbfgs_tolerance_change: float = 1e-14
    lbfgs_line_search: str = "strong_wolfe"
    gradient_clip_norm: float | None = None
    resample_each_epoch: bool = True
    seed: int = 0
    log_every: int = 100
    inference_batch_size: int = 1024


PSEUDOMODE = PseudomodeParameters()
MLP = MLPParameters()
MLP_MODEL_PATH = PROJECT_ROOT / "saved_models" / "mlp" / "mlp_state_dict.pt"


__all__ = ["MLP", "MLP_MODEL_PATH", "PSEUDOMODE", "PROJECT_ROOT"]
