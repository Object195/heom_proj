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
    t_start: float = 0.0
    t_stop: float = 0.1
    n_times: int = 1_000
    rtol: float = 1e-8
    atol: float = 1e-10


@dataclass(frozen=True)
class MLPParameters:
    hidden_sizes: tuple[int, ...] = (64, 64, 64, 64)
    activation: str = "tanh"
    dtype: str = "float64"
    device: str = "cuda"
    epochs: int = 2000
    collocation_points: int = 512
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None
    resample_each_epoch: bool = True
    seed: int = 0
    log_every: int = 100
    inference_batch_size: int = 1024


PSEUDOMODE = PseudomodeParameters()
MLP = MLPParameters()
MLP_MODEL_PATH = PROJECT_ROOT / "saved_models" / "mlp" / "mlp_state_dict.pt"


__all__ = ["MLP", "MLP_MODEL_PATH", "PSEUDOMODE", "PROJECT_ROOT"]
