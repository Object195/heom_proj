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
    heom_depth: int = 20
    qutip_depths: tuple[int, ...] = (20,)
    t_start: float = 0
    t_stop: float = 20
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
    # When both values are set, beta=lower_tier_loss_weight is assigned to
    # the mean loss over tiers 0..lower_tier_loss_cutoff and 1-beta to the
    # mean over the remaining tiers.
    lower_tier_loss_cutoff: int | None = None
    lower_tier_loss_weight: float | None = None
    # Alternative to the two-group weighting: tier l receives normalized
    # weight proportional to (l + 1)**(-tier_loss_power).
    tier_loss_power: float | None = None
    normalize_hierarchy_coordinates: bool = True
    # Replace only the root RDM by a normalized A A^dagger factorization.
    positive_rdm_ansatz: bool = False
    # Enabled for current baseline runs. Metadata v1-v3 migrates these flags
    # to False so existing checkpoints retain their original forward map.
    constant_loss_normalization: bool = True
    ansatz_scale_normalization: bool = True
    normalization_floor: float = 1e-12
    #time_switch: str = "exponential"
    time_switch: str = "linear"
    switch_time_constant: float = 1.0
    epochs: int = 200
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
    log_every: int = 20
    inference_batch_size: int = 1024


PSEUDOMODE = PseudomodeParameters()
MLP = MLPParameters()
MLP_MODEL_PATH = PROJECT_ROOT / "saved_models" / "mlp" / "mlp_state_dict.pt"


__all__ = ["MLP", "MLP_MODEL_PATH", "PSEUDOMODE", "PROJECT_ROOT"]
