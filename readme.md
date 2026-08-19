Repository for HEOM construction, sparse propagation, and neural baselines.

## Physics-informed MLP baseline

The `model` package implements the coordinate MLP from Section III of
`HEOM_DL.pdf`.  It uses the hierarchy's existing BFS ADO order, column-major
matrix vectorization, adjoint-partner symmetrization, and a vectorized JVP for
time derivatives.  `HEOMPINNLoss` builds the normalized hard-cutoff operator
with:

```python
liouvillian = hierarchy.build_Liouvillian(
    markovian_terminator=False,
    normalized=True,
)
```

The same sparse operator can be passed to both `solve_heom` and
`HEOMPINNLoss`, ensuring numerical and MLP trajectories use identical state
ordering and truncation.

```python
from model import (
    HEOMMLP,
    HEOMPINNLoss,
    LossWeights,
    TrainingConfig,
    train_mlp,
)

network = HEOMMLP(hierarchy, hidden_sizes=(64, 64, 64))
training = TrainingConfig(
    t_start=0.0,
    t_stop=100.0,
    collocation_points=512,
)
objective = HEOMPINNLoss(
    hierarchy,
    rho0,
    liouvillian=liouvillian,
    weights=LossWeights.balanced(training.collocation_points),
)
result = train_mlp(network, objective, training)
```

Run the three-trajectory comparison (explicit Lindbladian, sparse HEOM, and
MLP) with:

```powershell
python -m benchmark.benchmark_mlp
```

For a quick smoke run:

```powershell
python -m benchmark.benchmark_mlp --depth 3 --t-stop 5 `
    --n-times 101 --epochs 20 --collocation-points 32 --no-show
```

The neural pipeline requires PyTorch.  The benchmark additionally requires
QuTiP and Matplotlib.
