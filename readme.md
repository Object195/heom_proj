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

Physical pseudomode parameters and all MLP/training hyperparameters live in
`experiment_parameters.py`.

Train and save the configured network to
`saved_models/mlp/mlp_state_dict.pt` with:

```powershell
python -m model.train_mlp_model
```

Run the three-trajectory comparison (explicit Lindbladian, sparse HEOM, and
the saved MLP) with:

```powershell
python -m benchmark.benchmark_mlp
```

The benchmark performs no training. It reconstructs the configured MLP,
loads the saved state dictionary, and evaluates the trajectory.

The neural pipeline requires PyTorch.  The benchmark additionally requires
QuTiP and Matplotlib.
