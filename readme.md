Repository for HEOM construction, sparse propagation, and neural baselines.

## Installation

This version targets Python 3.11+, NumPy 2, and QuTiP 5.3.  Install the
PyTorch build that matches the local CUDA toolkit first, then install the
project dependencies without downgrading that build:

```powershell
python -m pip install -r requirements.txt
```

An environment that previously contained QuTiP 4 must be upgraded because
its compiled extension is not compatible with NumPy 2:

```powershell
python -m pip install --upgrade "qutip>=5.3.1,<6"
```

The benchmarks use the QuTiP 5 solver API (`qutip.solver.heom`) and pass
solver options and expectation operators with the keyword-based interface.

## Physics-informed MLP baseline

The `model` package implements the coordinate MLP from Section III of
`HEOM_DL.pdf`.  It uses the hierarchy's existing BFS ADO order, column-major
matrix vectorization, adjoint-partner symmetrization, and a vectorized JVP for
time derivatives. Physical times are passed through the public API and mapped
internally from the configured training interval to `[-1, 1]`; the JVP still
returns derivatives with respect to physical time. `HEOMPINNLoss` builds the
normalized hard-cutoff operator with:

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

The neural pipeline requires PyTorch. The benchmarks additionally use QuTiP
and Matplotlib.
