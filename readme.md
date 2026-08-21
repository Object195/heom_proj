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

Physical pseudomode parameters and all MLP/training hyperparameters in
`experiment_parameters.py` are the immutable defaults. A sparse TOML file can
override only the values needed for one run without editing that module.

The MLP predicts a partner-symmetric HEOM correction rather than the complete
state. Its root ADO is projected to be traceless, and the physical state is
constructed as `initial_state + s * correction`, with normalized-time switch
`s = (tau + 1) / 2`. The initial condition and unit trace are therefore exact,
so training uses only the HEOM dynamical residual.

Train and save the configured network to
`saved_models/mlp/mlp_state_dict.pt` with:

```powershell
python -m model.train_mlp_model
```

Continue training from that saved model with:

```powershell
python -m model.train_mlp_model --resume
```

### Sparse overrides and training sequences

Use `training_sequence.example.toml` as a starting point for a staged run:

```powershell
python -m model.train_mlp_model --sequence training_sequence.toml
```

The file has three layers. Omitted values always come from
`experiment_parameters.py`:

```toml
[pseudomode]
g = 0.2

[mlp]
device = "cuda"
dtype = "float64"
collocation_points = 512

[[sessions]]
name = "adam_warmup"
[sessions.mlp]
optimizer = "adam"
epochs = 100
learning_rate = 1e-3

[[sessions]]
name = "lbfgs_refinement"
[sessions.mlp]
optimizer = "lbfgs"
epochs = 25
collocation_points = 1024
```

Top-level `[pseudomode]` and `[mlp]` values form the shared experiment. Each
`[sessions.mlp]` table is a sparse override of that same shared MLP config, so
a setting from one session does not accidentally leak into the next. Model
weights do carry forward: the hierarchy, objective, and network are built
once, each session gets a fresh optimizer, and the checkpoint is updated after
every successful session. Checkpoint replacement is atomic, so an interrupted
save does not destroy the previous stage. Put architecture, dtype/device, time
interval, and physical changes at the top level; changing them inside a
session is rejected because it would make in-memory weight reuse ambiguous.
Because every stage shares one network dtype, a sequence containing L-BFGS
must use top-level `dtype = "float64"` (the default).

A TOML file with only top-level overrides runs one session. `--config` is an
alias for `--sequence`, and `--resume` loads the checkpoint once before the
first session. The legacy `--optimizer` option remains available for a
single run, but cannot be combined with a sequence whose sessions select their
own optimizers. Unknown or misspelled fields are reported before training.
Use `--model-path runs/experiment_a.pt` to keep experiments in separate
checkpoints; `--resume` checks saved physics and model-construction metadata
before loading.

Each checkpoint has a versioned `<checkpoint>.config.json` sidecar containing
the fully resolved parameters. This makes runs reproducible and lets the
benchmark reconstruct the exact physics and network even after the defaults
change.

Add `--plot-loss` to either command to display an interactive logarithmic
loss curve during training. Resuming restores the network parameters and
starts a new optimizer run with the configured learning rate.

Use L-BFGS with a fixed full collocation batch, float64 arithmetic, and a
strong-Wolfe line search with:

```powershell
python -m model.train_mlp_model --optimizer lbfgs --plot-loss
```

L-BFGS is the configured default. Use `--optimizer adam` for a single run, or
set `optimizer = "adam"` in a sequence session. The L-BFGS learning rate,
iteration/evaluation limits, history size, and line search can likewise be
overridden sparsely. For L-BFGS, one reported epoch is one optimizer step
(with any extra closure evaluations
required by the line search). The stopping thresholds are exposed as
`lbfgs_tolerance_grad` and `lbfgs_tolerance_change`.

Each L-BFGS training log also reports the post-step full-batch maximum
gradient norm `g_inf` and the maximum parameter change for that optimizer
step, `delta_theta_inf`. Computing the post-step gradient adds one full-batch
loss and backward evaluation only on logged epochs.

Run the three-trajectory comparison (explicit Lindbladian, sparse HEOM, and
the saved MLP) with:

```powershell
python -m benchmark.benchmark_mlp
```

The benchmark performs no training. It reconstructs the configured MLP,
loads the saved state dictionary, and evaluates the trajectory. It reads the
checkpoint sidecar automatically. For a legacy checkpoint without a sidecar,
it falls back to `experiment_parameters.py`; the source TOML can be supplied
explicitly with `--sequence`, and a non-default checkpoint with
`--model-path`:

```powershell
python -m benchmark.benchmark_mlp --model-path runs/experiment_a.pt
python -m benchmark.benchmark_mlp --sequence training_sequence.example.toml
```

To evaluate forward time-domain generalization without changing the model's
stored training-time normalization, provide a custom benchmark interval and
sampling count. For example, a model trained on `[0, 10]` can be evaluated on
`[0, 20]` with:

```powershell
python -m benchmark.benchmark_mlp --t-stop 20 --n-times 2000
```

The benchmark may also start later, such as `--t-start 10 --t-stop 20`. The
numerical references still propagate from the original initial time before
returning the requested interval, so the physical state is not reset at
`t=10`.

The neural pipeline requires PyTorch. The benchmarks additionally use QuTiP
and Matplotlib.
