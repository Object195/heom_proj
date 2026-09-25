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

By default the MLP predicts a partner-symmetric HEOM correction rather than the complete
state. Its root ADO is projected to be traceless, and the physical state is
constructed as `initial_state + s * a_s * correction`, with normalized-time
switch `s = (tau + 1) / 2`. The optional scale `a_s` is one when ansatz
normalization is disabled. The initial condition and unit trace are therefore
exact, so training uses only the HEOM dynamical residual.

For an optional positive-semidefinite tier-0 reduced density matrix, set:

```toml
[mlp]
positive_rdm_ansatz = true # default: false (the original additive ansatz)
```

At the root only, the unconstrained complex matrix `B(t)` from the raw MLP
output is interpreted as a factor correction:

```text
A(t) = sqrt(rho_init) + s(t) * a_s * B(t)
rho_root(t) = A(t) A(t)^dagger / Tr[A(t) A(t)^dagger]
```

The square root is the principal positive-semidefinite matrix square root.
The root output bypasses the old symmetrization and trace subtraction; all
higher ADOs retain their existing additive, partner-symmetric construction.
The same configured linear/exponential switch `s(t)` vanishes at `t_start`,
so the initial state is preserved. The root is Hermitian, PSD, and unit trace
up to floating-point roundoff, including during time extrapolation.

The existing `a_s = RMS(L chi_init) / s'(t_start)` is used in the factor
correction (or 1 if ansatz scaling is disabled). It stays linear in `a_s`
near the anchor through the cross terms with `sqrt(rho_init)`; using
`sqrt(a_s)` would give a different local scale. This is a conditioning
choice, not an exact equality of sensitivities between the two ansatzes.
Neither the HEOM residual, its tier weights, nor its constant-trajectory
normalizer `L_const` needs to change. Time derivatives differentiate the
complete normalized Gram matrix, including its denominator.

The initial root must itself be Hermitian, PSD, and unit trace. The model
repairs roundoff-sized violations only, and rejects materially invalid
anchors, including those caused by an insufficient HEOM cutoff. At the
exceptional point `A=0`, where the quotient is undefined, the implementation
returns `rho_init`; there is no continuous extension of this quotient at
that point. Nonzero factors are rescaled before forming the Gram matrix to
avoid overflow/underflow without adding an epsilon that changes the trace.
For a rank-deficient anchor, population growth in its null space begins
quadratically in elapsed time for a smooth `B`. This restriction fits the
initial root dynamics of the current factorized spin-boson setup, but may
limit an exact fit for other initial states or dissipative generators.

This setting is fixed across training sessions and saved in metadata v8.
Older metadata defaults to `false`, independent of current defaults. Resume,
network transfer, and benchmark loading reject incompatible ansatz settings
because the root network output has a different meaning. Square-root buffers
are rebuilt from the saved initial state when loading a checkpoint.

Train and save the configured network in a new automatically named folder
below `saved_models` with:

```powershell
python -m model.train_mlp_model
```

Continue training by specifying the saved checkpoint:

```powershell
python -m model.train_mlp_model --resume --model-path saved_models/<run-folder>/mlp_state_dict.pt
```

To initialize a new run from only the reusable network weights of an existing
checkpoint, set the source folder at the top level of the sequence file:

```toml
name = "depth_10"
initialize_from = "depth_5"
```

Then the usual sequence command is sufficient:

```powershell
python -m model.train_mlp_model --sequence training_sequence.toml
```

The command-line form remains available as an override:

```powershell
python -m model.train_mlp_model --sequence training_sequence.toml --initialize-from depth_5
```

This reads `saved_models/depth_5/mlp_state_dict.pt`, while the current run
rebuilds its own hierarchy, anchored initial state, loss, and optimizer. The
folder argument is optional; both `initialize_from = "mlp"` and bare
`--initialize-from` use `saved_models/mlp/mlp_state_dict.pt`. Omit the TOML
setting and the flag for a fresh random initialization. This differs from
`--resume`, which restores the complete same-hierarchy model and cannot be
combined with transfer initialization.

### Sparse overrides and training sequences

Use `training_sequence.toml` as a starting point for a staged run:

```powershell
python -m model.train_mlp_model --sequence training_sequence.toml
```

The file has three layers. Omitted values always come from
`experiment_parameters.py`:

```toml
name = "experiment_a"
initialize_from = "depth_5" # optional transfer-learning source folder

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

Set `tier_normalized_loss = true` in the top-level `[mlp]` table to give each
hierarchy tier equal weight. The loss averages each tier over its own ADO count
and then averages the tier losses with the global factor `1 / (L + 1)`. Set it
to `false` for the global ADO-normalized loss. This is a base-only setting
because the objective is shared across all sessions.

For a simple two-group diagnostic, keep tier normalization enabled and set:

```toml
[mlp]
tier_normalized_loss = true
lower_tier_loss_cutoff = 5
lower_tier_loss_weight = 0.5
```

Writing `beta = lower_tier_loss_weight`, this constructs
`beta * mean(E_0, ..., E_5) + (1 - beta) * mean(E_6, ..., E_L)`.
Both options must be set together, the cutoff must leave at least one upper
tier, and `beta` must lie between zero and one. Leaving them unset preserves
the ordinary equal-tier loss.

As an alternative to the two-group split, set one power:

```toml
[mlp]
tier_normalized_loss = true
tier_loss_power = 1.0
```

This constructs
`sum((l + 1)^(-p) E_l) / sum((l + 1)^(-p))`. A positive `p` emphasizes
lower tiers, `p = 0` recovers equal-tier weighting, and a negative `p`
emphasizes upper tiers. `tier_loss_power` and the two-group settings are
mutually exclusive.

The staged baseline config also enables two conditioning normalizations:

```toml
[mlp]
constant_loss_normalization = true
ansatz_scale_normalization = true
normalization_floor = 1e-12
```

`constant_loss_normalization` divides the training residual by
`L_const`, the residual loss of the constant anchored trajectory
`chi(t) = chi_s`. It uses exactly the same global, equal-tier, two-group, or
power-law weighting as the selected loss. `ansatz_scale_normalization` makes
the network output dimensionless and sets
`a_s = RMS(L chi_s) / s'(t_start)`; for the linear
switch this is `T * RMS(L chi_s)`. The ansatz scale always uses the global
complex-component RMS, so changing tier loss weights does not change the
model parameterization. `normalization_floor` keeps both constructions finite
near a stationary state. These scalars condition optimization but do not add
extra RDM weighting.

Hierarchy-coordinate inputs are divided by the hierarchy depth `L` by
default. To pass raw integer indices to the shared MLP instead, set:

```toml
[mlp]
normalize_hierarchy_coordinates = false
```

This is a model-parameterization setting and is recorded in checkpoint
metadata. A checkpoint cannot be resumed under the opposite setting.

The model uses `time_switch = "linear"` by default, preserving the constrained
output `rho(t) = rho(t_start) + ((t - t_start) / T) correction(t)`. For a
bounded switch during time extrapolation, set these top-level `[mlp]` options:

```toml
time_switch = "exponential"
switch_time_constant = 1.0
```

Here `switch_time_constant` is in physical time units. Internally it is divided
by `T = t_stop - t_start` and the switch is evaluated as
`(1 - exp(-(t-t_start)/t_c)) / (1 - exp(-T/t_c))`. It is therefore zero at
`t_start`, one at `t_stop`, and bounded for later times. The physical-time JVP
used by the residual differentiates the complete model, so it automatically
includes the exponential-switch derivative.

To train from a later point on the physical trajectory, set the top-level
`[pseudomode]` interval accordingly:

```toml
[pseudomode]
t_start = 5.0
t_stop = 10.0
```

When `t_start > 0`, the factorized state at physical time zero is first evolved
to `t_start` with the same normalized, hard-cutoff sparse HEOM used by the
training residual. The entire evolved hierarchy—including all nonzero ADOs—is
then used as the MLP's constrained initial state. Time is not reset: the MLP
still receives physical `t`, while its existing normalization maps
`[t_start, t_stop]` to `[-1, 1]`. Consequently, the physical-time JVP and HEOM
residual require no special adjustment.

A TOML file with only top-level overrides runs one session. `--config` is an
alias for `--sequence`, and `--resume` loads the checkpoint once before the
first session. The legacy `--optimizer` option remains available for a
single run, but cannot be combined with a sequence whose sessions select their
own optimizers. Unknown or misspelled fields are reported before training.
An optional bare top-level `name = "experiment_a"` stores the checkpoint,
metadata, and log under `saved_models/experiment_a`. With no name, fresh
training creates `saved_models/MM-DD-HH-MM_g_<g>_L_<L>_t_<t_start>_<t_stop>`,
using local time and the resolved physical parameters. Same-minute name
collisions receive `_2`, `_3`, etc.; the chosen name is recorded in metadata.
An explicit `--model-path runs/experiment_a.pt` overrides either choice.
`--resume` checks saved physics and model-construction metadata before loading;
with neither a name nor an explicit path it retains the legacy `saved_models/mlp`
destination, rather than creating a new run.

Each checkpoint has a versioned `<checkpoint>.config.json` sidecar containing
the fully resolved parameters. This makes runs reproducible and lets the
benchmark reconstruct the exact physics and network even after the defaults
change. A short-lived `.incomplete` marker protects the checkpoint/sidecar
pair while both files are replaced; resume and benchmark commands reject a
pair whose update was interrupted instead of silently using stale metadata.
Resuming with ansatz scaling enabled also requires the sidecar, since legacy
unscaled weights cannot safely be reinterpreted under a newly introduced
`a_s`.

Add `--plot-loss` to either command to display an interactive logarithmic
loss curve during training. Resuming restores the network parameters and
starts a new optimizer run with the configured learning rate. Console output,
including every printed optimization record, is mirrored to `training.log`
beside the checkpoint. A fresh run replaces that log; a resumed run appends to
it.

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
loads the saved state dictionary, and evaluates the trajectory. By default it
opens a folder-selection dialog rooted at `saved_models`; select the folder
that contains `mlp_state_dict.pt`. Use `--no-folder-dialog` to load
`saved_models/mlp/mlp_state_dict.pt` without prompting. The benchmark reads
the checkpoint sidecar automatically. For a legacy checkpoint without a
sidecar, it falls back to `experiment_parameters.py`; the source TOML can be
supplied explicitly with `--sequence`, and a checkpoint file with
`--model-path`. Either explicit option takes precedence over the dialog:

```powershell
python -m benchmark.benchmark_mlp --model-path runs/experiment_a.pt
python -m benchmark.benchmark_mlp --sequence training_sequence.toml
```

Every benchmark saves `trajectory.png` in the loaded checkpoint's folder,
including with `--no-show`. Repeated benchmarks replace that image.
`--output` optionally saves an additional copy at another path.

The default console report includes the mean absolute `sigma_z` expectation
error for sparse HEOM versus Lindbladian and for MLP versus sparse HEOM. It
also reports the mean absolute MLP-versus-sparse error over every complex
component and requested time in the full normalized HEOM state.

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

### Numerical restart diagnostic for MLP ADOs

This is separate from the direct-MLP trajectory benchmark: after querying the
MLP at one time, **all subsequent dynamics are numerical HEOM propagation**.
Run:

```powershell
python -m benchmark.benchmark_mlp_restart
```

By default, select a model folder in the dialog and run both restart modes:

- **A:** initialize every tier from the MLP prediction.
- **B:** initialize tiers `0..lower_tier_cutoff` from the numerical reference
  and all higher tiers from the MLP. The cutoff is inclusive and defaults to
  `0`, so only the RDM is numerical; all auxiliary tiers are tested.

The restart time defaults to the saved training end. The continuation lasts
one training-interval length by default (a model trained on `[10, 12]` is
continued on `[12, 14]`). Override these settings, for example, with:

```powershell
python -m benchmark.benchmark_mlp_restart --mode A --restart-time 12 --t-stop 20
python -m benchmark.benchmark_mlp_restart --mode B --lower-tier-cutoff 5 --t-stop 20
```

The uninterrupted reference starts at the checkpoint's stored full HEOM
initial state at its training start `t_0`, including nonzero ADOs for later
training windows. Reference and restart solves use the same normalized,
hard-truncated Liouvillian, cutoff `L`, and saved integration tolerances.
Hybridization copies whole ADO matrices without additional normalization or
projection. Setting `--lower-tier-cutoff` equal to `L` in mode B gives an
all-numerical control: its discrepancy estimates the numerical restart error.
This reference is converged only to the chosen solver tolerance and HEOM
cutoff; it is not an independent check of hierarchy-truncation accuracy.

Each mode has its own `sigma_z` panel with `E_z` and `E_H` to two significant
figures, plus the common `L` and restart time. These are mean absolute errors
against the uninterrupted reference **over the continuation samples only**,
including the restart point. `E_H` averages absolute complex-component errors
across the full normalized HEOM state, including tier 0; it is not a relative
or tier-balanced error. The direct MLP errors at the restart point are also
printed. Small subsequent `E_z` alone does not establish accuracy of every
higher-tier ADO; many small components can also dilute the global `E_H`.

The plot is always saved as `restart_trajectory.png` beside the checkpoint
(replaced on subsequent runs), leaving `trajectory.png` unchanged. Use
`--no-show` for noninteractive runs and `--output` for an additional plot copy.
`--model-path` accepts a folder or checkpoint file and bypasses the dialog;
`--no-folder-dialog` uses `saved_models/mlp`. `--n-times` overrides the saved
sample count, and `--device cpu` allows loading a GPU-trained model on CPU.
Saved metadata takes precedence over current experiment settings. A legacy
checkpoint without metadata requires its original `--sequence` TOML; supplying
a sequence also resolves its named checkpoint without a dialog.

The neural pipeline requires PyTorch. The benchmarks additionally use QuTiP
and Matplotlib.
