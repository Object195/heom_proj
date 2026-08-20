"""Scientific shape, ordering, residual, and autograd checks for the MLP."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from experiment_parameters import MLP
from heom import q_func
from heom.heom_rep import heom_state
from model import (
    EpochRecord,
    HEOMMLP,
    HEOMPINNLoss,
    TrainingConfig,
    column_vector_to_matrix,
    conjugate_ado_permutation,
    hierarchy_coordinates,
    matrix_to_column_vector,
    solve_mlp,
    state_and_time_derivative,
    train_mlp,
)
from model.train_mlp_model import (
    LiveLossPlot,
    build_argument_parser,
    build_optimizer,
    load_saved_model,
)


def make_hierarchy(depth=2):
    h_system = np.array([[0.5, 0.1], [0.1, -0.5]], dtype=np.complex128)
    h_coupling = np.diag([1.0, -1.0]).astype(np.complex128)
    return heom_state(
        K=0,
        L=depth,
        H_s=h_system,
        H_c=h_coupling,
        C_list=np.array([0.2 + 0.1j]),
        gamma_list=np.array([0.3 + 1.0j]),
    )


def make_rho0():
    return np.diag([1.0, 0.0]).astype(np.complex128)


def test_coordinates_and_partner_permutation_follow_bfs_order():
    hierarchy = make_hierarchy(depth=2)
    assert hierarchy.idx_to_node == [
        ((0,), (0,)),
        ((1,), (0,)),
        ((0,), (1,)),
        ((2,), (0,)),
        ((1,), (1,)),
        ((0,), (2,)),
    ]
    np.testing.assert_array_equal(
        conjugate_ado_permutation(hierarchy),
        np.array([0, 2, 1, 5, 4, 3]),
    )
    np.testing.assert_allclose(
        hierarchy_coordinates(hierarchy),
        np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [0.0, 0.5],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        ),
    )


def test_coordinate_minibatch_has_section_three_shape():
    hierarchy = make_hierarchy()
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(4,),
        rho0=make_rho0(),
        t_start=2.0,
        t_stop=6.0,
    )
    times = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)
    inputs = model.coordinate_inputs(times)
    assert inputs.shape == (3, hierarchy.nADO, 2 * hierarchy.K + 3)
    torch.testing.assert_close(
        inputs[:, :, -1],
        inputs.new_tensor([-1.0, 0.0, 1.0])[:, None].expand(
            -1, hierarchy.nADO
        ),
    )


def test_physical_time_derivative_includes_normalization_chain_rule():
    hierarchy = make_hierarchy(depth=1)
    short_interval = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=2.0,
    )
    long_interval = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=10.0,
    )
    long_interval.load_state_dict(short_interval.state_dict())

    short_state, short_derivative = state_and_time_derivative(
        short_interval,
        torch.tensor([1.0], dtype=torch.float64),
    )
    long_state, long_derivative = state_and_time_derivative(
        long_interval,
        torch.tensor([5.0], dtype=torch.float64),
    )
    torch.testing.assert_close(short_state, long_state)
    torch.testing.assert_close(short_derivative, 5.0 * long_derivative)


def test_column_major_vectorization_round_trip():
    vector = torch.tensor(
        [[1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j, 7.0 + 8.0j]],
        dtype=torch.complex128,
    )
    matrix = column_vector_to_matrix(vector, 2)
    expected = torch.tensor(
        [[[1.0 + 2.0j, 5.0 + 6.0j], [3.0 + 4.0j, 7.0 + 8.0j]]],
        dtype=torch.complex128,
    )
    torch.testing.assert_close(matrix, expected)
    torch.testing.assert_close(matrix_to_column_vector(matrix), vector)


def test_symmetrization_enforces_adjoint_partners():
    hierarchy = make_hierarchy()
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=1.0,
    )
    raw = torch.randn(3, hierarchy.nADO, 8, dtype=torch.float64)
    real_state, imaginary_state = model.symmetrize_raw(raw)
    ados = torch.complex(real_state, imaginary_state).reshape(
        3, hierarchy.nADO, hierarchy.system_size
    )
    matrices = column_vector_to_matrix(ados, 2)
    partners = torch.as_tensor(conjugate_ado_permutation(hierarchy))
    torch.testing.assert_close(
        matrices[:, partners],
        matrices.transpose(-2, -1).conj(),
        rtol=0.0,
        atol=0.0,
    )


def test_requested_liouvillian_and_sparse_rhs_match_scipy():
    hierarchy = make_hierarchy(depth=1)
    with patch.object(
        hierarchy,
        "build_Liouvillian",
        wraps=hierarchy.build_Liouvillian,
    ) as constructor:
        objective = HEOMPINNLoss(hierarchy)
    constructor.assert_called_once_with(
        markovian_terminator=False,
        normalized=True,
    )

    rng = np.random.default_rng(4)
    state = rng.normal(size=hierarchy.liouvillian.shape[0]) + 1j * rng.normal(
        size=hierarchy.liouvillian.shape[0]
    )
    expected = q_func.state_to_real(hierarchy.liouvillian @ state)
    actual = objective.rhs(
        torch.as_tensor(q_func.state_to_real(state)[None], dtype=torch.float64)
    )
    np.testing.assert_allclose(actual.numpy()[0], expected, atol=1e-13)


def test_output_enforces_initial_state_and_root_trace():
    hierarchy = make_hierarchy(depth=1)
    rho0 = make_rho0()
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=rho0,
        t_start=2.0,
        t_stop=6.0,
    )
    times = torch.tensor([2.0, 3.0, 6.0], dtype=torch.float64)
    states = model(times)
    torch.testing.assert_close(states[0], model.initial_state)

    correction = model.state_correction(times)
    real_correction, imaginary_correction = correction.split(
        model.state_size, dim=-1
    )
    real_trace = real_correction.index_select(
        1, model.root_diagonal_indices
    ).sum(dim=1)
    imaginary_trace = imaginary_correction.index_select(
        1, model.root_diagonal_indices
    ).sum(dim=1)
    torch.testing.assert_close(real_trace, torch.zeros_like(real_trace))
    torch.testing.assert_close(
        imaginary_trace, torch.zeros_like(imaginary_trace)
    )

    raw_real, raw_imaginary = model.symmetrize_raw(model.raw_output(times))
    torch.testing.assert_close(
        real_correction[:, model.system_size :],
        raw_real[:, model.system_size :],
    )
    torch.testing.assert_close(
        imaginary_correction[:, model.system_size :],
        raw_imaginary[:, model.system_size :],
    )

    correction_ados = torch.complex(
        real_correction, imaginary_correction
    ).reshape(times.numel(), hierarchy.nADO, hierarchy.system_size)
    correction_matrices = column_vector_to_matrix(correction_ados, 2)
    partners = torch.as_tensor(conjugate_ado_permutation(hierarchy))
    torch.testing.assert_close(
        correction_matrices[:, partners],
        correction_matrices.transpose(-2, -1).conj(),
    )

    root_traces = torch.diagonal(
        model.root_density_matrices(times), dim1=-2, dim2=-1
    ).sum(dim=-1)
    torch.testing.assert_close(root_traces, torch.ones_like(root_traces))

    _, initial_derivative = state_and_time_derivative(model, times[:1])
    expected_initial_derivative = model.state_correction(times[:1]) / (
        model.t_stop - model.t_start
    )
    torch.testing.assert_close(
        initial_derivative, expected_initial_derivative
    )


def test_jvp_and_loss_backpropagate_to_every_parameter():
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    rho0 = make_rho0()
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(8, 8),
        rho0=rho0,
        t_start=0.0,
        t_stop=4.0,
    )
    objective = HEOMPINNLoss(hierarchy, liouvillian=liouvillian)
    times = torch.linspace(0.25, 3.75, 4, dtype=torch.float64)
    states, derivatives = state_and_time_derivative(model, times)
    finite_difference = (
        model(times + 1e-6) - model(times - 1e-6)
    ) / 2e-6
    torch.testing.assert_close(
        derivatives,
        finite_difference,
        rtol=1e-7,
        atol=1e-8,
    )

    loss = objective(model, times)
    expected_loss = (
        derivatives - objective.rhs(states)
    ).square().sum() / (model.state_size * times.numel())
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    assert states.shape == (4, 2 * model.state_size)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_partial_minibatches_match_the_full_objective():
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=1.0,
    )
    objective = HEOMPINNLoss(hierarchy, liouvillian=liouvillian)
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    full = objective(model, times)
    first = objective(model, times[:2])
    second = objective(model, times[2:])
    combined = (2 * first + 3 * second) / 5
    torch.testing.assert_close(full, combined)


def test_short_training_and_solution_layout():
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    rho0 = make_rho0()
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(6,),
        rho0=rho0,
        t_start=0.0,
        t_stop=0.2,
    )
    objective = HEOMPINNLoss(hierarchy, liouvillian=liouvillian)
    result = train_mlp(
        model,
        objective,
        TrainingConfig(
            t_start=0.0,
            t_stop=0.2,
            epochs=2,
            collocation_points=4,
            batch_size=2,
        ),
        verbose=False,
    )
    solution = solve_mlp(model, np.linspace(0.0, 0.2, 3), batch_size=2)
    assert len(result.history) == 2
    assert np.isfinite(result.final.loss)
    np.testing.assert_allclose(solution.t, np.linspace(0.0, 0.2, 3))
    assert solution.y.shape == (model.state_size, 3)
    np.testing.assert_allclose(
        solution.primary_ados,
        solution.primary_ados.conj().transpose(0, 2, 1),
        atol=1e-13,
    )


def test_lbfgs_uses_fixed_full_batch_and_strong_wolfe():
    arguments = build_argument_parser().parse_args(
        ["--resume", "--plot-loss"]
    )
    assert arguments.resume
    assert arguments.plot_loss
    assert arguments.optimizer == "lbfgs"

    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=0.2,
        dtype=torch.float64,
    )
    objective = HEOMPINNLoss(
        hierarchy,
        liouvillian=liouvillian,
        dtype=torch.float64,
    )
    optimizer = build_optimizer(model, "lbfgs")
    config = TrainingConfig(
        t_start=0.0,
        t_stop=0.2,
        epochs=1,
        collocation_points=5,
        batch_size=2,
        resample_each_epoch=True,
    )

    with patch.object(objective, "forward", wraps=objective.forward) as call:
        result = train_mlp(
            model,
            objective,
            config,
            optimizer=optimizer,
            verbose=False,
        )

    expected_times = torch.linspace(0.0, 0.2, 5, dtype=torch.float64)
    assert isinstance(optimizer, torch.optim.LBFGS)
    assert optimizer.defaults["line_search_fn"] == "strong_wolfe"
    assert optimizer.defaults["tolerance_grad"] == MLP.lbfgs_tolerance_grad
    assert (
        optimizer.defaults["tolerance_change"]
        == MLP.lbfgs_tolerance_change
    )
    assert call.call_count >= 1
    for invocation in call.call_args_list:
        torch.testing.assert_close(invocation.args[1], expected_times)
    assert np.isfinite(result.final.loss)


def test_saved_model_can_be_loaded_for_additional_training():
    hierarchy = make_hierarchy(depth=1)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=1.0,
    )
    expected = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "mlp_state_dict.pt"
        torch.save(model.state_dict(), path)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        load_saved_model(model, path, torch.device("cpu"))

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_live_loss_plot_uses_log_scale():
    import matplotlib

    matplotlib.use("Agg", force=True)
    plotter = LiveLossPlot(update_every=100, final_epoch=201)
    plotter(EpochRecord(epoch=1, loss=10.0))
    plotter(EpochRecord(epoch=2, loss=1.0))
    plotter(EpochRecord(epoch=100, loss=1.0))
    plotter(EpochRecord(epoch=101, loss=0.5))
    plotter(EpochRecord(epoch=201, loss=0.0))
    plotter.finish(show=False)

    assert plotter.axis.get_yscale() == "log"
    np.testing.assert_array_equal(plotter.line.get_xdata(), [1, 100, 201])
    assert np.all(np.asarray(plotter.line.get_ydata()) > 0.0)
    plotter.plt.close(plotter.figure)
